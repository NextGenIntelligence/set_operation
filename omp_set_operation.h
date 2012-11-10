#pragma once

#include "balanced_path.h"
#include <thrust/scan.h>
#include <thrust/pair.h>
#include <thrust/system/omp/detail/tag.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/dispatchable.h>
#include <omp.h>
#include <cassert>


namespace set_operation_detail
{

template<typename Size, typename InputIterator1, typename InputIterator2, typename Compare>
  struct find_partition_offsets_functor
{
  Size partition_size;
  InputIterator1 first1;
  InputIterator2 first2;
  Size n1, n2;
  Compare comp;

  find_partition_offsets_functor(Size partition_size,
                                 InputIterator1 first1, InputIterator1 last1,
                                 InputIterator2 first2, InputIterator2 last2,
                                 Compare comp)
    : partition_size(partition_size),
      first1(first1), first2(first2),
      n1(last1 - first1), n2(last2 - first2),
      comp(comp)
  {}

  inline __host__ __device__
  thrust::pair<Size,Size> operator()(Size i) const
  {
    Size diag = thrust::min(n1 + n2, i * partition_size);

    // XXX the correctness of balanced_path depends critically on the ll suffix below
    //     why???
    return balanced_path(first1, n1, first2, n2, diag, 4ll, comp);
  }
};


template<typename Size, typename System, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  OutputIterator find_partition_offsets(thrust::dispatchable<System> &system,
                                        Size num_partitions,
                                        Size partition_size,
                                        InputIterator1 first1, InputIterator1 last1,
                                        InputIterator2 first2, InputIterator2 last2,
                                        OutputIterator result,
                                        Compare comp)
{
  find_partition_offsets_functor<Size,InputIterator1,InputIterator2,Compare> f(partition_size, first1, last1, first2, last2, comp);

  return thrust::transform(system,
                           thrust::counting_iterator<Size>(0),
                           thrust::counting_iterator<Size>(num_partitions),
                           result,
                           f);
}


}


template<typename System, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare, typename SetOperation>
  OutputIterator set_operation(thrust::omp::dispatchable<System> &system,
                               InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2, InputIterator2 last2,
                               OutputIterator result,
                               Compare comp,
                               SetOperation set_op)
{
  typedef typename thrust::iterator_difference<InputIterator1>::type difference;

  const difference n1 = last1 - first1;
  const difference n2 = last2 - first2;
  const difference input_size = n1 + n2;

  // skip empty input
  if(input_size == 0) return result;

  // XXX get the actual number of processors
  const difference num_processors = omp_get_num_procs();
  const difference min_partition_size = 1 << 20;

  // -1 because balanced_path adds a single element to the end of a "starred" partition, increasing its size by one
  const difference maximum_partition_size = thrust::max<difference>(min_partition_size, thrust::detail::util::divide_ri(input_size, num_processors)) - 1;
  const difference num_partitions = thrust::detail::util::divide_ri(input_size, maximum_partition_size);

  // find input partition offsets
  // +1 to handle the end of the input elegantly
  thrust::detail::temporary_array<thrust::pair<difference,difference>, System> input_partition_offsets(0, system, num_partitions + 1);
  set_operation_detail::find_partition_offsets<difference>(system, input_partition_offsets.size(), maximum_partition_size, first1, last1, first2, last2, input_partition_offsets.begin(), comp);

  // find output partition offsets
  // +1 to store the total size of the total
  thrust::detail::temporary_array<difference, System> output_partition_offsets(0, system, num_partitions + 1);

  #pragma omp parallel for
  for(difference partition_idx = 0;
      partition_idx < input_partition_offsets.size() - 1;
      ++partition_idx)
  {
    thrust::pair<difference,difference> partition_begin = input_partition_offsets[partition_idx];
    thrust::pair<difference,difference> partition_end   = input_partition_offsets[partition_idx+1];

    // count the size of the set operation
    output_partition_offsets[partition_idx] = set_op.count(first1 + partition_begin.first,  first1 + partition_end.first,
                                                           first2 + partition_begin.second, first2 + partition_end.second,
                                                           comp);
  }

  // turn the output partition counts into offsets to output partitions
  thrust::exclusive_scan(system, output_partition_offsets.begin(), output_partition_offsets.end(), output_partition_offsets.begin());

  #pragma omp parallel for
  for(difference partition_idx = 0;
      partition_idx < input_partition_offsets.size() - 1;
      ++partition_idx)
  {
    thrust::pair<difference,difference> partition_begin = input_partition_offsets[partition_idx];
    thrust::pair<difference,difference> partition_end   = input_partition_offsets[partition_idx+1];

    // do the set operation
    set_op(first1 + partition_begin.first,  first1 + partition_end.first,
           first2 + partition_begin.second, first2 + partition_end.second,
           result + output_partition_offsets[partition_idx],
           comp);
  }

  return result + output_partition_offsets[num_partitions];
}

