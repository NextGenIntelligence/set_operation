#pragma once

#include "omp_set_operation.h"
#include <thrust/set_operations.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/system/omp/detail/tag.h>

namespace set_intersection_detail
{


struct serial_set_intersection
{
  template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  inline OutputIterator operator()(InputIterator1 first1, InputIterator1 last1,
                                   InputIterator2 first2, InputIterator2 last2,
                                   OutputIterator result,
                                   Compare comp)
  {
    thrust::cpp::tag seq;
    return thrust::set_intersection(seq, first1, last1, first2, last2, result, comp);
  }


  template<typename InputIterator1, typename InputIterator2, typename Compare>
  inline __device__
    typename thrust::iterator_difference<InputIterator1>::type
      count(InputIterator1 first1, InputIterator1 last1,
            InputIterator2 first2, InputIterator2 last2,
            Compare comp)
  {
    thrust::cpp::tag seq;
    thrust::discard_iterator<> discard;
    return thrust::set_intersection(seq, first1, last1, first2, last2, discard) - discard;
  }
}; // end serial_set_intersection


} // end set_intersection_detail


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  OutputIterator my_set_intersection(InputIterator1 first1, InputIterator1 last1,
                                     InputIterator2 first2, InputIterator2 last2,
                                     OutputIterator result,
                                     Compare comp)
{
  thrust::omp::tag par;
  return ::set_operation(par, first1, last1, first2, last2, result, comp, set_intersection_detail::serial_set_intersection());
}

