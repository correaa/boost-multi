// Copyright 2024 Alfredo A. Correa

#if not defined(__clang_major__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Walloc-zero"
#endif

#include <boost/multi_array.hpp>
#include <boost/test/unit_test.hpp>

#include <multi/array.hpp>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(concepts) {

	using BMA = boost::multi_array<int, 2>;

	(void)boost::multi_array_concepts::ConstMultiArrayConcept<BMA, 2>();
	(void)boost::multi_array_concepts::MutableMultiArrayConcept<BMA, 2>();

	using MA = multi::array<int, 2>;

	(void)boost::multi_array_concepts::ConstMultiArrayConcept<MA, 2>();
	(void)boost::multi_array_concepts::MutableMultiArrayConcept<MA, 2>();
}

#if defined(__clang_major__)
#pragma GCC diagnostic pop
#endif
