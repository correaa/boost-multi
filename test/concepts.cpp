// Copyright 2022-2025 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wsign-conversion"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

// NOLINTBEGIN(misc-include-cleaner)
#include <boost/concept/assert.hpp>              // for BOOST_CONCEPT_ASSERT  // IWYU pragma: keep
#include <boost/concept_check.hpp>               // for Assignable, CopyCons...  // IWYU pragma: keep
#include <boost/iterator/iterator_facade.hpp>    // for operator-  // IWYU pragma: keep
#include <boost/multi_array.hpp>                 // for multi_array  // IWYU pragma: keep
#include <boost/multi_array/concept_checks.hpp>  // for ConstMultiArrayConcept  // IWYU pragma: keep
// NOLINTEND(misc-include-cleaner)

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <boost/multi/array.hpp>  // for operator!=, implicit...

#include <type_traits>  // for is_same_v, is_convertib...

namespace multi = boost::multi;

#include <boost/core/lightweight_test.hpp>

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	// BOOST_AUTO_TEST_CASE(convertibles_1D)
	{
		using NDArray = multi::array<double, 1>;

		NDArray const nda;
		(void)nda;

		static_assert(std::is_same_v<typename NDArray::element_type, typename multi::array<double, 1>::value_type>);
		static_assert(std::is_same_v<typename NDArray::element_ref, typename multi::array<double, 1>::reference>);

		using NDRef = typename NDArray::ref;

		static_assert(std::is_convertible_v<NDRef, NDArray>);

#ifndef __NVCC__
		static_assert(std::is_convertible_v<typename NDRef::reference, typename NDRef::value_type>);
		static_assert(std::is_convertible_v<typename NDRef::const_reference, typename NDRef::value_type>);
#else
		static_assert(std::is_convertible<typename NDRef::reference, typename NDRef::value_type>::value);
		static_assert(std::is_convertible<typename NDRef::const_reference, typename NDRef::value_type>::value);
#endif

		static_assert(std::is_same_v<typename NDRef::element_type, typename multi::array<double, 1>::value_type>);
		static_assert(std::is_same_v<typename NDRef::element_ref, typename multi::array<double, 1>::reference>);
	}

	// BOOST_AUTO_TEST_CASE(convertibles_2D)
	{
		using NDArray = multi::array<double, 2>;

		NDArray const nda;
		(void)nda;

		static_assert(std::is_same_v<typename NDArray::element_type, typename multi::array<double, 1>::value_type>);
		static_assert(std::is_same_v<typename NDArray::element_ref, typename multi::array<double, 1>::reference>);

		using NDRef = typename NDArray::ref;

		static_assert(std::is_convertible_v<NDRef, NDArray>);

		static_assert(std::is_convertible_v<typename NDRef::reference, typename NDRef::value_type>);
		static_assert(std::is_convertible_v<typename NDRef::const_reference, typename NDRef::value_type>);

		static_assert(std::is_same_v<typename NDRef::element_type, typename multi::array<double, 1>::value_type>);
		static_assert(std::is_same_v<typename NDRef::element_ref, typename multi::array<double, 1>::reference>);
	}

	// BOOST_AUTO_TEST_CASE(convertibles_3D)
	{
		using NDArray = multi::array<double, 3>;

		NDArray const nda;
		(void)nda;

		static_assert(std::is_same_v<typename NDArray::element_type, typename multi::array<double, 1>::value_type>);
		static_assert(std::is_same_v<typename NDArray::element_ref, typename multi::array<double, 1>::reference>);

		using NDRef = typename NDArray::ref;

		static_assert(std::is_convertible_v<NDRef, NDArray>);

		// multi::what<typename NDRef::reference>();

		static_assert(std::is_convertible_v<typename NDRef::reference, typename NDRef::value_type>);
		static_assert(std::is_convertible_v<typename NDRef::const_reference, typename NDRef::value_type>);

		static_assert(std::is_same_v<typename NDRef::element_type, typename multi::array<double, 1>::value_type>);
		static_assert(std::is_same_v<typename NDRef::element_ref, typename multi::array<double, 1>::reference>);
	}

	return boost::report_errors();
}
