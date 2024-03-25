// Copyright 2024 Alfredo A. Correa

#include <boost/test/unit_test.hpp>

#include <multi/array.hpp>

#include <chrono>
#include <complex>
#include <iostream>
#include <thread>

#if defined(TBB_FOUND) || (defined(__GNUC__) && !defined(__clang__) && !defined(__NVCOMPILER) && (__GLIBCXX__ >= 20190502))
#if !defined(__NVCC__) && !(defined(__clang__) && defined(__CUDA__))
#if !defined(PSTL_USE_PARALLEL_POLICIES) || !(PSTL_USE_PARALLEL_POLICIES==0)
#include <execution>
#endif
#endif
#endif

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_par_construct_1d) {
	multi::static_array<double, 1> const arr(multi::extensions_t<1>{multi::iextension{10}}, 1.0);
	//  multi::static_array<double, 1> arr(multi::array<double, 1>::extensions_type{10}, 1.0);
	BOOST_REQUIRE( size(arr) == 10 );
	BOOST_REQUIRE( arr[1] == 1.0 );

#if defined(TBB_FOUND) || (defined(__GNUC__) && !defined(__clang__) && !defined(__NVCOMPILER) && (__GLIBCXX__ >= 20190502))
#if !defined(__NVCC__) && !(defined(__clang__) && defined(__CUDA__))
#if !defined(PSTL_USE_PARALLEL_POLICIES) || !(PSTL_USE_PARALLEL_POLICIES==0)
	multi::static_array<double, 1> const arr2(std::execution::par, arr);

	BOOST_REQUIRE( arr2 == arr );
#endif
#endif
#endif
}

BOOST_AUTO_TEST_CASE(copy_par_1d) {
	multi::array<double, 1> const arr(1000000, 1.0);
	BOOST_REQUIRE( size(arr) == 1000000 );
	BOOST_REQUIRE( arr[1] == 1.0 );

#if defined(TBB_FOUND) || (defined(__GNUC__) && !defined(__clang__) && !defined(__NVCOMPILER) && (__GLIBCXX__ >= 20190502))
#if !defined(__NVCC__) && !(defined(__clang__) && defined(__CUDA__))
#if !defined(PSTL_USE_PARALLEL_POLICIES) || !(PSTL_USE_PARALLEL_POLICIES==0)
#if defined(__cpp_lib_execution) && (__cpp_lib_execution >= 201603L)
	multi::array<double, 1> arr2(arr.extensions());

	std::copy(std::execution::par, arr.begin(), arr.end(), arr2.begin());

	BOOST_REQUIRE( arr2 == arr );
#endif
#endif
#endif
#endif
}

class watch  // NOLINT(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)  // NOSONAR
: private std::chrono::high_resolution_clock {
	std::string label_;
	time_point  start_ = now();

 public:
	explicit watch(std::string label) : label_{std::move(label)} {}

	~watch() {
		std::cerr << label_ << ": " << std::chrono::duration<double>(now() - start_).count() << " sec" << std::endl;
	}
};

class slow_assign {
	double val_;

 public:
	constexpr explicit slow_assign(double const& vv) noexcept : val_{vv} {}
	~slow_assign() = default;

	slow_assign(slow_assign&& other) noexcept = default;

	slow_assign(slow_assign const& other) : val_{other.val_} {
		using namespace std::chrono_literals;
		std::this_thread::sleep_for(10ms);
	}
	auto operator=(slow_assign const& other) -> slow_assign& {
		if(this == &other) {return *this;}
		val_ = other.val_;
		using namespace std::chrono_literals;
		std::this_thread::sleep_for(10ms);
		return *this;
	}
	auto operator=(slow_assign&& other) noexcept -> slow_assign& = default;

	auto operator==(slow_assign const& other) const noexcept {return val_ == other.val_;}
	auto operator!=(slow_assign const& other) const noexcept {return val_ != other.val_;}
};

using T = slow_assign;

auto const nelem = 80;

#if defined(TBB_FOUND) || (defined(__GNUC__) && !defined(__clang__) && !defined(__NVCOMPILER) && (__GLIBCXX__ >= 20190502))
#if !defined(__NVCC__) && !(defined(__clang__) && defined(__CUDA__))
#if !defined(PSTL_USE_PARALLEL_POLICIES) || !(PSTL_USE_PARALLEL_POLICIES==0)
#if defined(__cpp_lib_execution) && (__cpp_lib_execution >= 201603L)

BOOST_AUTO_TEST_CASE(timing_copy_par_1d) {
	T const val { 1.0};
	T const val2{99.9};

	multi::array<T, 1> const arr(nelem, val);
	BOOST_REQUIRE( size(arr) == nelem );
	BOOST_REQUIRE( arr[1] == val );

	{
		multi::array<T, 1> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("normal copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy(arr.begin(), arr.end(), arr2.begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
	{
		multi::array<T, 1> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("par copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy(std::execution::par, arr.begin(), arr.end(), arr2.begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
}

BOOST_AUTO_TEST_CASE(timing_copy_par_2d_warm) {
	T const val { 1.0};

	multi::array<T, 2> const arr({8, nelem/8}, val);
	BOOST_REQUIRE( arr.num_elements() == nelem );
	BOOST_REQUIRE( arr[1][1] == val );
}

BOOST_AUTO_TEST_CASE(timing_copy_par_2d) {
	T const val { 1.0};
	T const val2{99.9};

	multi::array<T, 2> const arr({8, nelem/8}, val);
	BOOST_REQUIRE( arr.num_elements() == nelem );
	BOOST_REQUIRE( arr[1][1] == val );

	{
		multi::array<T, 2> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("normal copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy(arr.begin(), arr.end(), arr2.begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
	{
		multi::array<T, 2> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("par copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy(std::execution::par, arr.begin(), arr.end(), arr2.begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
}

BOOST_AUTO_TEST_CASE(timing_copy_par_2d_skinny) {
	T const val { 1.0};
	T const val2{99.9};

	multi::array<T, 2> const arr({4, nelem/4}, val);
	BOOST_REQUIRE( arr.num_elements() == nelem );
	BOOST_REQUIRE( arr[1][1] == val );

	{
		multi::array<T, 2> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("normal copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy(arr.begin(), arr.end(), arr2.begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
	{
		multi::array<T, 2> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("par copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy(std::execution::par, arr.begin(), arr.end(), arr2.begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
}

BOOST_AUTO_TEST_CASE(timing_copy_par_2d_ultra_skinny) {
	T const val { 1.0};
	T const val2{99.9};

	multi::array<T, 2> const arr({2, nelem/2}, val);
	BOOST_REQUIRE( arr.num_elements() == nelem );
	BOOST_REQUIRE( arr[1][1] == val );

	{
		multi::array<T, 2> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("normal copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy(arr.begin(), arr.end(), arr2.begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
	{
		multi::array<T, 2> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("par copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy(std::execution::par, arr.begin(), arr.end(), arr2.begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
	{
		multi::array<T, 2> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("~copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy((~arr).begin(), (~arr).end(), (~arr2).begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
	{
		multi::array<T, 2> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("~par copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy(std::execution::par, (~arr).begin(), (~arr).end(), (~arr2).begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
	{
		multi::array<T, 2> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("elements copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy(arr.elements().begin(), arr.elements().end(), arr2.elements().begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
	{
		multi::array<T, 2> arr2(arr.extensions(), val2);
		BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
		{
			watch const _("par elements copy");  // NOLINT(fuchsia-default-arguments-calls)
			std::copy(std::execution::par, arr.elements().begin(), arr.elements().end(), arr2.elements().begin());
		}
		BOOST_REQUIRE( arr2 == arr );
	}
	{
		{
			watch const _("constructor");  // NOLINT(fuchsia-default-arguments-calls)
			multi::array<T, 2> arr2(arr);  // same as  ...= arr;
			BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
			BOOST_REQUIRE( arr2 == arr );
			arr2.clear();
		}
	}
	{
		{
			watch const _("par constructor");  // NOLINT(fuchsia-default-arguments-calls)
			multi::array<T, 2> const arr2(std::execution::par, arr);
			BOOST_REQUIRE( arr2.num_elements() == arr.num_elements() );
			BOOST_REQUIRE( arr2 == arr );
		}
	}
}
#endif
#endif
#endif
#endif
