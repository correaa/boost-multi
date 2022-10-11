// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2020-2022 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi FFTW move"
#include<boost/test/unit_test.hpp>

#include "../../../adaptors/fftw.hpp"
#include "../../../array.hpp"

namespace multi = boost::multi;
using complex = std::complex<double>; [[maybe_unused]] complex const I{0, 1};  // NOLINT(readability-identifier-length) imag unit

template<class M> auto power(M const& array) {
	return std::transform_reduce(array.elements().begin(), array.elements().end(), 0., std::plus<>{}, [](auto zee) {return std::norm(zee);});
}

using fftw_fixture = multi::fftw::environment;
BOOST_TEST_GLOBAL_FIXTURE( fftw_fixture );

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_fft_move) {
	#if not defined(__circle_build__)
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		auto const in_copy = in;
	//  auto* const in_base = in.base();

		multi::array<complex, 2> in2(in.extensions());

		in2 = multi::fftw::fft(std::move(in));

		BOOST_REQUIRE( power(in2)/num_elements(in2) - power(in_copy) < 1e-8 );
//		BOOST_REQUIRE( in2.base() == in_base );
//		BOOST_REQUIRE( in.is_empty() );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing
	}
	#endif
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_move) {
	#if not defined(__circle_build__)
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	BOOST_REQUIRE( in[1][1] == 7. - 4.*I );

	{
		auto const in_copy = in;
		auto* const in_base = in.base();
		BOOST_REQUIRE( in_base == in.base() );

		in = multi::fftw::ref(in);

		BOOST_REQUIRE( in == in_copy );
		BOOST_REQUIRE( in_base == in.base() );  // prove no allocation
	}
	#endif
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_transposed) {
	#if not defined(__circle_build__)
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	BOOST_REQUIRE( in[1][1] == 7. - 4.*I );

	{
		auto const in_copy = in;
		auto* const in_base = in.base();
		BOOST_REQUIRE( in_base == in.base() );
		BOOST_REQUIRE( in.size() == 5 );

		in = multi::fftw::ref(in).transposed();

		BOOST_REQUIRE( in.size() == 3 );
		BOOST_REQUIRE( in == in_copy.transposed() );  // prove correctness
		BOOST_REQUIRE( in_base == in.base() );  // prove no allocation
	}
	#endif
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_transposed_naive) {
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	BOOST_REQUIRE( in[1][1] == 7. - 4.*I );

	{
		auto const in_copy = in;
		auto* const in_base = in.base();
		BOOST_REQUIRE( in_base == in.base() );
		BOOST_REQUIRE( in.size() == 5 );

		in = in.transposed();  // this is UB

		BOOST_REQUIRE( in.size() == 3 );
	//	BOOST_REQUIRE( in != in_copy.transposed() );  // prove it is incorrect
		BOOST_REQUIRE( in_base == in.base() );  // prove no allocation
	}
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_transposed_naive_copy) {
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	BOOST_REQUIRE( in[1][1] == 7. - 4.*I );

	{
		auto const in_copy = in;
		auto* const in_base = in.base();
		BOOST_REQUIRE( in_base == in.base() );
		BOOST_REQUIRE( in.size() == 5 );

		in = + in.transposed();

		BOOST_REQUIRE( in.size() == 3 );
		BOOST_REQUIRE( in == in_copy.transposed() );  // prove correctness
		BOOST_REQUIRE( in_base != in.base() );  // prove no allocation
	}
}


BOOST_AUTO_TEST_CASE(fftw_2D_const_range_fft_copy) {
	#if not defined(__circle_build__)
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		auto const in_copy = in;
		auto* const in_base = in.base();

		multi::array<complex, 2> in2 = multi::fftw::fft(in);

		BOOST_REQUIRE( power(in2)/num_elements(in2) - power(in_copy) < 1e-8 );
		BOOST_REQUIRE( in2.base() != in_base );
		BOOST_REQUIRE( not in.is_empty() );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing
	}
	#endif
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_transposed_copyconstruct) {
	#if not defined(__circle_build__)
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		auto const in_copy = in;
		auto* const in_base = in.base();

		multi::array<complex, 2> in2 = multi::fftw::ref(in).transposed();

		BOOST_REQUIRE( in2 == in_copy.transposed() );
		BOOST_REQUIRE( in2.base() != in_base );
		BOOST_REQUIRE( in .base() == in_base );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing
	}
	#endif
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_transposed_moveconstruct) {
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		auto const in_copy = in;
		auto* const in_base = in.base();

		multi::array<complex, 2> in2 = multi::fftw::ref(std::move(in)).transposed();

		BOOST_REQUIRE( in2 == in_copy.transposed() );
		BOOST_REQUIRE( in2.base() == in_base );
		BOOST_REQUIRE( in.is_empty() );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing
	}
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_transposed_moveconstruct_implicit) {
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		auto const in_copy = in;
		auto* const in_base = in.base();

		auto in2 = +multi::fftw::ref(std::move(in)).transposed();

		BOOST_REQUIRE( in2 == in_copy.transposed() );
		#if not defined(__INTEL_COMPILER)  // TODO(correaa) problem with icpc 2022.3.0.8751
		BOOST_REQUIRE( in2.base() == in_base );
		#endif
		BOOST_REQUIRE( in.is_empty() );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing
	}
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_transposed_moveassign_from_temp) {
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		auto const in_copy = in;
		auto* const in_base = in.base();

		multi::array<complex, 2> in2;
		in2 = static_cast<multi::array<complex, 2>>(multi::fftw::ref(std::move(in)).transposed());

		BOOST_REQUIRE( in2 == in_copy.transposed() );
		#if not defined(__INTEL_COMPILER)  // TODO(correaa) problem with icpc 2022.3.0.8751
		BOOST_REQUIRE( in2.base() == in_base );
		#endif
		BOOST_REQUIRE( in.is_empty() );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing
	}
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_transposed_moveassign) {
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		auto const in_copy = in;
		auto* const in_base = in.base();

		multi::array<complex, 2> in2;
		in2 = multi::fftw::ref(std::move(in)).transposed();

		BOOST_REQUIRE( in2 == in_copy.transposed() );
		#if not defined(__INTEL_COMPILER)  // TODO(correaa) problem with icpc 2022.3.0.8751
		BOOST_REQUIRE( in2.base() == in_base );
		#endif
		BOOST_REQUIRE( in.is_empty() );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing
	}
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_transposed_fftwmove) {
	multi::array<complex, 2> in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		auto const in_copy = in;
		auto* const in_base = in.base();

		multi::array<complex, 2> in2;
		in2 = multi::fftw::move(in).transposed();

		BOOST_REQUIRE( in2 == in_copy.transposed() );
		#if not defined(__INTEL_COMPILER)  // TODO(correaa) problem with icpc 2022.3.0.8751
		BOOST_REQUIRE( in2.base() == in_base );
		#endif
		BOOST_REQUIRE( in.is_empty() );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing
	}
}

