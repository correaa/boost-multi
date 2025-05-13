// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi FFT adaptor"

#include <boost/multi/adaptors/fft.hpp>
#include <boost/multi/adaptors/fftw.hpp>
#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

// IWYU pragma: no_include <array>
#include <complex>
// IWYU pragma: no_include <utility>                          // for forward
// IWYU pragma: no_include <tuple>

namespace multi = boost::multi;
using complex   = std::complex<double>;

template<>
constexpr bool multi::force_element_trivial_default_construction<std::complex<double>> = true;

namespace {
template<class T>
__attribute__((always_inline)) inline void DoNotOptimize(T const& value) {  // NOLINT(readability-identifier-naming) consistency with Google benchmark
	asm volatile("" : "+m"(const_cast<T&>(value)));                         // NOLINT(hicpp-no-assembler,cppcoreguidelines-pro-type-const-cast) hack
}
}  // end namespace

namespace {
void zip_iterator_test(multi::array<complex, 2> const& in_cpu) {
	multi::array<complex, 2> fw_cpu_out(in_cpu.extensions());
	auto zit = multi::fftw::io_zip_iterator(
		{true},
		in_cpu.begin(),
		fw_cpu_out.element_moved().begin(),
		multi::fftw::forward
	);

	zit.execute();
	zit+=1;
	auto zit2 = zit;
	for(int i = 1; i != in_cpu.size(); ++i) {  // NOLINT(altera-unroll-loops)
		zit.execute();
		++zit;
	}

	multi::array<complex, 2> const fw_cpu = multi::fft::dft_forward({false, true}, in_cpu);

	BOOST_TEST( fw_cpu_out == fw_cpu );
}
}  // end namespace

auto main() -> int {  // NOLINT(bugprone-exception-escape,readability-function-cognitive-complexity)
	complex const I{0.0, 1.0};  // NOLINT(readability-identifier-length)

	auto const in_cpu = multi::array<complex, 2>{
		{ 1.0 + 2.0 * I,  9.0 - 1.0 * I,  2.0 + 4.0 * I},
		{ 3.0 + 3.0 * I,  7.0 - 4.0 * I,  1.0 + 9.0 * I},
		{ 4.0 + 1.0 * I,  5.0 + 3.0 * I,  2.0 + 4.0 * I},
		{ 3.0 - 1.0 * I,  8.0 + 7.0 * I,  2.0 + 1.0 * I},
		{31.0 - 1.0 * I, 18.0 + 7.0 * I, 2.0 + 10.0 * I},
	};

	auto fw_cpu = multi::array<complex, 2>(extensions(in_cpu));
	multi::fftw::dft_forward({true, true}, in_cpu, fw_cpu);

	BOOST_TEST( fw_cpu[3][2].real() != 0.0 );
	BOOST_TEST( fw_cpu[3][2].imag() != 0.0 );

	// check properties
	{
		auto const& dft = multi::fft::dft({true, true}, in_cpu, multi::fft::forward);

		BOOST_TEST( dft.extensions() == in_cpu.extensions() );
		BOOST_TEST( (*dft.begin()).size() == (*in_cpu.begin()).size() );
		BOOST_TEST( (*dft.begin()).extensions() == (*in_cpu.begin()).extensions() );
	}
	// assignment with right size
	{
		multi::array<complex, 2> fw_cpu_out(in_cpu.extensions());
		complex* const           persistent_base = fw_cpu_out.base();

		fw_cpu_out = multi::fft::dft({true, true}, in_cpu, multi::fft::forward);

		BOOST_TEST( fw_cpu_out == fw_cpu );
		BOOST_TEST( persistent_base == fw_cpu_out.base() );
	}
	// assignment with incorrect size (need reallocation)
	{
		multi::array<complex, 2> fw_cpu_out({2, 2});

		fw_cpu_out = multi::fft::dft({true, true}, in_cpu, multi::fft::forward);

		BOOST_TEST( fw_cpu_out == fw_cpu );
	}
	// assignment to empty
	{
		multi::array<complex, 2> fw_cpu_out;

		fw_cpu_out = multi::fft::dft({true, true}, in_cpu, multi::fft::forward);

		BOOST_TEST( fw_cpu_out == fw_cpu );
	}
	// constructor
	{
		multi::array<complex, 2> const fw_cpu_out = multi::fft::dft({true, true}, in_cpu, multi::fft::forward);
		BOOST_TEST( fw_cpu_out == fw_cpu );
	}
	// check properties
	{
		auto const& dft = multi::fft::dft({true, true}, in_cpu, multi::fft::forward);

		BOOST_TEST( dft.extensions() == in_cpu.extensions() );
		BOOST_TEST( (*dft.begin()).size() == (*in_cpu.begin()).size() );
		BOOST_TEST( (*dft.begin()).extensions() == (*in_cpu.begin()).extensions() );
	}
	// assignment with right size
	{
		multi::array<complex, 2> fw_cpu_out(in_cpu.extensions());
		complex* const           persistent_base = fw_cpu_out.base();

		fw_cpu_out = multi::fft::dft({true, true}, in_cpu, multi::fft::forward);

		BOOST_TEST( fw_cpu_out == fw_cpu );
		BOOST_TEST( persistent_base == fw_cpu_out.base() );
	}
	// assignment with incorrect size (need reallocation)
	{
		multi::array<complex, 2> fw_cpu_out({2, 2});

		fw_cpu_out = multi::fft::dft({true, true}, in_cpu, multi::fft::forward);

		BOOST_TEST( fw_cpu_out == fw_cpu );
	}
	// assignment to empty
	{
		multi::array<complex, 2> fw_cpu_out;

		fw_cpu_out = multi::fft::dft({true, true}, in_cpu, multi::fft::forward);

		BOOST_TEST( fw_cpu_out == fw_cpu );
	}
	// constructor
	{
		multi::array<complex, 2> const fw_cpu_out = multi::fft::dft({true, true}, in_cpu, multi::fft::forward);
		BOOST_TEST( fw_cpu_out == fw_cpu );
	}
	// constructor forward
	{
		multi::array<complex, 2> const fw_cpu_out = multi::fft::dft_forward({true, true}, in_cpu);
		BOOST_TEST( fw_cpu_out == fw_cpu );
	}
	// constructor forward default
	{
		multi::array<complex, 2> const fw_cpu_out = multi::fft::dft({true, true}, in_cpu);
		BOOST_TEST( fw_cpu_out == fw_cpu );
	}
	// constructor all default
	{
		multi::array<complex, 2> const fw_cpu_out = multi::fft::dft_all(in_cpu);
		BOOST_TEST( fw_cpu_out == fw_cpu );
	}
	// constructor none
	{
		multi::array<complex, 2> const fw_cpu_out = multi::fft::dft({false, false}, in_cpu);
		BOOST_TEST( fw_cpu_out == in_cpu );
	}
	// constructor none
	{
		multi::array<complex, 2> const fw_cpu_out = multi::fft::dft({}, in_cpu);
		BOOST_TEST( fw_cpu_out == in_cpu );
	}
	{
		auto const fw_cpu_out = +multi::fft::dft({}, in_cpu);
		BOOST_TEST( fw_cpu_out == in_cpu );
	}
	// constructor none
	{
		multi::array<complex, 2> const fw_cpu_out = multi::fft::dft({}, in_cpu());
		BOOST_TEST( fw_cpu_out == in_cpu );
	}
	// transposed
	{
		multi::array<complex, 2> const fw_cpu_out = in_cpu.transposed();
		BOOST_TEST( fw_cpu_out == in_cpu.transposed() );
	}

	zip_iterator_test(in_cpu);

	return boost::report_errors();
}
