// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// © Alfredo A. Correa 2020-2021

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi FFTW transpose"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

//#include "../../../adaptors/../complex.hpp"
#include "../../../adaptors/fftw.hpp"
#include "../../../array.hpp"

#include<chrono>
#include<iostream>
#include<random>
//#include<thrust/complex.h> // TODO(correaa) make lib work with thrust complex

namespace {

namespace multi = boost::multi;
namespace fftw = multi::fftw;

using complex = std::complex<double>; MAYBE_UNUSED complex const I{0, 1};

template<class M> auto power(M const& m)->decltype(std::norm(m)) {return std::norm(m);}

template<class M, DELETE((M::rank_v < 1))> auto power(M const& m) {
	return accumulate(begin(m), end(m), 0., [](auto const& a, auto const& b){return a + power(b);});
}

struct sum_power{
	template<class A, class B> auto operator()(A const& a, B const& b) const{return a+power(b);}
};

MAYBE_UNUSED constexpr int N = 16;

} // end anonymous namespace

class watch : private std::chrono::high_resolution_clock{
	std::string label;
	time_point start = now();

 public:
	explicit watch(std::string label) : label{std::move(label)} {}
	watch(watch const&) = delete;
	watch(watch&&) = default;
	auto operator=(watch const&) = delete;
	auto operator=(watch&&) -> watch& = default; // NOLINT(fuchsia-trailing-return):
	~watch(){
		std::cerr<< label<<": "<< std::chrono::duration<double>(now() - start).count() <<" sec"<<std::endl;
	}
};

template<class T> struct randomizer{
	template<class M> void operator()(M&& m) const {for(auto&& e : m) {operator()(e);}}
	void operator()(T& e) const{ // NOLINT(runtime/references) : passing by reference
		static std::random_device r; static std::mt19937 g{r()}; static std::normal_distribution<T> d;
		e = d(g);
	}
};

template<class T> struct randomizer<std::complex<T>>{
	template<class M> void operator()(M&& m) const {for(auto&& e : m) {operator()(e);}}
	void operator()(std::complex<T>& e) const{ // NOLINT(runtime/references) : passing by reference
		static std::random_device r; static std::mt19937 g{r()}; static std::normal_distribution<T> d;
		e = std::complex<T>(d(g), d(g));
	}
};

struct fftw_fixture : fftw::environment{
//	void setup(){}
//	void teardown(){}//fftw_cleanup();}
};

BOOST_TEST_GLOBAL_FIXTURE( fftw_fixture );

BOOST_AUTO_TEST_CASE(fftw_3D) {
	using complex = std::complex<double>;  // TODO(correaa) make it work with thrust
	multi::array<complex, 3> in({10, 10, 10});
	in[2][3][4] = 99.;
	multi::fftw::dft_forward(in);
	BOOST_REQUIRE(in[2][3][4] == 99.);
}

BOOST_AUTO_TEST_CASE(fftw_1D_const) {
	multi::array<complex, 1> const in = {1. + 2.*I, 2. + 3. *I, 4. + 5.*I, 5. + 6.*I};

	auto fwd = multi::fftw::dft(in, fftw::forward); // Fourier[in, FourierParameters -> {1, -1}]
	BOOST_REQUIRE( size(fwd) == size(in) );
	BOOST_REQUIRE( fwd[2] == -2. - 2.*I  );
	BOOST_REQUIRE( in[1]  == +2. + 3.*I  );

	auto bwd = multi::fftw::dft(in, fftw::forward); // InverseFourier[in, FourierParameters -> {-1, -1}]
	BOOST_REQUIRE( bwd[2] == -2. - 2.*I  );
}

BOOST_AUTO_TEST_CASE(fftw_2D_identity_2, *boost::unit_test::tolerance(0.0001)) {
	multi::array<complex, 2> const in = {
		{  1. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{  3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{  4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{  3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	multi::array<complex, 2> out(extensions(in));
	multi::fftw::dft({false, false}, in, out, fftw::forward); // out = in;
	BOOST_REQUIRE( out == in );
}

BOOST_AUTO_TEST_CASE(fftw_2D_identity, *boost::unit_test::tolerance(0.0001)) {
	multi::array<complex, 2> const in = {
		{ 1. + 2.*I, 9. - 1.*I, 2. + 4.*I},
		{ 3. + 3.*I, 7. - 4.*I, 1. + 9.*I},
		{ 4. + 1.*I, 5. + 3.*I, 2. + 4.*I},
		{ 3. - 1.*I, 8. + 7.*I, 2. + 1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	auto fwd = multi::fftw::dft({}, in, fftw::forward);
	BOOST_REQUIRE( fwd == in );
}

BOOST_AUTO_TEST_CASE(fftw_2D, *boost::unit_test::tolerance(0.0001)) {
	multi::array<complex, 2> const in = {
		{  1. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{  3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{  4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{  3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	namespace fftw = multi::fftw;
	auto fwd = fftw::dft_forward(in);
	BOOST_TEST_REQUIRE( fwd[3][1].real() == -19.0455  ); // Fourier[in, FourierParameters -> {1, -1}][[4]][[2]]
	BOOST_TEST_REQUIRE( fwd[3][1].imag() == - 2.22717 );

	multi::array<complex, 1> const in0 = {1. + 2.*I, 9. - 1.*I, 2. + 4.*I};

	BOOST_REQUIRE( fftw::dft_forward(in[0]) == fftw::dft_forward(in0) );
}

BOOST_AUTO_TEST_CASE(fftw_2D_rotated, *boost::unit_test::tolerance(0.0001)) {
	using multi::array;
	array<complex, 2> const in = {
		{  1. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{  3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{  4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{  3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	using multi::fftw::dft_forward;
	auto fwd = dft_forward(in);
	BOOST_REQUIRE(
		dft_forward(rotated(in)[0])
			== dft_forward(array<complex, 1>{1.+2.*I, 3.+3.*I, 4. + 1.*I,  3. - 1.*I, 31. - 1.*I})
	);
	BOOST_REQUIRE( dft_forward(rotated(in)) == rotated(fwd) );
}

BOOST_AUTO_TEST_CASE(fftw_2D_many, *boost::unit_test::tolerance(0.0001)) {
	multi::array<complex, 2> const in = {
		{  1. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{  3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{  4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{  3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	multi::array<complex, 2> out(extensions(in));

	using multi::fftw::dft_forward;

	multi::fftw::dft({fftw::none, fftw::forward}, in, out);
	BOOST_REQUIRE( dft_forward(in[0]) == out[0] );

	multi::fftw::dft({false, true}, rotated(in), rotated(out), fftw::forward);
	BOOST_REQUIRE( dft_forward(rotated(in)[0]) == rotated(out)[0] );

	multi::fftw::dft_forward({false, false}, rotated(in), rotated(out));
	BOOST_REQUIRE( in == out );

	multi::fftw::many_dft(in.begin(), in.end(), out.begin(), fftw::forward);
	BOOST_REQUIRE( dft_forward(in[0]) == out[0] );
}

BOOST_AUTO_TEST_CASE(fftw_1D_const_forward) {
	multi::array<complex, 1> const in = {1. + 2.*I, 2. + 3. *I, 4. + 5.*I, 5. + 6.*I};

	auto fwd = multi::fftw::dft_forward(in); // Fourier[in, FourierParameters -> {1, -1}]
	BOOST_REQUIRE( size(fwd) == size(in) );
	BOOST_REQUIRE( fwd[2] == -2. - 2.*I  );
	BOOST_REQUIRE( in[1]  == +2. + 3.*I  );

	auto bwd = multi::fftw::dft_forward(in); // InverseFourier[in, FourierParameters -> {-1, -1}]
	BOOST_REQUIRE( bwd[2] == -2. - 2.*I  );
}

BOOST_AUTO_TEST_CASE(fftw_1D_const_sign) {
	multi::array<complex, 1> const in = {1. + 2.*I, 2. + 3. *I, 4. + 5.*I, 5. + 6.*I};

	auto const fwd = multi::fftw::dft(in, static_cast<multi::fftw::sign>(+1)); // Fourier[in, FourierParameters -> {1, -1}]
	BOOST_REQUIRE( size(fwd) == size(in) );
	BOOST_REQUIRE( fwd[2] == -2. - 2.*I  );
}

BOOST_AUTO_TEST_CASE(fftw_1D_const_copy_by_false) {
	multi::array<complex, 1> const in = {1. + 2.*I, 2. + 3. *I, 4. + 5.*I, 5. + 6.*I};

	auto const out = multi::fftw::dft({false}, in, static_cast<multi::fftw::sign>(+1));
	BOOST_REQUIRE( out == in );
}

BOOST_AUTO_TEST_CASE(fftw_1D_const_copy_by_false_forward) {
	multi::array<complex, 1> const in = {1. + 2.*I, 2. + 3. *I, 4. + 5.*I, 5. + 6.*I};

	auto const out = multi::fftw::dft_forward({false}, in);
	BOOST_REQUIRE( out == in );
}

BOOST_AUTO_TEST_CASE(fftw_many1_from_2) {
	multi::array<complex, 2> in({3, 10}); randomizer<complex>{}(in);
	multi::array<complex, 2> out({3, 10});
	fftw::dft({false, true}, in, out, fftw::forward);

	multi::array<complex, 2> out2({3, 10});
	for(int i = 0; i!=size(in); ++i) {
		fftw::dft_forward(in[i], out2[i]);
	}

	BOOST_REQUIRE(out2 == out);
}

BOOST_AUTO_TEST_CASE(fftw_many2_from_3) {
	multi::array<complex, 3> in({3, 5, 6}); randomizer<complex>{}(in);
	multi::array<complex, 3> out({3, 5, 6});
	fftw::dft_forward({false, true, true}, in, out);

	multi::array<complex, 3> out2({3, 5, 6});
	for(int i = 0; i!=size(in); ++i) {
		fftw::dft_forward(in[i], out2[i]);
	}

	BOOST_REQUIRE(out2 == out);
}

BOOST_AUTO_TEST_CASE(fftw_many2_from_2) {
	multi::array<complex, 2> in({5, 6}); randomizer<complex>{}(in);
	multi::array<complex, 2> out({5, 6});
	fftw::dft({true, true}, in, out, static_cast<fftw::sign>(FFTW_FORWARD));

	multi::array<complex, 2> out2({5, 6});
	fftw::dft(in, out2, FFTW_FORWARD);
	BOOST_REQUIRE(out2 == out);
}

BOOST_AUTO_TEST_CASE(fftw_4D) {
	multi::array<complex, 4> const in = [] {
		multi::array<complex, 4> in({10, 10, 10, 10}); in[2][3][4][5] = 99.; return in;
	}();
	auto fwd = multi::fftw::dft({true, true, true, true}, in, fftw::forward);
	BOOST_REQUIRE(in[2][3][4][5] == 99.);
}

BOOST_AUTO_TEST_CASE(fftw_4D_many) {
	auto const in = [] {
		multi::array<complex, 4> in({97, 95, 101, 10}, 0.);
		in[2][3][4][5] = 99.; return in;
	}();
	auto fwd = multi::fftw::dft({true, true, true, false}, in, fftw::forward);
	BOOST_REQUIRE( in[2][3][4][5] == 99. );

	multi::array<complex, 4> out(extensions(in));
	multi::fftw::many_dft(begin(unrotated(in)), end(unrotated(in)), begin(unrotated(out)), fftw::forward);
	BOOST_REQUIRE( out == fwd );
}

BOOST_AUTO_TEST_CASE(cufft_many_2D) {
	auto const in = [] {
		multi::array<complex, 3> ret({10, 10, 10});
		std::generate(ret.data_elements(), ret.data_elements() + ret.num_elements(),
			[eng = std::default_random_engine{std::random_device{}()}, uniform_01 = std::uniform_real_distribution<>{}]() mutable{
				return complex{uniform_01(eng), uniform_01(eng)};
			}
		);
		return ret;
	}();
	multi::array<complex, 3> out(extensions(in));
	multi::fftw::many_dft((in.rotated()).begin(), (in.rotated()).end(), (out.rotated()).begin(), multi::fftw::forward);

	multi::array<complex, 3> out2(extensions(in));
	multi::fftw::dft_forward({true, false, true}, in, out2);
	BOOST_REQUIRE( out == out2 );
}

BOOST_AUTO_TEST_CASE(fftw_5D) {
	multi::array<complex, 5> in({4, 5, 6, 7, 8}, 0.);
	BOOST_REQUIRE( size(in) == 4 );

	in[2][3][4][5][6] = 99.;
	auto const out_fwd = multi::fftw::dft(in, fftw::forward);

	BOOST_REQUIRE(in[2][3][4][5][6] == 99.);
	BOOST_REQUIRE( power(in) - power(out_fwd)/num_elements(out_fwd) < 1e-8 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_power_plan) {
	multi::array<complex, 2> in({16, 16});
	std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic): test code
	multi::array<complex, 2> out(extensions(in));
	multi::fftw::plan const p{in, out, fftw::forward, fftw::preserve_input};
	p(); //execute(p); //p.execute();
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-8 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_power_plan_modern) {
	multi::array<complex, 2> in({16, 16});
	std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic): test code
	multi::array<complex, 2> out(extensions(in));
	multi::fftw::plan const p{in.layout(), out.layout(), fftw::forward, fftw::preserve_input};
	p(in.base(), out.base()); //execute(p); //p.execute();
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-8 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_power_plan_modern_measure) {
	multi::array<complex, 2> in({16, 16});
	std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic): test code
	multi::array<complex, 2> out(extensions(in));
	multi::fftw::plan const p{in.layout(), out.layout(), fftw::forward, fftw::preserve_input};
	p(in.base(), out.base()); //execute(p); //p.execute();
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-8 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_power_dft) {
	multi::array<complex, 2> in({16, 16});
	std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic): test code
	multi::array<complex, 2> out(extensions(in));
	multi::fftw::dft_forward(in, out);
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-8 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_power_dft_out) {
	multi::array<complex, 2> in({16, 16});
	std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic): test code
	auto out = multi::fftw::dft(in, fftw::forward);
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-8 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_power_dft_out_default) {
	multi::array<complex, 2> in({16, 16});
	std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic): test code
	auto out = multi::fftw::dft(in, fftw::forward);
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-8 );
}

BOOST_AUTO_TEST_CASE(fftw_3D_power) {
	multi::array<complex, 3> in({4, 4, 4}); std::iota(in.data_elements(), in.data_elements() + in.num_elements(), 1.2);
	multi::array<complex, 3> out = fftw::dft(in, fftw::forward);
	BOOST_REQUIRE( std::abs(power(in) - power(out)/num_elements(out)) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(fftw_3D_power_in_place) {
	multi::array<complex, 3> io({4, 4, 4}); std::iota(io.data_elements(), io.data_elements() + io.num_elements(), 1.2);
	auto powerin = power(io);
	fftw::dft_inplace(io, fftw::forward);
	BOOST_REQUIRE( powerin - power(io)/num_elements(io) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(fftw_3D_power_in_place_over_ref_inplace) {
	multi::array<complex, 3> io({4, 4, 4});
	std::iota(io.data_elements(), io.data_elements() + io.num_elements(), 1.2); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic): test code
	auto const powerin = power(io);
//	fftw::dft_inplace(multi::array_ref<complex, 3>(io.data(), io.extensions()), fftw::forward);
	fftw::dft_inplace(multi::array_ref<complex, 3>(data_elements(io), extensions(io)), fftw::forward);
	BOOST_REQUIRE( powerin - power(io)/num_elements(io) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(fftw_3D_power_out_of_place_over_ref) {
	multi::array<complex, 3> in({4, 4, 4});
	std::iota(data_elements(in), data_elements(in)+num_elements(in), 1.2); //  NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic): test code
	multi::array<complex, 3> out({4, 4, 4});
	multi::array_ref<complex, 3>(data_elements(out), extensions(out)) = fftw::dft(multi::array_cref<complex, 3>(data_elements(in), extensions(in)), fftw::forward);
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(fftw_3D_power_out_of_place_over_temporary) {
	double powerin = NAN;
	auto f = [&](){
		multi::array<complex, 3> in({4, 4, 4});
		std::iota(data_elements(in), data_elements(in)+num_elements(in), 1.2); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic): test code
		powerin = power(in);
		return in;
	};
	auto out = fftw::dft(f(), fftw::forward);
	BOOST_REQUIRE( std::abs(powerin - power(out)/num_elements(out)) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_transposition_square_inplace) {
	multi::array<complex, 2> in = {
		{11., 12.},
		{21., 22.}
	};
	BOOST_REQUIRE( in[1][0] == 21. );

	multi::fftw::copy(in, rotated(in));
	BOOST_TEST( in[0][1].real() == 21. );
	BOOST_TEST( in[0][1].imag() ==  0. );
}

BOOST_AUTO_TEST_CASE(fftw_4D_inq_poisson) {
	multi::array<complex, 4> const in = [] {
		multi::array<complex, 4> in({50, 100, 137, 1});
		std::iota(data_elements(in), data_elements(in)+num_elements(in), 1.2); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic): test code
		return in;
	}();

	multi::array<complex, 4> out(extensions(in));
	using multi::fftw::sign;
	multi::fftw::dft(
		{static_cast<sign>(0), static_cast<sign>(+1), static_cast<sign>(+1), static_cast<sign>(0)},
		in, out
	);

	using boost::multi::detail::get;
	BOOST_TEST( power(in) == power(out)/get<1>(sizes(out))/get<2>(sizes(out)) , boost::test_tools::tolerance(1e-10) );
}

BOOST_AUTO_TEST_CASE(fftw_1D_power) {
	multi::array<complex, 1> in(N, 0.);
	BOOST_REQUIRE( size(in) == N );

	std::iota(begin(in), end(in), 1.);
	BOOST_TEST_REQUIRE( power(in) == 1496. );

	multi::array<complex, 1> out(extensions(in));

	auto* p = multi::fftw_plan_dft(in, out, fftw::forward, fftw::preserve_input);
	fftw_execute(p);
	fftw_destroy_plan(p);
	BOOST_TEST( power(in) == power(out)/num_elements(out), boost::test_tools::tolerance(1e-15) );
}

namespace boost::multi::fftw {

template<class MDIterator>
class fft_iterator {
	MDIterator base_;
	std::array<fftw::sign, MDIterator::rank_v> which_ = {};

 public:
	using iterator_type = MDIterator;

	using difference_type = typename std::iterator_traits<MDIterator>::difference_type;
	using value_type      = typename std::iterator_traits<MDIterator>::value_type;
	using pointer         = void*;
	class reference {
		typename MDIterator::reference impl_;
		explicit reference(typename MDIterator::reference impl) : impl_{std::move(impl)} {}
		friend class fft_iterator;

	 public:
		using extensions_type = typename MDIterator::reference::extensions_type;
		auto extensions() const -> extensions_type {return impl_.extensions();}
	};

	using iterator_category = std::random_access_iterator_tag;  // using iterator_category = std::input_iterator_tag;

	explicit fft_iterator(iterator_type base, std::array<fftw::sign, MDIterator::rank_v> which) noexcept : base_{std::move(base)}, which_{which} {}

	friend auto operator-(fft_iterator const& a, fft_iterator const& b) -> difference_type {
		return a.base_ - b.base_;
	}

	template<class ItOut>
	friend auto copy(fft_iterator first, fft_iterator last, ItOut d_first) {
		assert(first.which_ == last.which_);
		fftw::dft(
			first.which_,
			multi::ref(first.base_, last.base_),
			multi::ref(d_first, d_first + (last.base_ - first.base_))
		);

		return d_first + (last.base_ - first.base_);
	}
	template<class ItOut>
	friend auto uninitialized_copy(fft_iterator first, fft_iterator last, ItOut d_first) {
		return copy(first, last, d_first);
	}

	auto operator*() const {return reference{*base_};}
};

template<class A> void what(A&&) = delete;

template<class Array>
class fft_range {
	typename Array::const_iterator first_;
	typename Array::size_type      n_;
	std::array<fftw::sign, Array::rank_v> which_;

 public:
	using iterator_type = typename Array::const_iterator;

	using size_type = typename Array::size_type;
	using iterator = fft_iterator<iterator_type>;

	using decay_type = typename Array::decay_type;

	explicit fft_range(Array const& in, std::array<fftw::sign, Array::rank_v> which) : first_{in.begin()}, n_{in.size()}, which_{which} {}

	auto begin() const {return iterator{first_     , which_};}
	auto end()   const {return iterator{first_ + n_, which_};}

	auto size()       const {return n_;}
	auto extensions() const {return multi::ref(first_, first_ + n_).extensions();}

	auto operator+() const {return decay_type(*this);}
	auto base() const {return multi::ref(first_, first_ + n_);}
	auto rotated() const {
		auto new_which = which_;
		std::rotate(new_which.begin(), new_which.begin() + 1, new_which.end());
		return fft_range<std::decay_t<decltype(multi::ref(first_, first_ + n_).rotated())>>{multi::ref(first_, first_ + n_).rotated(), new_which};
	}
	auto unrotated() const {
		auto new_which = which_;
		std::rotate(new_which.rbegin(), new_which.rbegin() + 1, new_which.rend());
		return fft_range<std::decay_t<decltype(multi::ref(first_, first_ + n_).unrotated())>>{multi::ref(first_, first_ + n_).unrotated(), new_which};
	}
	auto transposed() const {
		auto new_which = which_;
		std::swap(std::get<0>(new_which), std::get<1>(new_which));
		return fft_range<std::decay_t<decltype(multi::ref(first_, first_ + n_).transposed())>>{multi::ref(first_, first_ + n_).transposed(), new_which};
	}

	template<class... FBNs>
	auto operator()(FBNs... fbns) const {
		static_assert( sizeof...(fbns) <= Array::rank_v , "too many arguments");
		auto new_which = which_;
		std::array<fftw::sign, sizeof...(fbns)> fbna{fbns...};
		std::transform(fbna.begin(), fbna.end(), new_which.begin(), new_which.begin(), 
			[](auto fbn, auto nw) {
				if(fbn == fftw::none) {return nw;}
				assert(nw == fftw::none);
				return fbn;
			}
		);
		return fft_range<std::decay_t<decltype(multi::ref(first_, first_ + n_)())>>{multi::ref(first_, first_ + n_)(), new_which};
	}
};

template<class Array> 
auto  ref(Array const& in) {return fft_range{in, {}};}

template<dimensionality_type ND = 1, class Array>
auto  fft(Array const& in) {
	std::array<fftw::sign, Array::rank_v> which{};
	for(auto it = which.begin(); it != which.begin() + ND; ++it) {*it = fftw::forward ;}
	return fft_range{in, which};
}
template<dimensionality_type ND = 1, class Array>
auto ifft(Array const& in) {
	std::array<fftw::sign, Array::rank_v> which{};
	for(auto it = which.begin(); it != which.begin() + ND; ++it) {*it = fftw::backward;}
	return fft_range{in, which};
}

}  // end namespace boost::multi::fftw

#if 1
BOOST_AUTO_TEST_CASE(fftw_2D_const_range_part1) {
	multi::array<complex, 2> const in = {
		{  1. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{  3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{  4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{  3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		multi::static_array<complex, 2> fwd(in.extensions());

		auto* data = fwd.data_elements();

		fwd = multi::fftw::fft(in);

		BOOST_REQUIRE( size(fwd) == size(in) );
		BOOST_REQUIRE( data == fwd.data_elements() );

		BOOST_TEST_REQUIRE( power(  in    ) - power(  fwd    )/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[0]) - power((~fwd)[0])/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[2]) - power((~fwd)[2])/size(fwd) < 1e-8 );
	}
	{
		multi::array<complex, 2> fwd(in.extensions());

		auto* data = fwd.data_elements();

		fwd = multi::fftw::fft(in);

		BOOST_REQUIRE( size(fwd) == size(in) );
		BOOST_REQUIRE( data == fwd.data_elements() );

		BOOST_TEST_REQUIRE( power(  in    ) - power(  fwd    )/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[0]) - power((~fwd)[0])/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[2]) - power((~fwd)[2])/size(fwd) < 1e-8 );
	}
	{
		multi::array<complex, 2> fwd;

		auto* data = fwd.data_elements();

		fwd = multi::fftw::fft(in);

		BOOST_REQUIRE( size(fwd) == size(in) );
		BOOST_REQUIRE( data != fwd.data_elements() );

		BOOST_TEST_REQUIRE( power(  in    ) - power(  fwd    )/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[0]) - power((~fwd)[0])/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[2]) - power((~fwd)[2])/size(fwd) < 1e-8 );
	}
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_part2) {
	multi::array<complex, 2> const in = {
		{  1. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{  3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{  4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{  3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		multi::array<complex, 2> fwd(multi::fftw::fft(in));

		BOOST_REQUIRE( size(fwd) == size(in) );

		BOOST_TEST_REQUIRE( power(in      ) - power(  fwd    )/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[0]) - power((~fwd)[0])/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[2]) - power((~fwd)[2])/size(fwd) < 1e-8 );
	}
	{
		multi::array<complex, 2> fwd = multi::fftw::fft(in);

		BOOST_REQUIRE( size(fwd) == size(in) );

		BOOST_TEST_REQUIRE( power(in      ) - power(  fwd    )/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[0]) - power((~fwd)[0])/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[2]) - power((~fwd)[2])/size(fwd) < 1e-8 );
	}
	{
		auto fwd = multi::array<complex, 2>(multi::fftw::fft(in));

		BOOST_REQUIRE( size(fwd) == size(in) );

		BOOST_TEST_REQUIRE( power(in      ) - power(  fwd    )/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[0]) - power((~fwd)[0])/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[2]) - power((~fwd)[2])/size(fwd) < 1e-8 );
	}
	{
		auto fwd = + multi::fftw::fft(in);

		BOOST_REQUIRE( fwd.extensions() == in.extensions() );

		BOOST_TEST_REQUIRE( power(in      ) - power(  fwd    )/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[0]) - power((~fwd)[0])/size(fwd) < 1e-8 );
		BOOST_TEST_REQUIRE( power((~in)[2]) - power((~fwd)[2])/size(fwd) < 1e-8 );
	}
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_ref) {
	multi::array<complex, 2> const in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		multi::array<complex, 2> fwd(in.extensions());

		auto* data = fwd.data_elements();

		fwd = multi::fftw::ref(in);

		BOOST_REQUIRE( data == fwd.data_elements() );
		BOOST_REQUIRE( fwd == in );
	}
	{
		multi::array<complex, 2> fwd = multi::fftw::ref(in);
		BOOST_REQUIRE( fwd == in );
	}
	{
		multi::array<complex, 2> fwd = in.transposed();
		BOOST_REQUIRE(   fwd .size() == 3 );
		BOOST_REQUIRE( (~fwd).size() == 5 );

		BOOST_TEST_REQUIRE( fwd[0][0] == in[0][0] );
		BOOST_TEST_REQUIRE( fwd[1][0] == in[0][1] );
		BOOST_TEST_REQUIRE( fwd[2][0] == in[0][2] );
	}
	{
		multi::array<complex, 2> fwd({3, 5}, 0.);
		fwd() = multi::fftw::ref(in.transposed());
		BOOST_REQUIRE(   fwd .size() == 3 );
		BOOST_REQUIRE( (~fwd).size() == 5 );

		BOOST_TEST_REQUIRE( fwd[0][0] == in[0][0] );
		BOOST_TEST_REQUIRE( fwd[1][0] == in[0][1] );
		BOOST_TEST_REQUIRE( fwd[2][0] == in[0][2] );
	}
}

BOOST_AUTO_TEST_CASE(fftw_2D_const_range_ref_part2) {
	multi::array<complex, 2> const in = {
		{  100. + 2.*I,  9. - 1.*I, 2. +  4.*I},
		{    3. + 3.*I,  7. - 4.*I, 1. +  9.*I},
		{    4. + 1.*I,  5. + 3.*I, 2. +  4.*I},
		{    3. - 1.*I,  8. + 7.*I, 2. +  1.*I},
		{   31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};

	{
		auto in2 = + multi::fftw::ref(in);
		BOOST_REQUIRE( in2 == in );
	}
	{
		auto in2 = + multi::fftw::ref(in.transposed());
		BOOST_REQUIRE( in2 == in.transposed() );
	}
	{
		auto in2 = + multi::fftw::ref(in.rotated());
		BOOST_REQUIRE( in2 == in.rotated() );
	}
	{
		auto&& r = multi::fftw::ref(in);
		auto in2 = + r;
		BOOST_REQUIRE( in2 == in );

		multi::array<complex, 2> tt({3, 5});
		multi::fftw::dft({}, in, tt.transposed(), multi::fftw::forward);

		multi::array<complex, 2> in2t({3, 5}, 99.);
		in2t() = multi::fftw::ref(in).transposed();

		{
			auto xs = std::get<0>(in.extensions());  // TODO(correaa) use structured bindings
			auto ys = std::get<1>(in.extensions());
			for(auto const x : xs) {
				for(auto const y : ys) {
					std::cout<< in[x][y] <<'\t';
				}
				std::cout<< std::endl;
			}
			std::cout<< std::endl;
		}
		{
			auto xs = std::get<0>(in2t.extensions());
			auto ys = std::get<1>(in2t.extensions());
			for(auto const x : xs) {
				for(auto const y : ys) {
					std::cout<< in2t[x][y] <<'\t';
				}
				std::cout<< std::endl;
			}
			std::cout<< std::endl;
		}
		{
			auto xs = std::get<0>(tt.extensions());
			auto ys = std::get<1>(tt.extensions());
			for(auto x : xs) {
				for(auto y : ys) {
					std::cout<< tt[x][y] <<'\t';
				}
				std::cout<< std::endl;
			}
			std::cout<< std::endl;
		}

		BOOST_REQUIRE( tt   == in.transposed() );
		BOOST_REQUIRE( in2t == in.transposed() );
	}
	{
		auto in2 = + multi::fftw::ref(in).transposed();
		BOOST_REQUIRE( in2 == in.transposed() );
	}
	{
		multi::array<complex, 2> fwd({3, 5}, 0.);
		fwd() = multi::fftw::ref(in).transposed();
		BOOST_REQUIRE(   fwd .size() == 3 );
		BOOST_REQUIRE( (~fwd).size() == 5 );

		BOOST_TEST_REQUIRE( fwd[0][0] == in[0][0] );
		BOOST_TEST_REQUIRE( fwd[1][0] == in[0][1] );
		BOOST_TEST_REQUIRE( fwd[2][0] == in[0][2] );
	}
}

BOOST_AUTO_TEST_CASE(fftw_4D_many_new_interface) {
	auto const in = [] {
		multi::array<complex, 4> in({97, 95, 101, 10}, 0.);
		in[2][3][4][5] = 99.;
		return in;
	}();
	{
		auto fwd = + multi::fftw::ref(in)(fftw::forward, fftw::forward, fftw::forward, fftw::none);
		BOOST_REQUIRE( in[2][3][4][5] == 99. );

		multi::array<complex, 4> out(extensions(in));
		multi::fftw::many_dft(begin(unrotated(in)), end(unrotated(in)), begin(unrotated(out)), fftw::forward);
		BOOST_REQUIRE( out == fwd );
	}
	{
		auto fwd = + multi::fftw::ref(in)(fftw::forward, fftw::forward, fftw::forward);
		BOOST_REQUIRE( in[2][3][4][5] == 99. );

		multi::array<complex, 4> out(extensions(in));
		multi::fftw::many_dft(begin(unrotated(in)), end(unrotated(in)), begin(unrotated(out)), fftw::forward);
		BOOST_REQUIRE( out == fwd );
	}
}

#endif
