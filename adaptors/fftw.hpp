#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&clang++ -std=c++17 -Ofast -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_ADAPTORS_FFTW $0.cpp -o$0x `pkg-config --libs fftw3` -lboost_timer -lboost_unit_test_framework&&$0x&&rm $0x $0.cpp; exit
#endif
#ifndef MULTI_ADAPTORS_FFTW_HPP
#define MULTI_ADAPTORS_FFTW_HPP
// © Alfredo A. Correa 2018-2019

#include<fftw3.h>

#include "../../multi/utility.hpp"
#include "../../multi/array.hpp"

#include<cmath>
#include<complex>
#include<memory>
#include<numeric> // accumulate

#include<experimental/tuple> // experimental::apply

#if __cplusplus>=201703L and __has_cpp_attribute(nodiscard)>=201603
#define NODISCARD(MsG) [[nodiscard]]
#elif __has_cpp_attribute(gnu::warn_unused_result)
#define NODISCARD(MsG) [[gnu::warn_unused_result]]
#else
#define NODISCARD(MsG)
#endif

namespace boost{
namespace multi{

namespace fftw{
	template<class T>
	auto alignment_of(T* p){return fftw_alignment_of((double*)p);}
}

namespace{
	using std::complex;
}

template<typename Size>
auto fftw_plan_dft_1d(
	Size N, 
	complex<double> const* in, complex<double>* out, int sign, 
	unsigned flags = FFTW_ESTIMATE
){
#ifndef NDEBUG
	auto check = in[N/3]; // check that const data will not been overwritten
#endif
	assert( fftw::alignment_of(in) == fftw::alignment_of(out) );
	auto ret=::fftw_plan_dft_1d(N, (fftw_complex*)in, (fftw_complex*)out, sign, flags | FFTW_PRESERVE_INPUT );
	assert(check == in[N/3]); // check that const data has not been overwritten
	return ret;
}

template<typename Size>
auto fftw_plan_dft_1d(
	Size N, 
	complex<double>* in, std::complex<double>* out, int sign, 
	unsigned flags = FFTW_ESTIMATE
){
	assert( fftw::alignment_of(in) == fftw::alignment_of(out) );
	return ::fftw_plan_dft_1d(N, (fftw_complex*)in, (fftw_complex*)out, sign, flags);
}

template<typename Size>
auto fftw_plan_dft_2d(
	Size N1, Size N2, 
	complex<double> const* in, complex<double>* out, int sign, 
	unsigned flags = FFTW_ESTIMATE
){
	assert( fftw::alignment_of(in) == fftw::alignment_of(out) );
#ifndef NDEBUG
	auto check = in[N1*N2/3]; // check that const data will not been overwritten
#endif
	auto ret = ::fftw_plan_dft_2d(N1, N2, (fftw_complex*)in, (fftw_complex*)out, sign, flags | FFTW_PRESERVE_INPUT);
	assert( check == in[N1*N2/3] ); // check that const data has not been overwritten
	return ret;
}

template<typename Size>
auto fftw_plan_dft_2d(
	Size N1, Size N2, 
	complex<double>* in, complex<double>* out, int sign, 
	unsigned flags = FFTW_ESTIMATE
){
	assert(fftw_alignment_of((double*)in) == fftw_alignment_of((double*)out));
	return ::fftw_plan_dft_2d(N1, N2, (fftw_complex*)in, (fftw_complex*)out, sign, flags);
}

template<typename Size>
auto fftw_plan_dft_3d(
	Size N1, Size N2, Size N3, 
	complex<double>* in, complex<double>* out, int sign, 
	unsigned flags = FFTW_ESTIMATE
){
	assert(fftw_alignment_of((double*)in) == fftw_alignment_of((double*)out));
	return ::fftw_plan_dft_3d(N1, N2, N3, (fftw_complex*)in, (fftw_complex*)out, sign, flags);
}
template<typename Size>
auto fftw_plan_dft_3d(
	Size N1, Size N2, Size N3, 
	complex<double> const* in, complex<double>* out, int sign, 
	unsigned flags = FFTW_ESTIMATE
){
	assert( flags & FFTW_PRESERVE_INPUT );
	assert(fftw_alignment_of((double*)in) == fftw_alignment_of((double*)out));
	return ::fftw_plan_dft_3d(N1, N2, N3, (fftw_complex*)in, (fftw_complex*)out, sign, flags | FFTW_PRESERVE_INPUT);
}

template<typename Rank>
auto fftw_plan_dft(
	Rank r, int* ns, 
	complex<double>* in, std::complex<double>* out, 
	int sign, unsigned flags = FFTW_ESTIMATE
){
	assert(fftw_alignment_of((double*)in) == fftw_alignment_of((double*)out));
	return ::fftw_plan_dft(r, ns, (fftw_complex*)in, (fftw_complex*)out, sign, flags);
}
template<typename RankType>
auto fftw_plan_dft(
	RankType r, int* ns, 
	complex<double> const* in, complex<double>* out, 
	int sign, unsigned flags = FFTW_ESTIMATE | FFTW_PRESERVE_INPUT
){
	assert( flags & FFTW_PRESERVE_INPUT );
	assert(fftw::alignment_of(in) == fftw::alignment_of(out));
#ifndef NDEBUG
	size_t ne = 1; for(RankType i = 0; i != r; ++i) ne*=ns[i];
	auto check = in[ne/3]; // check that const data will not been overwritten
#endif
	auto ret=::fftw_plan_dft(r, ns, (fftw_complex*)in, (fftw_complex*)out, sign, flags);
	assert(check == in[ne/3]); // check that const data has not been overwritten
	return ret;
}

//////////////////////////////////////////////////////////////////////////////

#if 0
template<class InIt, typename Size, class OutIt>
auto fftw_plan_dft_1d_n(InIt it, Size n, OutIt dest, decltype(FFTW_FORWARD) sign, decltype(FFTW_ESTIMATE) flags = FFTW_ESTIMATE){
	return fftw_plan_dft_1d(n, data(it), data(dest), sign, flags);
}
template<class InIt, class OutIt>
auto fftw_plan_dft_1d(InIt first, InIt last, OutIt dest, decltype(FFTW_FORWARD) sign, decltype(FFTW_ESTIMATE) flags = FFTW_ESTIMATE){
	using std::distance;
	return fftw_plan_dft_1d_n(first, distance(first, last), dest, sign, flags);
}
#endif

///////////////////////////////////////////////////////////////////////////////

template<typename In, typename Out>
auto fftw_plan_dft_1d(
	In&& in, Out&& out, int sign, unsigned flags = FFTW_ESTIMATE
){
	static_assert(in.dimensionality == 1, "!"); assert(size(in) == size(out));
	assert( in.is_compact() ); assert( out.is_compact() );
	return multi::fftw_plan_dft_1d(size(in), data_elements(in), data_elements(out), sign, flags);
}

template<class In, class Out>
auto fftw_plan_dft_2d(
	In&& in, Out&& out, int sign, unsigned flags = FFTW_ESTIMATE
){
	static_assert(in.dimensionality == 2, "!"); assert(in.sizes() == out.sizes());
	assert( in.is_compact() ); assert( out.is_compact() );
	return multi::fftw_plan_dft_2d(
		sizes(in)[0], sizes(in)[1], 
		data_elements(in), data_elements(out), sign, flags
	);
}

template<class In, class Out>
auto fftw_plan_dft_3d(
	In&& in, Out&& out, int sign, unsigned flags = FFTW_ESTIMATE
){
	static_assert(in.dimensionality == 3, "!"); assert(in.sizes() == out.sizes());
	assert( in.is_compact() ); assert( out.is_compact() );
	return multi::fftw_plan_dft_3d(
		sizes(in)[0], sizes(in)[1], sizes(in)[2],
		data(in), data(out),
		sign, flags
	);
}

/*
namespace detail {
template<class T, class Tuple, std::size_t... I>
constexpr std::array<std::remove_cv_t<T>, std::tuple_size<Tuple>{}> 
to_array_impl(Tuple&& t, std::index_sequence<I...>){
	return { static_cast<std::remove_cv_t<T>>(std::get<I>(std::forward<Tuple>(t)))... };
}
}
*/

template<class T, class Tuple> constexpr auto to_array(Tuple&& t){
    return detail::to_array_impl<T>(std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size<Tuple>{}>{});
}

#if(__cpp_if_constexpr>=201606)
//https://stackoverflow.com/a/35110453/225186
template<class T> constexpr std::remove_reference_t<T> _constx(T&&t){return t;}
#define logic_assert(C, M) \
	if constexpr(noexcept(_constx(C))) static_assert((C), M); else assert((C)&& M);
#else
#define logic_assert(ConditioN, MessagE) assert(ConditioN && MessagE);
#endif

template<class In, class Out, dimensionality_type D = std::decay_t<In>::dimensionality,
typename = std::enable_if_t<D == std::decay_t<Out>::dimensionality>>
fftw_plan fftw_plan_many_dft(dimensionality_type Rank, In&& in, Out&& out, int sign, unsigned flags = FFTW_ESTIMATE){
	using multi::sizes; using multi::strides; assert(sizes(in) == sizes(out));
	auto ion = to_array<std::ptrdiff_t>(sizes(in));
	auto istrides = to_array<std::ptrdiff_t>(strides(in));
	auto ostrides = to_array<std::ptrdiff_t>(strides(out));
	std::array<fftw_iodim64, D> dims;
	for(int i = 0; i != D; ++i) dims[i] = fftw_iodim64{ion[i], istrides[i], ostrides[i]};
//	std::array<fftw_iodim64, D> howmany_dims = dims;//{ion[0], istrides[0], ostrides[0]};
	assert( Rank <= D );
	return fftw_plan_guru64_dft(
		/*int rank*/ Rank, 
		/*const fftw_iodim64 *dims*/ &dims[D - Rank],
		/*int howmany_rank*/ D - Rank,
		/*const fftw_iodim *howmany_dims*/ dims.data(), //nullptr, //howmany_dims.data(), //;//nullptr,
		/*fftw_complex *in*/ const_cast<fftw_complex*>(reinterpret_cast<fftw_complex const*>(base(in))), 
		/*fftw_complex *out*/ reinterpret_cast<fftw_complex*>(base(out)),
		sign, flags
	);
}

template<class In, class Out, std::size_t D = std::decay_t<In>::dimensionality,
typename = std::enable_if_t<D == std::decay_t<Out>::dimensionality>>
fftw_plan fftw_plan_many_dft(std::array<bool, D> which, In&& in, Out&& out, int sign, unsigned flags = FFTW_ESTIMATE){
	using multi::sizes; using multi::strides; assert(sizes(in) == sizes(out));
	auto ion = to_array<std::ptrdiff_t>(sizes(in));
	auto istrides = to_array<std::ptrdiff_t>(strides(in));
	auto ostrides = to_array<std::ptrdiff_t>(strides(out));
	std::array<fftw_iodim64, D> dims   ; auto next_yes = dims.begin();
	std::array<fftw_iodim64, D> howmany; auto next_not = howmany.begin();
	for(int i = 0; i != D; ++i){
		if(which[i]){*next_yes = fftw_iodim64{ion[i], istrides[i], ostrides[i]}; ++next_yes;}
		else        {*next_not = fftw_iodim64{ion[i], istrides[i], ostrides[i]}; ++next_not;}
	};
//	std::array<fftw_iodim64, D> howmany_dims = dims;//{ion[0], istrides[0], ostrides[0]};
//	assert( Rank <= D );
	return fftw_plan_guru64_dft(
		/*int rank*/ next_yes - dims.begin(), 
		/*const fftw_iodim64 *dims*/ dims.data(), 
		/*int howmany_rank*/ next_not - howmany.begin(),
		/*const fftw_iodim *howmany_dims*/ howmany.data(), //nullptr, //howmany_dims.data(), //;//nullptr,
		/*fftw_complex *in*/ const_cast<fftw_complex*>(reinterpret_cast<fftw_complex const*>(base(in))), 
		/*fftw_complex *out*/ reinterpret_cast<fftw_complex*>(base(out)),
		sign, flags
	);
}

template<dimensionality_type Rank, class In, class Out, dimensionality_type D = std::decay_t<In>::dimensionality,
typename = std::enable_if_t<D == std::decay_t<Out>::dimensionality>>
fftw_plan fftw_plan_many_dft(In&& in, Out&& out, int sign, unsigned flags = FFTW_ESTIMATE){
	static_assert( Rank <= D );
	return fftw_plan_many_dft(Rank, in, out, sign, flags);
}

template<class In, class Out, dimensionality_type D = std::decay_t<In>::dimensionality,
typename = std::enable_if_t<D == std::decay_t<Out>::dimensionality>>
fftw_plan fftw_plan_many_dft(In&& in, Out&& out, int sign, unsigned flags = FFTW_ESTIMATE){
	static_assert( 1 <= D );
	return fftw_plan_many_dft<1>(in, out, sign, flags);
}

template<class ArrayIn, class ArrayOut, dimensionality_type D = std::decay_t<ArrayIn>::dimensionality, typename = std::enable_if_t<D == std::decay_t<ArrayOut>::dimensionality> >
auto fftw_plan_dft(
	ArrayIn&& in, ArrayOut&& out, 
	int sign, unsigned flags = FFTW_ESTIMATE
){
	using multi::sizes; using multi::strides; assert(sizes(in) == sizes(out));
#if 0
	using multi::is_compact;
	if( is_compact(in) and is_compact(out) and strides(in) == strides(out) ){
		auto ion = to_array<int>(sizes(in));
		return fftw_plan_dft(
			dimensionality(in), ion.data(), 
			origin(std::forward<ArrayIn>(in)), origin(std::forward<ArrayOut>(out)), 
			sign, flags
		);
	}
#endif
	auto ion = to_array<std::ptrdiff_t>(sizes(in));
	auto istrides = to_array<std::ptrdiff_t>(strides(in));
	auto ostrides = to_array<std::ptrdiff_t>(strides(out));
	std::array<fftw_iodim64, D> dims;
	for(int i = 0; i != D; ++i) dims[i] = fftw_iodim64{ion[i], istrides[i], ostrides[i]};
//	std::array<fftw_iodim64, D> howmany_dims = dims;//{ion[0], istrides[0], ostrides[0]};
	return fftw_plan_guru64_dft(
		/*int rank*/ D, 
		/*const fftw_iodim64 *dims*/ dims.data(),
		/*int howmany_rank*/ 0,
		/*const fftw_iodim *howmany_dims*/ nullptr, //howmany_dims.data(), //;//nullptr,
		/*fftw_complex *in*/ const_cast<fftw_complex*>(reinterpret_cast<fftw_complex const*>(base(in))), 
		/*fftw_complex *out*/ reinterpret_cast<fftw_complex*>(base(out)),
		sign, flags
	);
}

std::complex<double> const* base(std::complex<double> const& c){return &c;}

void fftw_execute_dft(
	fftw_plan p, std::complex<double>* in, std::complex<double>* out){
	assert(fftw_alignment_of((double*)in) == fftw_alignment_of((double*)out) );
	::fftw_execute_dft(p, (fftw_complex*)in, (fftw_complex*)out);
}

template<class ArrayIn, class ArrayOut>
auto fftw_execute_dft(fftw_plan p, ArrayIn&& in, ArrayOut&& out)
->decltype(multi::fftw_execute_dft(p, data_elements(in), data_elements(out))){
	return multi::fftw_execute_dft(p, data_elements(in), data_elements(out));}

namespace fftw{

class plan{
	plan() : impl_{nullptr, &fftw_destroy_plan}{}
	std::unique_ptr<std::remove_pointer_t<fftw_plan>, decltype(&fftw_destroy_plan)> impl_;
public:
	template<dimensionality_type R, class...  As>
	static auto many(As&&... as){
		plan r; r.impl_.reset(multi::fftw_plan_many_dft<R>(std::forward<As>(as)...)); return r;
	}
	template<std::size_t R, class...  As>
	static auto many(std::array<bool, R> which, As&&... as){
		plan r; r.impl_.reset(multi::fftw_plan_many_dft(which, std::forward<As>(as)...)); return r;
	}

	plan(plan const&) = delete;//default;
	plan(plan&&) = default;
	template<typename... Args>
	plan(Args&&... args) : impl_{multi::fftw_plan_dft(std::forward<Args>(args)...), &fftw_destroy_plan}{}
	void execute() const{fftw_execute(impl_.get());}
	template<class I, class O>
	void execute_dft(I&& i, O&& o) const{fftw_execute_dft(std::forward<I>(i), std::forward<O>(o));}
	template<class I, class O>
	void execute(I&& i, O&& o) const{execute_dft(std::forward<I>(i), std::forward<O>(o));}
	void operator()() const{execute();}
	template<typename I, typename O>
	void operator()(I&& i, O&& o) const{return execute(std::forward<I>(i), std::forward<O>(o));}
	friend void execute(plan const& p){p.execute();}
	double cost() const{return fftw_cost(impl_.get());}
	auto flops() const{
		struct{double add; double mul; double fma;} r;
		fftw_flops(impl_.get(), &r.add, &r.mul, &r.fma);
		return r;
	}
};

enum sign : decltype(FFTW_FORWARD) {none = 0, forward = FFTW_FORWARD, backward = FFTW_BACKWARD };
enum strategy : decltype(FFTW_ESTIMATE) { estimate = FFTW_ESTIMATE, measure = FFTW_MEASURE };

sign invert(sign s){
	switch(s){
		case sign::forward : return sign::backward;
		case sign::backward: return sign::forward;
		case sign::none: return sign::none;
	} __builtin_unreachable();
}

template<typename In, class Out>
auto dft(In const& i, Out&& o, sign s){//, strategy st = fftw::estimate){
	if(s == sign::none) o = i;
	else execute(fftw::plan{i, std::forward<Out>(o), static_cast<int>(s), static_cast<unsigned>(fftw::estimate) | FFTW_PRESERVE_INPUT});
	return std::forward<Out>(o);
}

template<dimensionality_type R, typename In, class Out>
Out&& many_dft(In const& i, Out&& o, sign s){//, strategy st = fftw::estimate){
	if(s == sign::none) o = i;
	else execute(fftw::plan::many<R>(i, std::forward<Out>(o), static_cast<int>(s), static_cast<unsigned>(fftw::estimate) | FFTW_PRESERVE_INPUT));
	return std::forward<Out>(o);
}

template<typename In, class Out, std::size_t D = std::decay_t<In>::dimensionality>
Out&& dft(std::array<bool, D> which, In const& i, Out&& o, sign s){
	execute(fftw::plan::many(which, i, std::forward<Out>(o), static_cast<int>(s), static_cast<unsigned>(fftw::estimate) | FFTW_PRESERVE_INPUT));
	return std::forward<Out>(o);
}

#if 1
template<typename In, class Out, std::size_t D = std::decay_t<In>::dimensionality>
auto dft(std::array<sign, D> which, In const& i, Out&& o){//, strategy st = fftw::estimate){
	{
		std::array<bool, D> which_forward;
		std::transform(begin(which), end(which), begin(which_forward), [](auto e){return e==sign::forward?true:false;});
		assert(which_forward[0] == false);
		assert(which_forward[1] == true);
		dft(which_forward, i, o, sign::forward);
	}
	{
		std::array<bool, D> which_backward;
		std::transform(which.begin(), which.end(), which_backward.begin(), [](auto e){return e==sign::backward;}); 
		assert(which_backward[0] == false);
		assert(which_backward[1] == false);
		if(std::accumulate(begin(which_backward), end(which_backward)))
			dft(which_backward, o, o, sign::backward);
	}
}
#endif

template<typename In, typename Out = multi::array<typename std::decay_t<In>::element_type, std::decay_t<In>::dimensionality, decltype(get_allocator(std::declval<In const&>()))>>
NODISCARD("when first argument is const")
Out dft(In const& i, sign s){//, strategy st = fftw::estimate){
	Out ret(extensions(i), get_allocator(i));
	dft(i, ret, s);
	return ret;
}

template<dimensionality_type R, typename In, typename Out = multi::array<typename std::decay_t<In>::element_type, std::decay_t<In>::dimensionality, decltype(get_allocator(std::declval<In const&>()))>>
NODISCARD("when first argument is const")
Out many_dft(In const& i, sign s){
	Out ret(extensions(i), get_allocator(i));
	many_dft<R>(i, ret, s);
	return ret;
}

template<typename T, dimensionality_type D, class... As, typename Out = multi::array<T, D, As...>>//typename std::decay_t<In>::element_type, std::decay_t<In>::dimensionality>>
NODISCARD("when first argument can be destroyed")
Out dft(multi::array<T, D, As...>&& i, sign s){
	Out ret(extensions(i), get_allocator(i));
	execute(fftw::plan{i, ret, static_cast<int>(s), static_cast<unsigned>(fftw::estimate) | FFTW_DESTROY_INPUT}); // to do destroy input for move iterators
	return ret;
}

template<class T>
auto dft(std::initializer_list<T> il, sign s){return dft(multi::array<T, 1>(il), s);}

template<class A> decltype(auto) dft_forward(A&& a){return dft(std::forward<A>(a), fftw::forward);}
template<class A> decltype(auto) dft_backward(A&& a){return dft(std::forward<A>(a), fftw::backward);}

template<class T>
auto dft_forward(std::initializer_list<T> il){return dft_forward(multi::array<T, 1>(il));}
template<class T>
auto dft_backward(std::initializer_list<T> il){return dft_backward(multi::array<T, 1>(il));}

template<typename In>
In& dft_inplace(In&& i, sign s, strategy st = fftw::estimate){
	execute(fftw::plan{i, i, (int)s, (unsigned)st});
	return i;
}

}

}}

#if _TEST_MULTI_ADAPTORS_FFTW

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi FFTW adaptor"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include <boost/timer/timer.hpp>

#include "../adaptors/fftw/allocator.hpp"
#include<iostream>
#include "../array.hpp"
#include<complex>
#include<numeric>

#include<experimental/array>
#include<experimental/tuple>

#include<random>

namespace{
//	using namespace Catch::literals;

	using std::cout;
	namespace multi = boost::multi;
	namespace fftw = multi::fftw;

	using complex = std::complex<double>;
	complex const I{0, 1};

	template<class M>
	auto power(M const& m){
		auto sum_norm = [](auto& a, auto& b){return a + std::norm(b);};
		using multi::num_elements; using multi::data_elements; using std::accumulate;
		return accumulate(data_elements(m), data_elements(m) + num_elements(m), double{}, sum_norm);
	}

	constexpr int N = 16;
}

template<class T> struct randomizer{
	template<class M> void operator()(M&& m) const{for(auto&& e:m) operator()(e);}
	void operator()(T& e) const{
		static std::random_device r; static std::mt19937 g{r()}; static std::normal_distribution<T> d;
		e = d(g);
	}
};

template<class T> struct randomizer<std::complex<T>>{
	template<class M> void operator()(M&& m) const{for(auto&& e:m) operator()(e);}
	void operator()(std::complex<T>& e) const{
		static std::random_device r; static std::mt19937 g{r()}; static std::normal_distribution<T> d;
		e = std::complex<T>(d(g), d(g));
	}
};
BOOST_AUTO_TEST_CASE(fftw_2D_identity_2, *boost::unit_test::tolerance(0.0001)){
	multi::array<complex, 2> const in = {
		{ 1. + 2.*I, 9. - 1.*I, 2. + 4.*I},
		{ 3. + 3.*I, 7. - 4.*I, 1. + 9.*I},
		{ 4. + 1.*I, 5. + 3.*I, 2. + 4.*I},
		{ 3. - 1.*I, 8. + 7.*I, 2. + 1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	multi::array<complex, 2> fwd(extensions(in));
	multi::fftw::dft({false, false}, in, fwd, multi::fftw::forward);
	std::cout << fwd[3][2] << ' ' << in[3][2] << std::endl;
	BOOST_REQUIRE( fwd == in );
}

BOOST_AUTO_TEST_CASE(fftw_1D){
	multi::array<complex, 1> in = {1. + 2.*I, 2. + 3. *I, 4. + 5.*I, 5. + 6.*I};
	auto fwd = multi::fftw::dft(in, multi::fftw::forward); // Fourier[in, FourierParameters -> {1, -1}]
	BOOST_REQUIRE(fwd[2] == -2. - 2.*I);
	BOOST_REQUIRE( in[1] == 2. + 3.*I );

	auto bwd = multi::fftw::dft(in, multi::fftw::backward); // InverseFourier[in, FourierParameters -> {-1, -1}]
	BOOST_REQUIRE(bwd[2] == -2. - 2.*I);
}

BOOST_AUTO_TEST_CASE(fftw_2D_identity, *boost::unit_test::tolerance(0.0001)){
	multi::array<complex, 2> const in = {
		{ 1. + 2.*I, 9. - 1.*I, 2. + 4.*I},
		{ 3. + 3.*I, 7. - 4.*I, 1. + 9.*I},
		{ 4. + 1.*I, 5. + 3.*I, 2. + 4.*I},
		{ 3. - 1.*I, 8. + 7.*I, 2. + 1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	auto fwd = multi::fftw::dft(in, multi::fftw::none);
	std::cout << fwd[3][2] << ' ' << in[3][2] << std::endl;
	BOOST_REQUIRE( fwd == in );
}

BOOST_AUTO_TEST_CASE(fftw_2D, *boost::unit_test::tolerance(0.0001)){
	multi::array<complex, 2> const in = {
		{ 1. + 2.*I, 9. - 1.*I, 2. + 4.*I},
		{ 3. + 3.*I, 7. - 4.*I, 1. + 9.*I},
		{ 4. + 1.*I, 5. + 3.*I, 2. + 4.*I},
		{ 3. - 1.*I, 8. + 7.*I, 2. + 1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	auto fwd = multi::fftw::dft(in, multi::fftw::forward);
	BOOST_TEST( real(fwd[3][1]) == -19.0455 ); // Fourier[in, FourierParameters -> {1, -1}][[4]][[2]]
	BOOST_TEST( imag(fwd[3][1]) == - 2.22717 );

	multi::array<complex, 1> const in0 = { 1. + 2.*I, 9. - 1.*I, 2. + 4.*I};
	using multi::fftw::dft_forward;

	BOOST_REQUIRE( dft_forward(in[0]) == dft_forward(in0) );
	BOOST_REQUIRE( dft_forward(in[3]) == dft_forward({3.-1.*I, 8.+7.*I, 2.+1.*I}) );
	BOOST_REQUIRE( dft_forward(rotated(in)[0]) == dft_forward({1.+2.*I, 3.+3.*I, 4. + 1.*I,  3. - 1.*I, 31. - 1.*I}) );
}

BOOST_AUTO_TEST_CASE(fftw_2D_rotated, *boost::unit_test::tolerance(0.0001)){
	multi::array<complex, 2> const in = {
		{ 1. + 2.*I, 9. - 1.*I, 2. + 4.*I},
		{ 3. + 3.*I, 7. - 4.*I, 1. + 9.*I},
		{ 4. + 1.*I, 5. + 3.*I, 2. + 4.*I},
		{ 3. - 1.*I, 8. + 7.*I, 2. + 1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	auto fwd = multi::fftw::dft(in, multi::fftw::forward);

	using multi::fftw::dft_forward;
	BOOST_REQUIRE( dft_forward(rotated(in)[0]) == dft_forward({1.+2.*I, 3.+3.*I, 4. + 1.*I,  3. - 1.*I, 31. - 1.*I}) );
	BOOST_REQUIRE( dft_forward(rotated(in)) == rotated(fwd) );// rotated(fwd) );
}

BOOST_AUTO_TEST_CASE(fftw_2D_many, *boost::unit_test::tolerance(0.0001)){
	multi::array<complex, 2> const in = {
		{ 1. + 2.*I, 9. - 1.*I, 2. + 4.*I},
		{ 3. + 3.*I, 7. - 4.*I, 1. + 9.*I},
		{ 4. + 1.*I, 5. + 3.*I, 2. + 4.*I},
		{ 3. - 1.*I, 8. + 7.*I, 2. + 1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	multi::array<complex, 2> out(extensions(in));
//	multi::fftw::many_dft<1>(in, out, multi::fftw::forward);
	multi::fftw::dft({false, true}, in, out, multi::fftw::forward);

	using multi::fftw::dft_forward;
	BOOST_REQUIRE( dft_forward(in[0]) == out[0] );

	multi::fftw::many_dft<1>(rotated(in), rotated(out), multi::fftw::forward);
	BOOST_REQUIRE( dft_forward(rotated(in)[0]) == rotated(out)[0] );

	multi::fftw::many_dft<0>(rotated(in), rotated(out), multi::fftw::forward);
	BOOST_REQUIRE( in == out );
}

BOOST_AUTO_TEST_CASE(fftw_many1_from_2){
	multi::array<complex, 2> in({3, 10}); randomizer<complex>{}(in);
	multi::array<complex, 2> out({3, 10});
	fftw::many_dft<1>(in, out, multi::fftw::forward);

	multi::array<complex, 2> out2({3, 10});
	for(int i = 0; i!=size(in); ++i)
		fftw::dft(in[i], out2[i], multi::fftw::forward);

	BOOST_REQUIRE(out2 == out);
}

BOOST_AUTO_TEST_CASE(fftw_many2_from_3){
	multi::array<complex, 3> in({3, 5, 6}); randomizer<complex>{}(in);
	multi::array<complex, 3> out({3, 5, 6});
	fftw::many_dft<2>(in, out, multi::fftw::forward);

	multi::array<complex, 3> out2({3, 5, 6});
	for(int i = 0; i!=size(in); ++i)
		fftw::dft(in[i], out2[i], multi::fftw::forward);

	BOOST_REQUIRE(out2 == out);
}

BOOST_AUTO_TEST_CASE(fftw_many2_from_2){
	multi::array<complex, 2> in({5, 6}); randomizer<complex>{}(in);
	multi::array<complex, 2> out({5, 6});
	fftw::many_dft<2>(in, out, multi::fftw::forward);

	multi::array<complex, 2> out2({5, 6});
	fftw::dft(in, out2, multi::fftw::forward);
	BOOST_REQUIRE(out2 == out);
}

BOOST_AUTO_TEST_CASE(fftw_3D){
	multi::array<complex, 3> in({10, 10, 10});
	in[2][3][4] = 99.;
	auto fwd = multi::fftw::dft(in, multi::fftw::forward);
	BOOST_REQUIRE(in[2][3][4] == 99.);
}

BOOST_AUTO_TEST_CASE(fftw_4D){
	multi::array<complex, 4> in({10, 10, 10, 10});
	in[2][3][4][5] = 99.;
	auto fwd = multi::fftw::dft(in, multi::fftw::forward);
	BOOST_REQUIRE(in[2][3][4][5] == 99.);
}

BOOST_AUTO_TEST_CASE(fftw_5D){
	multi::array<complex, 5> in({4, 5, 6, 7, 8});
	in[2][3][4][5][6] = 99.;
	auto fwd = multi::fftw::dft(in, multi::fftw::forward);
	BOOST_REQUIRE(in[2][3][4][5][6] == 99.);

	BOOST_REQUIRE( std::get<2>(sizes(in)) == 6 );
	auto sizes_as_int = std::experimental::apply(
		[](auto... n){
			auto safe = [](auto i){assert(i<=std::numeric_limits<int>::max()); return static_cast<int>(i);};
			return std::array<int, sizeof...(n)>{safe(n)...};
		}, 
		sizes(in)
	);
	BOOST_REQUIRE( sizes_as_int[2] == 6 );
}

BOOST_AUTO_TEST_CASE(fftw_1D_power){
	multi::array<complex, 1> in(N, 0.); assert( size(in) == N );
	std::iota(begin(in), end(in), 1.);
	multi::array<complex, 1> out(extensions(in));
	static_assert(dimensionality(in)==dimensionality(out), "!");
	auto p = multi::fftw_plan_dft(in, out, FFTW_FORWARD, FFTW_PRESERVE_INPUT);
	fftw_execute(p); 
	fftw_destroy_plan(p);
	BOOST_REQUIRE( (power(in) - power(out)/num_elements(out)) < 1e-17 );
}

BOOST_AUTO_TEST_CASE(fftw_1D_allocator_power){
	using multi::fftw::allocator;
	multi::array<complex, 1, allocator<complex>> in(16, 0.); std::iota(begin(in), end(in), 1.);
	assert( size(in) == N );
	multi::array<complex, 1, allocator<complex>> out(extensions(in));
	auto p = multi::fftw_plan_dft(in, out, FFTW_FORWARD, FFTW_PRESERVE_INPUT);
	fftw_execute(p);
	fftw_destroy_plan(p);
	BOOST_REQUIRE( (power(in) - power(out)/num_elements(out)) < 1e-12 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_power){
	multi::array<complex, 2> in({N, N});
	std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2);
	multi::array<complex, 2> out(extensions(in));
	auto p = multi::fftw_plan_dft(in, out, FFTW_FORWARD, FFTW_PRESERVE_INPUT);
	fftw_execute(p); fftw_destroy_plan(p);
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-12 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_power_plan){
	multi::array<complex, 2> in({16, 16});
	std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2);
	multi::array<complex, 2> out(extensions(in));
	multi::fftw::plan const p{in, out, FFTW_FORWARD, FFTW_PRESERVE_INPUT};
	p(); //execute(p); //p.execute();
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-8 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_power_dft){
	multi::array<complex, 2> in({16, 16}); std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2);
	multi::array<complex, 2> out(extensions(in));
	multi::fftw::dft(in, out, multi::fftw::forward);
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-8 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_power_dft_out){
	multi::array<complex, 2> in({16, 16}); std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2);
	auto out = multi::fftw::dft(in, multi::fftw::forward);
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-8 );
}

BOOST_AUTO_TEST_CASE(fftw_2D_power_dft_out_default){
	multi::array<complex, 2> in({16, 16}); std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2);
	auto out = multi::fftw::dft(in, fftw::forward);
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-8 );
}

/*
BOOST_AUTO_TEST_CASE(fftw_2D_carray_power){
	int const N = 16;
	complex in[N][N];
	using multi::data_elements;	using multi::num_elements;
	std::iota(data_elements(in), data_elements(in) + num_elements(in), 1.2);
	complex out[N][N];
	auto p = multi::fftw_plan_dft(in, out, FFTW_FORWARD | FFTW_PRESERVE_INPUT);
	fftw_execute(p); fftw_destroy_plan(p);
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-8 );
}
*/

BOOST_AUTO_TEST_CASE(fftw_3D_power){
	multi::array<complex, 3> in({4, 4, 4}); std::iota(in.data_elements(), in.data_elements() + in.num_elements(), 1.2);
	multi::array<complex, 3> out = fftw::dft(in, fftw::forward);
	BOOST_REQUIRE( std::abs(power(in) - power(out)/num_elements(out)) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(fftw_3D_power_in_place){
	multi::array<complex, 3> io({4, 4, 4}); std::iota(io.data_elements(), io.data_elements() + io.num_elements(), 1.2);
	auto powerin = power(io);
	fftw::dft_inplace(io, fftw::forward);
	BOOST_REQUIRE( powerin - power(io)/num_elements(io) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(fftw_3D_power_in_place_over_ref_inplace){
	multi::array<complex, 3> io({4, 4, 4}); std::iota(io.data_elements(), io.data_elements() + io.num_elements(), 1.2);
	auto powerin = power(io);
//	fftw::dft_inplace(multi::array_ref<complex, 3>(io.data(), io.extensions()), fftw::forward);
	fftw::dft_inplace(multi::array_ref<complex, 3>(data_elements(io), extensions(io)), fftw::forward);
	BOOST_REQUIRE( powerin - power(io)/num_elements(io) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(fftw_3D_power_out_of_place_over_ref){
	multi::array<complex, 3> in({4, 4, 4}); std::iota(data_elements(in), data_elements(in)+num_elements(in), 1.2);
	multi::array<complex, 3> out({4, 4, 4});
	multi::array_ref<complex, 3>(data_elements(out), extensions(out)) = fftw::dft(multi::array_cref<complex, 3>(data_elements(in), extensions(in)), fftw::forward);
	BOOST_REQUIRE( power(in) - power(out)/num_elements(out) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(fftw_3D_power_out_of_place_over_temporary){
	double powerin;
	auto f = [&](){
		multi::array<complex, 3> in({4, 4, 4}); 
		std::iota(data_elements(in), data_elements(in)+num_elements(in), 1.2);
		powerin = power(in);
		return in;
	};
	auto out = fftw::dft(f(), fftw::forward);
	BOOST_REQUIRE( std::abs(powerin - power(out)/num_elements(out)) < 1e-10 );
}

#if 0
{
	auto in3 = [](){
		multi::array<complex, 3> in3({N, N, N});
		std::iota(in3.data(), in3.data() + in3.num_elements(), 1.2);
		return in3;
	}();
	multi::array<complex, 3> out3(extensions(in3));
	auto p = 
//		multi::fftw_plan_dft_3d(in3, out3, FFTW_FORWARD);
		multi::fftw_plan_dft(in3, out3, FFTW_FORWARD| FFTW_PRESERVE_INPUT)
	;
	fftw_execute(p);
	assert( power_diff(in3, out3) < 1e-3 );	
	fftw_execute_dft(p, in3, out3);//, out3);
	fftw_destroy_plan(p);
	assert( power_diff(in3, out3) < 1e-3 );
}
{
	auto in4 = [](){
		multi::array<complex, 4> in4({5, 6, 7, 8});
		std::iota(in4.data(), in4.data() + in4.num_elements(), 10.2);
		return in4;
	}();
	multi::array<complex, 4> out4(extensions(in4));
	auto p = multi::fftw_plan_dft(in4, out4, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
	fftw_execute(p);
	fftw_destroy_plan(p);
	assert( power_diff(in4, out4) < 1e-3 );
}
#endif
#endif
#endif

