#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&`#nvcc -x cu --expt-relaxed-constexpr`$CXX -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_ADAPTORS_FFTW $0.cpp -o $0x -lcudart `pkg-config --libs fftw3` -lboost_unit_test_framework&&$0x&&rm $0x $0.cpp;exit
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
	template<class T> auto alignment_of(T* p){return fftw_alignment_of((double*)p);}
}

#if 0
template<typename Size>
auto fftw_plan_dft_1d(
	Size N, 
	std::complex<double> const* in, std::complex<double>* out, int sign, 
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
	std::complex<double>* in, std::complex<double>* out, int sign, 
	unsigned flags = FFTW_ESTIMATE
){
	assert( fftw::alignment_of(in) == fftw::alignment_of(out) );
	return ::fftw_plan_dft_1d(N, (fftw_complex*)in, (fftw_complex*)out, sign, flags);
}

template<typename Size>
auto fftw_plan_dft_2d(
	Size N1, Size N2, 
	std::complex<double> const* in, std::complex<double>* out, int sign, 
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
	std::complex<double>* in, std::complex<double>* out, int sign, 
	unsigned flags = FFTW_ESTIMATE
){
	assert(fftw_alignment_of((double*)in) == fftw_alignment_of((double*)out));
	return ::fftw_plan_dft_2d(N1, N2, (fftw_complex*)in, (fftw_complex*)out, sign, flags);
}

template<typename Size>
auto fftw_plan_dft_3d(
	Size N1, Size N2, Size N3, 
	std::complex<double>* in, std::complex<double>* out, int sign, 
	unsigned flags = FFTW_ESTIMATE
){
	assert(fftw_alignment_of((double*)in) == fftw_alignment_of((double*)out));
	return ::fftw_plan_dft_3d(N1, N2, N3, (fftw_complex*)in, (fftw_complex*)out, sign, flags);
}
template<typename Size>
auto fftw_plan_dft_3d(
	Size N1, Size N2, Size N3, 
	std::complex<double> const* in, std::complex<double>* out, int sign, 
	unsigned flags = FFTW_ESTIMATE
){
	assert( flags & FFTW_PRESERVE_INPUT );
	assert(fftw_alignment_of((double*)in) == fftw_alignment_of((double*)out));
	return ::fftw_plan_dft_3d(N1, N2, N3, (fftw_complex*)in, (fftw_complex*)out, sign, flags | FFTW_PRESERVE_INPUT);
}
#endif

#if 0
template<typename Rank>
auto fftw_plan_dft(
	Rank r, int* ns, 
	std::complex<double>* in, std::complex<double>* out, 
	int sign, unsigned flags = FFTW_ESTIMATE
){
	assert(fftw_alignment_of((double*)in) == fftw_alignment_of((double*)out));
	return ::fftw_plan_dft(r, ns, (fftw_complex*)in, (fftw_complex*)out, sign, flags);
}
template<typename RankType>
auto fftw_plan_dft(
	RankType r, int* ns, 
	std::complex<double> const* in, std::complex<double>* out, 
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
#endif

#if 0
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
#endif

template<class T, class Tpl> constexpr auto to_array(Tpl const& t){
	return detail::to_array_impl<T>(t, std::make_index_sequence<std::tuple_size<Tpl>{}>{});
}

#if(__cpp_if_constexpr>=201606)
//https://stackoverflow.com/a/35110453/225186
template<class T> constexpr std::remove_reference_t<T> _constx(T&&t){return t;}
#define logic_assert(C, M) \
	if constexpr(noexcept(_constx(C))) static_assert((C), M); else assert((C)&& M);
#else
#define logic_assert(ConditioN, MessagE) assert(ConditioN && MessagE);
#endif

template<typename It1, class It2>
fftw_plan fftw_plan_many_dft(It1 first, It1 last, It2 d_first, int sign, unsigned flags = FFTW_ESTIMATE){
	assert(strides(*first) == strides(*last));
	assert(sizes(*first)==sizes(*d_first));
	auto ion      = to_array<int>(sizes(*first));
	auto istrides = to_array<int>(strides(*first));
	auto ostrides = to_array<int>(strides(*d_first));

	int istride = istrides.back();
	auto inembed = istrides; inembed.fill(0);
	for(int i = 1; i != inembed.size(); ++i){
		assert( istrides[i-1]%istrides[i] == 0 );
		inembed[i]=istrides[i-1]/istrides[i];
	}

	int ostride = ostrides.back();
	auto onembed = ostrides; onembed.fill(0);
	for(int i = 1; i != onembed.size(); ++i){
		assert( ostrides[i-1]%ostrides[i] == 0 );
		onembed[i]=ostrides[i-1]/ostrides[i];
	}

	return ::fftw_plan_many_dft(
		/*int rank*/ ion.size(), 
		/*const int* n*/ ion.data(),
		/*int howmany*/ last - first,
		/*fftw_complex * in */ reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(static_cast<std::complex<double> const*>(base(first)))), 
		/*const int *inembed*/ inembed.data(),
		/*int*/ istride, 
		/*int idist*/ stride(first),
		/*fftw_complex * out */ reinterpret_cast<fftw_complex*>(static_cast<std::complex<double>*>(base(d_first))),
		/*const int *onembed*/ onembed.data(),
		/*int*/ ostride, 
		/*int odist*/ stride(d_first),
		/*int*/ sign, /*unsigned*/ flags
	);
}

template<class In, class Out, std::size_t D = std::decay_t<In>::dimensionality,
typename = std::enable_if_t<D == std::decay_t<Out>::dimensionality>>
fftw_plan fftw_plan_dft(std::array<bool, D> which, In&& in, Out&& out, int sign, unsigned flags = FFTW_ESTIMATE){
	using multi::sizes; using multi::strides; assert(sizes(in) == sizes(out));
	auto ion      = to_array<ptrdiff_t>(sizes(in));
	auto istrides = to_array<ptrdiff_t>(strides(in));
	auto ostrides = to_array<ptrdiff_t>(strides(out));
	std::array<fftw_iodim64, D> dims   ; auto l_dims = dims.begin();
	std::array<fftw_iodim64, D> howmany; auto l_howmany = howmany.begin();
	for(int i = 0; i != D; ++i) 
		(which[i]?*l_dims++:*l_howmany++) = fftw_iodim64{ion[i], istrides[i], ostrides[i]};
	return fftw_plan_guru64_dft(
		/*int rank*/ l_dims - dims.begin(), 
		/*const fftw_iodim64 *dims*/ dims.data(), 
		/*int howmany_rank*/ l_howmany - howmany.begin(),
		/*const fftw_iodim *howmany_dims*/ howmany.data(), //nullptr, //howmany_dims.data(), //;//nullptr,
		/*fftw_complex *in*/ const_cast<fftw_complex*>(reinterpret_cast<fftw_complex const*>(base(in))), 
		/*fftw_complex *out*/ reinterpret_cast<fftw_complex*>(base(out)),
		sign, flags | FFTW_ESTIMATE
	);
}

template<class In, class Out, dimensionality_type D = In::dimensionality>
auto fftw_plan_dft(In const& in, Out&& out, int s, unsigned flags = FFTW_ESTIMATE){
	static_assert( D == std::decay_t<Out>::dimensionality , "!");
	using multi::sizes; using multi::strides; assert(sizes(in) == sizes(out));
	auto 
		ion      = to_array<ptrdiff_t>(sizes(in)), 
		istrides = to_array<ptrdiff_t>(strides(in)),
		ostrides = to_array<ptrdiff_t>(strides(out))
	;
	std::array<fftw_iodim64, D> dims;
	for(int i=0; i!=D; ++i) dims[i] = {ion[i], istrides[i], ostrides[i]};
	return fftw_plan_guru64_dft(
		/*int rank*/ s?D:0,
		/*const fftw_iodim64 *dims*/ dims.data(),
		/*int howmany_rank*/ 0,
		/*const fftw_iodim *howmany_dims*/ nullptr, //howmany_dims.data(), //;//nullptr,
		/*fftw_complex *in*/ const_cast<fftw_complex*>(reinterpret_cast<fftw_complex const*>(static_cast<std::complex<double> const*>(base(in)))), 
		/*fftw_complex *out*/ reinterpret_cast<fftw_complex*>(static_cast<std::complex<double>*>(base(out))),
		s, flags
	);
}

//std::complex<double> const* base(std::complex<double> const& c){return &c;}

namespace fftw{

class plan{
	plan() : impl_{nullptr, &fftw_destroy_plan}{}
	std::unique_ptr<std::remove_pointer_t<fftw_plan>, decltype(&fftw_destroy_plan)> impl_;
public:
	plan(plan const&) = delete;//default;
	plan(plan&&) = default;
	template<typename... As> plan(As&&... as) : impl_{fftw_plan_dft(std::forward<As>(as)...), &fftw_destroy_plan}{
		assert(impl_);
	}
	template<typename... As>
	static plan many(As&&... as){
		plan r; r.impl_.reset(fftw_plan_many_dft(std::forward<As>(as)...)); return r;
	}
private:
	void execute() const{fftw_execute(impl_.get());}
	template<class I, class O>
	void execute_dft(I&& i, O&& o) const{
		::fftw_execute_dft(impl_.get(), const_cast<fftw_complex*>(reinterpret_cast<fftw_complex const*>(static_cast<std::complex<double> const*>(base(i)))), reinterpret_cast<fftw_complex*>(static_cast<std::complex<double>*>(base(o))));
	}
	template<class I, class O> void execute(I&& i, O&& o) const{execute_dft(std::forward<I>(i), std::forward<O>(o));}
	friend void execute(plan const& p){p.execute();}
public:
	void operator()() const{execute();}
	template<class I, class O> void operator()(I&& i, O&& o) const{return execute(std::forward<I>(i), std::forward<O>(o));}
	double cost() const{return fftw_cost(impl_.get());}
	auto flops() const{
		struct{double add; double mul; double fma;} r;
		fftw_flops(impl_.get(), &r.add, &r.mul, &r.fma);
		return r;
	}
};

enum sign: decltype(FFTW_FORWARD){none = 0, forward = FFTW_FORWARD, backward = FFTW_BACKWARD };
enum strategy: decltype(FFTW_ESTIMATE){ estimate = FFTW_ESTIMATE, measure = FFTW_MEASURE };

sign invert(sign s){
	switch(s){
		case sign::forward : return sign::backward;
		case sign::backward: return sign::forward;
		case sign::none: return sign::none;
	} __builtin_unreachable();
}

template<typename In, class Out>
auto dft(In const& i, Out&& o, sign s){
	if(s == sign::none) o = i; else plan{i, std::forward<Out>(o), s}(i, std::forward<Out>(o));
	return std::forward<Out>(o);
}

template<typename In, class Out, std::size_t D = std::decay_t<In>::dimensionality>
Out&& dft(std::array<bool, D> which, In const& i, Out&& o, sign s){
	plan{which, i, o, s}();//(i, std::forward<Out>(o)); 
	return std::forward<Out>(o);
}

template<dimensionality_type R, class In, class Out, std::size_t D = std::decay_t<In>::dimensionality>
Out&& dft(In const& i, Out&& o, sign s){
	static_assert( R <= D , "dimension of transpformation cannot be larger than total dimension" );
	std::array<bool, D> which; std::fill(std::fill_n(begin(which), R, false), end(which), true);
	plan{which, i, o, s}();//(i, std::forward<Out>(o)); 
	return std::forward<Out>(o);
}

template<typename In, class Out, std::size_t D = std::decay_t<In>::dimensionality>
auto dft(std::array<sign, D> w, In const& i, Out&& o){
	std::array<bool, D> fwd, bwd;

	std::transform(begin(w), end(w), begin(fwd), [](auto e){return e==sign::forward;});
	dft(fwd, i, o, sign::forward);

	std::transform(begin(w), end(w), begin(bwd), [](auto e){return e==sign::backward;}); 
	if(std::accumulate(begin(bwd), end(bwd), false)) dft(bwd, o, o, sign::backward);

	return std::forward<Out>(o);
}

template<typename It1, typename It2>
auto many_dft(It1 first, It1 last, It2 d_first, int sign){
	plan::many(first, last, d_first, sign)();
	return d_first + (last - first);
}

template<typename In, typename R = multi::array<typename In::element_type, In::dimensionality, decltype(get_allocator(std::declval<In>()))>>
NODISCARD("when first argument is const")
R dft(In const& i, sign s){return dft(i, R(extensions(i), get_allocator(i)), s);}

template<typename In, typename R = multi::array<typename In::element_type, In::dimensionality, decltype(get_allocator(std::declval<In>()))>>
NODISCARD("when first argument is const")
R dft(std::array<bool, In::dimensionality> which, In const& i, sign s){
	return dft(which, i, R(extensions(i), get_allocator(i)), s);
}

template<typename In, typename R = multi::array<typename In::element_type, In::dimensionality, decltype(get_allocator(std::declval<In>()))>>
NODISCARD("when second argument is const")
R dft(std::array<sign, In::dimensionality> which, In const& i){
	return dft(which, i, R(extensions(i), get_allocator(i)));
}

template<dimensionality_type Rank, typename In, typename R = multi::array<typename In::element_type, In::dimensionality, decltype(get_allocator(std::declval<In>()))>>
NODISCARD("when second argument is const")
R dft(In const& i, sign s){
	return dft<Rank>(i, R(extensions(i), get_allocator(i)), s);
}

template<typename T, dimensionality_type D, class... As, typename R = multi::array<T, D, As...>>//typename std::decay_t<In>::element_type, std::decay_t<In>::dimensionality>>
NODISCARD("when first argument can be destroyed")
R dft(multi::array<T, D, As...>&& i, sign s){
	R ret(extensions(i), get_allocator(i));
	plan{i, ret, s, static_cast<unsigned>(fftw::estimate) | FFTW_DESTROY_INPUT}();//(i, ret); // to do destroy input for move iterators
	return ret;
}

template<typename T> auto dft(std::initializer_list<T> il, sign s){return dft(multi::array<T, 1>(il), s);}
template<typename T> auto dft(std::initializer_list<std::initializer_list<T>> il, sign s){return dft(multi::array<T, 2>(il), s);}

template<typename... A> auto dft_forward(A&&... a){return dft(std::forward<A>(a)..., fftw::forward);}
template<typename... A> auto dft_backward(A&&... a){return dft(std::forward<A>(a)..., fftw::backward);}

template<typename T, typename... As> auto dft_forward(As&... as, std::initializer_list<T> il){return dft_forward(std::forward<As>(as)..., multi::array<T, 1>(il));}
template<typename T, typename... As> auto dft_forward(As&... as, std::initializer_list<std::initializer_list<T>> il){return dft_forward(std::forward<As>(as)..., multi::array<T, 2>(il));}

template<typename T, typename... As> auto dft_backward(As&... as, std::initializer_list<T> il){return dft_backward(std::forward<As>(as)..., multi::array<T, 1>(il));}
template<typename T, typename... As> auto dft_backward(As&... as, std::initializer_list<std::initializer_list<T>> il){return dft_backward(std::forward<As>(as)..., multi::array<T, 2>(il));}

template<class In> In&& dft_inplace(In&& i, sign s){
	fftw::plan{i, i, (int)s}();//(i, i); 
	return std::forward<In>(i);
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

#include "../adaptors/cuda.hpp"

namespace{

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
//	multi::fftw::dft({false, false}, in, fwd, multi::fftw::forward);
//	multi::fftw::dft<2>(in, fwd, multi::fftw::forward);
//	multi::fftw::dft({multi::fftw::none, multi::fftw::none}, in, fwd);
	multi::fftw::dft(in, fwd, multi::fftw::none);
	BOOST_REQUIRE( fwd == in );
}

BOOST_AUTO_TEST_CASE(fftw_1D){
	multi::array<complex, 1> in = {1. + 2.*I, 2. + 3. *I, 4. + 5.*I, 5. + 6.*I};
	auto fwd = multi::fftw::dft(in, multi::fftw::forward); // Fourier[in, FourierParameters -> {1, -1}]
//	auto fwd = multi::fftw::dft({multi::fftw::forward}, in); // Fourier[in, FourierParameters -> {1, -1}]
//	auto fwd = multi::fftw::dft<0>(in, multi::fftw::forward); // Fourier[in, FourierParameters -> {1, -1}]

	BOOST_REQUIRE(fwd[2] == -2. - 2.*I);
	BOOST_REQUIRE( in[1] == 2. + 3.*I );

	auto bwd = multi::fftw::dft(in, multi::fftw::backward); // InverseFourier[in, FourierParameters -> {-1, -1}]
	BOOST_REQUIRE(bwd[2] == -2. - 2.*I);
}

/*
BOOST_AUTO_TEST_CASE(fftw_1D_cuda){
	multi::cuda::managed::array<complex, 1> in = {1. + 2.*I, 2. + 3. *I, 4. + 5.*I, 5. + 6.*I};
	auto fwd = multi::fftw::dft(in, multi::fftw::forward); // Fourier[in, FourierParameters -> {1, -1}]
//	auto fwd = multi::fftw::dft(in, multi::fftw::forward); // Fourier[in, FourierParameters -> {1, -1}]
//	auto fwd = multi::fftw::dft({multi::fftw::forward}, in); // Fourier[in, FourierParameters -> {1, -1}]
//	auto fwd = multi::fftw::dft<0>(in, multi::fftw::forward); // Fourier[in, FourierParameters -> {1, -1}]
	BOOST_REQUIRE(fwd[2] == -2. - 2.*I);
	BOOST_REQUIRE( in[1] == 2. + 3.*I );

	auto bwd = multi::fftw::dft(in, multi::fftw::backward); // InverseFourier[in, FourierParameters -> {-1, -1}]
	BOOST_REQUIRE(bwd[2] == -2. - 2.*I);
}
*/

BOOST_AUTO_TEST_CASE(fftw_2D_identity, *boost::unit_test::tolerance(0.0001)){
	multi::array<complex, 2> const in = {
		{ 1. + 2.*I, 9. - 1.*I, 2. + 4.*I},
		{ 3. + 3.*I, 7. - 4.*I, 1. + 9.*I},
		{ 4. + 1.*I, 5. + 3.*I, 2. + 4.*I},
		{ 3. - 1.*I, 8. + 7.*I, 2. + 1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	auto fwd = multi::fftw::dft(in, multi::fftw::none);
//	auto fwd = multi::fftw::dft<0>(in, multi::fftw::none);
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
	using multi::fftw::forward;
	auto fwd = dft(in, forward);
//	auto fwd = multi::fftw::dft<0>(in, forward);
//	auto fwd = multi::fftw::dft({forward, forward}, in);
//	auto fwd = dft({true, true}, in, forward);

	BOOST_TEST( real(fwd[3][1]) == -19.0455 ); // Fourier[in, FourierParameters -> {1, -1}][[4]][[2]]
	BOOST_TEST( imag(fwd[3][1]) == - 2.22717 );

	multi::array<complex, 1> const in0 = { 1. + 2.*I, 9. - 1.*I, 2. + 4.*I};
	using multi::fftw::dft_forward;

	BOOST_REQUIRE( dft_forward(in[0]) == dft_forward(in0) );
//	BOOST_REQUIRE( dft_forward(in[3]) == dft_forward({3.-1.*I, 8.+7.*I, 2.+1.*I}) );
//	BOOST_REQUIRE( dft_forward(rotated(in)[0]) == dft_forward({1.+2.*I, 3.+3.*I, 4. + 1.*I,  3. - 1.*I, 31. - 1.*I}) );
}

BOOST_AUTO_TEST_CASE(fftw_2D_rotated, *boost::unit_test::tolerance(0.0001)){
	multi::array<complex, 2> const in = {
		{ 1. + 2.*I, 9. - 1.*I, 2. + 4.*I},
		{ 3. + 3.*I, 7. - 4.*I, 1. + 9.*I},
		{ 4. + 1.*I, 5. + 3.*I, 2. + 4.*I},
		{ 3. - 1.*I, 8. + 7.*I, 2. + 1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	using multi::fftw::forward;
	auto fwd = dft(in, forward);
//	auto fwd = multi::fftw::dft<0>(in, forward);
//	auto fwd = dft({true, true}, in, forward);
//	auto fwd = multi::fftw::dft({forward, forward}, in);

	using multi::fftw::dft_forward;
//	BOOST_REQUIRE( dft_forward(rotated(in)[0]) == dft_forward({1.+2.*I, 3.+3.*I, 4. + 1.*I,  3. - 1.*I, 31. - 1.*I}) );
//	BOOST_REQUIRE( dft_forward(rotated(in)) == rotated(fwd) );// rotated(fwd) );
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
	multi::fftw::dft<1>(in, out, multi::fftw::forward);
//	dft({false, true}, in, out, multi::fftw::forward);
//	multi::fftw::dft({multi::fftw::none, multi::fftw::forward}, in, out);

	using multi::fftw::dft_forward;
	BOOST_REQUIRE( dft_forward(in[0]) == out[0] );

	multi::fftw::dft({false, true}, rotated(in), rotated(out), multi::fftw::forward);
	BOOST_REQUIRE( dft_forward(rotated(in)[0]) == rotated(out)[0] );

	multi::fftw::dft({false, false}, rotated(in), rotated(out), multi::fftw::forward);
	BOOST_REQUIRE( in == out );

	
	multi::fftw::many_dft(begin(in), end(in), begin(out), multi::fftw::forward);
	using multi::fftw::dft_forward;
	BOOST_REQUIRE( dft_forward(in[0]) == out[0] );
}

BOOST_AUTO_TEST_CASE(fftw_many1_from_2){
	multi::array<complex, 2> in({3, 10}); randomizer<complex>{}(in);
	multi::array<complex, 2> out({3, 10});
	fftw::dft({false, true}, in, out, multi::fftw::forward);

	multi::array<complex, 2> out2({3, 10});
	for(int i = 0; i!=size(in); ++i)
		fftw::dft(in[i], out2[i], multi::fftw::forward);

	BOOST_REQUIRE(out2 == out);
}

BOOST_AUTO_TEST_CASE(fftw_many2_from_3){
	multi::array<complex, 3> in({3, 5, 6}); randomizer<complex>{}(in);
	multi::array<complex, 3> out({3, 5, 6});
	fftw::dft({false, true, true}, in, out, multi::fftw::forward);

	multi::array<complex, 3> out2({3, 5, 6});
	for(int i = 0; i!=size(in); ++i)
		fftw::dft(in[i], out2[i], multi::fftw::forward);

	BOOST_REQUIRE(out2 == out);
}

BOOST_AUTO_TEST_CASE(fftw_many2_from_2){
	multi::array<complex, 2> in({5, 6}); randomizer<complex>{}(in);
	multi::array<complex, 2> out({5, 6});
	fftw::dft({true, true}, in, out, multi::fftw::forward);

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
	auto fwd = multi::fftw::dft({true, true, true, true}, in, multi::fftw::forward);
	BOOST_REQUIRE(in[2][3][4][5] == 99.);
}

BOOST_AUTO_TEST_CASE(fftw_4D_many){

	multi::array<complex, 4> in({97, 95, 101, 10});
	in[2][3][4][5] = 99.;
	auto fwd = multi::fftw::dft({true, true, true, false}, in, multi::fftw::forward);
	BOOST_REQUIRE( in[2][3][4][5] == 99. );

	multi::array<complex, 4> out(extensions(in));
	multi::fftw::many_dft(begin(unrotated(in)), end(unrotated(in)), begin(unrotated(out)), multi::fftw::forward);
	BOOST_REQUIRE( fwd == out );

//	multi::array<complex, 4> out2({10, 97, 95, 101});
//	multi::fftw::many_dft(begin(unrotated(in)), end(unrotated(in)), begin(out2), multi::fftw::forward);
//	BOOST_REQUIRE( fwd == rotated(out2) );

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

