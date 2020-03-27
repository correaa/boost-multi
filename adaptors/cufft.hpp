#ifdef COMPILATION_INSTRUCTIONS//-*-indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4;-*-
nvcc               -D_TEST_MULTI_ADAPTORS_CUFFT -x c++ $0 -o $0x          -lcufft `pkg-config --libs fftw3` -lboost_timer -lboost_unit_test_framework&&$0x&&rm $0x
clang++ -std=c++14 -D_TEST_MULTI_ADAPTORS_CUFFT -x c++ $0 -o $0x -lcudart -lcufft `pkg-config --libs fftw3` -lboost_timer -lboost_unit_test_framework&&$0x&&rm $0x
exit
#endif
// © Alfredo A. Correa 2020

#ifndef MULTI_ADAPTORS_CUFFTW_HPP
#define MULTI_ADAPTORS_CUFFTW_HPP

#include "../../multi/utility.hpp"
#include "../../multi/array.hpp"
#include "../../multi/config/NODISCARD.hpp"

#include<numeric>

#include<experimental/tuple>
#include<experimental/array>

#include<cufft.h>
//#include<cuda_runtime.h> // cudaDeviceSynchronize

namespace boost{
namespace multi{
namespace cufft{

class sign{
	int impl_;
public:
	sign() = default;
	constexpr sign(int i) : impl_{i}{}
	constexpr operator int() const{return impl_;}
};

constexpr sign forward{CUFFT_FORWARD};
constexpr sign none{0};
constexpr sign backward{CUFFT_INVERSE};

class plan{
	using complex_type = cufftDoubleComplex;
	complex_type const* idata_     = nullptr;
	complex_type*       odata_     = nullptr;
	int                 direction_ = 0;
	cufftHandle h_;
	plan() = default;
	plan(plan const&) = delete;
	plan(plan&& other) : h_{std::exchange(other.h_, {})}{} // needed in <=C++14 for return
	void ExecZ2Z(complex_type const* idata, complex_type* odata, int direction) const{
		assert(idata_ and odata_); assert(direction_!=0);
		cufftResult r = ::cufftExecZ2Z(h_, const_cast<complex_type*>(idata), odata, direction); 
		switch(r){
			case CUFFT_SUCCESS        : break;// "cuFFT successfully executed the FFT plan."
			case CUFFT_INVALID_PLAN   : throw std::runtime_error{"The plan parameter is not a valid handle."};
		//	case CUFFT_ALLOC_FAILED   : throw std::runtime_error{"CUFFT failed to allocate GPU memory."};
		//	case CUFFT_INVALID_TYPE   : throw std::runtime_error{"The user requests an unsupported type."};
			case CUFFT_INVALID_VALUE  : throw std::runtime_error{"At least one of the parameters idata, odata, and direction is not valid."};
			case CUFFT_INTERNAL_ERROR : throw std::runtime_error{"Used for all internal driver errors."};
			case CUFFT_EXEC_FAILED    : throw std::runtime_error{"CUFFT failed to execute an FFT on the GPU."};
			case CUFFT_SETUP_FAILED   : throw std::runtime_error{"The cuFFT library failed to initialize."};
		//	case CUFFT_INVALID_SIZE   : throw std::runtime_error{"The user specifies an unsupported FFT size."};
		//	case CUFFT_UNALIGNED_DATA : throw std::runtime_error{"Unaligned data."};
		//	case CUFFT_INCOMPLETE_PARAMETER_LIST: throw std::runtime_error{"Incomplete parameter list."};
		//	case CUFFT_INVALID_DEVICE : throw std::runtime_error{"Invalid device."};
		//	case CUFFT_PARSE_ERROR    : throw std::runtime_error{"Parse error."};
		//	case CUFFT_NO_WORKSPACE   : throw std::runtime_error{"No workspace."};
		//	case CUFFT_NOT_IMPLEMENTED: throw std::runtime_error{"Not implemented."};
		//	case CUFFT_LICENSE_ERROR  : throw std::runtime_error{"License error."};
		//	case CUFFT_NOT_SUPPORTED  : throw std::runtime_error{"CUFFT_NOT_SUPPORTED"};
			default                   : throw std::runtime_error{"cufftExecZ2Z unknown error"};
		}
	//	if(cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error{"Cuda error: Failed to synchronize"};
	}
	void swap(plan& other){ 
		using std::swap;
		swap(idata_, other.idata_);
		swap(odata_, other.odata_);
		swap(direction_, other.direction_);
		swap(h_, other.h_);
	}
public:
	plan& operator=(plan other){swap(other); return *this;}
	void operator()() const{ExecZ2Z(idata_, odata_, direction_);}
	template<class I, class O>
	O&& execute_dft(I&& i, O&& o, int direction) const{
		ExecZ2Z(
			const_cast<complex_type*>(reinterpret_cast<complex_type const*>(base(i))),
			const_cast<complex_type*>(reinterpret_cast<complex_type const*>(base(o))),
			direction
		);
		return std::forward<O>(o);
	}
	template<class I, class O>
	void execute_dft(I&& i, O&& o) const{execute_dft(std::forward<I>(i), std::forward<O>(o), direction_);}
	~plan(){if(h_) cufftDestroy(h_);}
	using size_type = int;
	using ssize_type = int;

	template<class I, class O, std::enable_if_t<I::dimensionality <= 4, int> =0, 
		typename = decltype(raw_pointer_cast(base(std::declval<I const&>())), reinterpret_cast<complex_type*      >(raw_pointer_cast(base(std::declval<O&&>()))))
	>
	plan(I const& i, O&& o, sign s) : 
		idata_{                          reinterpret_cast<complex_type const*>(raw_pointer_cast(base(i))) },
		odata_{const_cast<complex_type*>(reinterpret_cast<complex_type*      >(raw_pointer_cast(base(o))))},
		direction_{s}
	{
		assert( CUFFT_FORWARD == s or CUFFT_INVERSE == s or s == 0 );
		assert( sizes(i) == sizes(o) );
#if 0
		assert( stride(i) == 1 );
		assert( stride(o) == 1 );
		switch(::cufftPlan1d(&h_, size(i), CUFFT_Z2Z, 1)){
			case CUFFT_SUCCESS        : break;// "cuFFT successfully executed the FFT plan."
			case CUFFT_ALLOC_FAILED   : throw std::runtime_error{"CUFFT failed to allocate GPU memory."};
			case CUFFT_INVALID_VALUE  : throw std::runtime_error{"At least one of the parameters idata, odata, and direction is not valid."};
			case CUFFT_INTERNAL_ERROR : throw std::runtime_error{"Used for all internal driver errors."};
			case CUFFT_SETUP_FAILED   : throw std::runtime_error{"The cuFFT library failed to initialize."};
			case CUFFT_INVALID_SIZE   : throw std::runtime_error{"The user specifies an unsupported FFT size."};
			default                   : throw std::runtime_error{"cufftPlanMany unknown error"};
		}
#endif
		using namespace std::experimental;
		auto ion = apply([](auto... t){return make_array<size_type>(t...);}, sizes(i));

		auto istrides = apply([](auto... t){return make_array<ssize_type>(t...);}, strides(i));
		auto ostrides = apply([](auto... t){return make_array<ssize_type>(t...);}, strides(o));

		int istride = istrides.back();
		auto iembed = istrides;
		std::adjacent_difference(begin(istrides), end(istrides), begin(iembed), [](auto a, auto b){assert(b%a==0); return b/a;});

		int ostride = ostrides.back();
		auto oembed = ostrides;
		std::adjacent_difference(begin(ostrides), end(ostrides), begin(oembed), [](auto a, auto b){assert(b%a==0); return b/a;});

		direction_ = s;
		idata_ =                           reinterpret_cast<complex_type const*>(raw_pointer_cast(base(i))) ;
		odata_ = const_cast<complex_type*>(reinterpret_cast<complex_type*      >(raw_pointer_cast(base(o))));

		switch(::cufftPlanMany(
			/*cufftHandle *plan*/ &h_, 
			/*int rank*/          ion.size(), 
			/*int *n*/            ion.data(), //	/*NX*/      last - first,
			/*int *inembed*/      iembed.data(),
			/*int istride*/       istride, 
			/*int idist*/         1, //stride(first), 
			/*int *onembed*/      oembed.data(), 
			/*int ostride*/       ostride, 
			/*int odist*/         1, //stride(d_first), 
			/*cufftType type*/    CUFFT_Z2Z, 
			/*int batch*/         1 //BATCH
		)){
			case CUFFT_SUCCESS        : break;// "cuFFT successfully executed the FFT plan."
			case CUFFT_ALLOC_FAILED   : throw std::runtime_error{"CUFFT failed to allocate GPU memory."};
			case CUFFT_INVALID_VALUE  : throw std::runtime_error{"At least one of the parameters idata, odata, and direction is not valid."};
			case CUFFT_INTERNAL_ERROR : throw std::runtime_error{"Used for all internal driver errors."};
			case CUFFT_SETUP_FAILED   : throw std::runtime_error{"The cuFFT library failed to initialize."};
			case CUFFT_INVALID_SIZE   : throw std::runtime_error{"The user specifies an unsupported FFT size."};
			default                   : throw std::runtime_error{"cufftPlanMany unknown error"};
		}
	}
	template<class It1, class It2>
	static auto many(It1 first, It1 last, It2 d_first, int sign = 0, unsigned = 0)
	->std::decay_t<decltype(const_cast<complex_type*>(reinterpret_cast<complex_type*>(raw_pointer_cast(base(d_first)))), std::declval<plan>())>
	{
		assert( CUFFT_FORWARD == sign or CUFFT_INVERSE == sign or sign == 0 );
		assert( size(*first) == size(*last) );
		using namespace std::experimental;
		auto ion = apply([](auto... t){return make_array<size_type>(t...);}, sizes(*  first));

		assert( stride(first)==stride(last) );
		auto istrides = apply([](auto... t){return make_array<ssize_type>(t...);}, strides(*  first));
		auto ostrides = apply([](auto... t){return make_array<ssize_type>(t...);}, strides(*d_first));

		int istride = istrides.back();
		auto iembed = istrides;
		std::adjacent_difference(begin(istrides), end(istrides), begin(iembed), [](auto a, auto b){assert(b%a==0); return b/a;});//std::divides<>{});

		int ostride = ostrides.back();
		auto oembed = ostrides;
		std::adjacent_difference(begin(ostrides), end(ostrides), begin(oembed), [](auto a, auto b){assert(b%a==0); return b/a;});

		plan ret;
		ret.direction_ = sign;
		ret.idata_ =                           reinterpret_cast<complex_type const*>(raw_pointer_cast(base(  first))) ;
		ret.odata_ = const_cast<complex_type*>(reinterpret_cast<complex_type*      >(raw_pointer_cast(base(d_first))));

		switch(::cufftPlanMany(
			/*cufftHandle *plan*/ &ret.h_, 
			/*int rank*/          ion.size(), 
			/*int *n*/            ion.data(), //	/*NX*/      last - first,
			/*int *inembed*/      iembed.data(),
			/*int istride*/       istride, 
			/*int idist*/         stride(first), 
			/*int *onembed*/      oembed.data(), 
			/*int ostride*/       ostride, 
			/*int odist*/         stride(d_first), 
			/*cufftType type*/    CUFFT_Z2Z, 
			/*int batch*/         last - first //BATCH
		)){
			case CUFFT_SUCCESS        : break;// "cuFFT successfully executed the FFT plan."
		//	case CUFFT_INVALID_PLAN   : throw std::runtime_error{"The plan parameter is not a valid handle."};
			case CUFFT_ALLOC_FAILED   : throw std::runtime_error{"CUFFT failed to allocate GPU memory."};
		//	case CUFFT_INVALID_TYPE   : throw std::runtime_error{"The user requests an unsupported type."};
			case CUFFT_INVALID_VALUE  : throw std::runtime_error{"At least one of the parameters idata, odata, and direction is not valid."};
			case CUFFT_INTERNAL_ERROR : throw std::runtime_error{"Used for all internal driver errors."};
		//	case CUFFT_EXEC_FAILED    : throw std::runtime_error{"CUFFT failed to execute an FFT on the GPU."};
			case CUFFT_SETUP_FAILED   : throw std::runtime_error{"The cuFFT library failed to initialize."};
			case CUFFT_INVALID_SIZE   : throw std::runtime_error{"The user specifies an unsupported FFT size."};
		//	case CUFFT_UNALIGNED_DATA : throw std::runtime_error{"Unaligned data."};
		//	case CUFFT_INCOMPLETE_PARAMETER_LIST: throw std::runtime_error{"Incomplete parameter list."};
		//	case CUFFT_INVALID_DEVICE : throw std::runtime_error{"Invalid device."};
		//	case CUFFT_PARSE_ERROR    : throw std::runtime_error{"Parse error."};
		//	case CUFFT_NO_WORKSPACE   : throw std::runtime_error{"No workspace."};
		//	case CUFFT_NOT_IMPLEMENTED: throw std::runtime_error{"Not implemented."};
		//	case CUFFT_LICENSE_ERROR  : throw std::runtime_error{"License error."};
		//	case CUFFT_NOT_SUPPORTED  : throw std::runtime_error{"CUFFT_NOT_SUPPORTED"};
			default                   : throw std::runtime_error{"cufftPlanMany unknown error"};
		}
		return ret;
	}
};

template<typename In, class Out>
auto dft(In const& i, Out&& o, int s)
->decltype(plan{i, std::forward<Out>(o), s}()){
	return plan{i, std::forward<Out>(o), s}();}

template<typename In, typename R = multi::array<typename In::element_type, In::dimensionality, decltype(get_allocator(std::declval<In>()))>>
NODISCARD("when first argument is const")
R dft(In const& i, int s){
	R ret(extensions(i), get_allocator(i));
	dft(i, ret, s);
	return ret;
}

template<typename It1, typename It2>
auto many_dft(It1 first, It1 last, It2 d_first, sign s)
->decltype(plan::many(first, last, d_first, s)(), d_first + (last - first)){
	return plan::many(first, last, d_first, s)(), d_first + (last - first);}

}



}}

#include "../adaptors/cuda.hpp"

namespace boost{
namespace multi{
namespace fft{
	using cufft::many_dft;
	using cufft::dft;

	template<dimensionality_type D, class Complex, class... TP1, class... R1, class It2>
	auto many_dft(
		array_iterator<Complex, D, memory::cuda::managed::ptr<TP1...>, R1...> first,
		array_iterator<Complex, D, memory::cuda::managed::ptr<TP1...>, R1...> last,
		It2 d_first, int direction
	)
	->decltype(cufft::many_dft(first, last, d_first, direction)){
		return cufft::many_dft(first, last, d_first, direction);}

	template<class Complex, dimensionality_type D, class... PAs, class... As, class Out>
	auto dft(basic_array<Complex, D, memory::cuda::managed::ptr<PAs...>, As...> const& i, Out&& o, int s)
	->decltype(cufft::dft(i, o, s)){
		return cufft::dft(i, o, s);}

	template<class Complex, dimensionality_type D, class... AAs, class Out>
	auto dft(array<Complex, D, memory::cuda::managed::allocator<AAs...> > const& i, Out&& o, int s)
	->decltype(cufft::dft(i, o, s)){
		return cufft::dft(i, o, s);}

	template<class Complex, dimensionality_type D, class... AAs> NODISCARD("when first argument is const")
	auto dft(array<Complex, D, memory::cuda::managed::allocator<AAs...> > const& i, int s)
	->decltype(cufft::dft(i, s)){
		return cufft::dft(i, s);}

	template<class Complex, dimensionality_type D, class... PAs, class... As> NODISCARD("when first argument is const")
	auto dft(basic_array<Complex, D, memory::cuda::managed::ptr<PAs...>, As...> const& i, int s)
	->decltype(cufft::dft(i, s)){
		return cufft::dft(i, s);}


}

#if 0

namespace fftw{

enum sign: decltype(FFTW_FORWARD){none = 0, forward = FFTW_FORWARD, backward = FFTW_BACKWARD };
enum strategy: decltype(FFTW_ESTIMATE){ estimate = FFTW_ESTIMATE, measure = FFTW_MEASURE };

sign invert(sign s){
	switch(s){
		case sign::forward : return sign::backward;
		case sign::backward: return sign::forward ;
		case sign::none    : return sign::none    ;
	} __builtin_unreachable();
}

template<typename In, class Out>
Out&& dft(In const& i, Out&& o, sign s){
	plan{i, std::forward<Out>(o), s}(i, std::forward<Out>(o));
	return std::forward<Out>(o);
}

template<class In, class Out, std::size_t D = std::decay_t<In>::dimensionality>
Out&& transpose(In const& i, Out&& o){
	return dft(i, std::forward<Out>(o), sign::none);
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
	std::array<bool, D> fwd, /*non,*/ bwd;

	std::transform(begin(w), end(w), begin(fwd), [](auto e){return e==sign::forward;});
	dft(fwd, i, o, sign::forward);

//	std::transform(begin(w), end(w), begin(non), [](auto e){return e==sign::none;});

	std::transform(begin(w), end(w), begin(bwd), [](auto e){return e==sign::backward;}); 
	if(std::accumulate(begin(bwd), end(bwd), false)) dft(bwd, o, o, sign::backward);

	return std::forward<Out>(o);
}


template<typename In, typename R = multi::array<typename In::element_type, In::dimensionality, decltype(get_allocator(std::declval<In>()))>>
NODISCARD("when first argument is const")
R dft(In const& i, sign s){return dft(i, R(extensions(i), get_allocator(i)), s);}

template<typename In, typename R = multi::array<typename In::element_type, In::dimensionality, decltype(get_allocator(std::declval<In>()))>>
NODISCARD("when first argument is const")
R transpose(In const& i){
	return transpose(i, R(extensions(i), get_allocator(i)));
}

template<typename T, dimensionality_type D, class... Args>
decltype(auto) rotate(multi::array<T, D, Args...>& i, int = 1){
	multi::array_ref<T, D, typename multi::array<T, D, Args...>::element_ptr> before(data_elements(i), extensions(i));
//	std::cout << "1. "<< size(i) <<' '<< size(rotated(i)) << std::endl;
	i.reshape(extensions(rotated(before) ));
//	auto x = extensions(i);
//	std::cout << "2. "<< size(i) <<' '<< size(rotated(i)) << std::endl;
	fftw::dft(before, i, sign::none);
//	std::cout << "3. "<< size(i) <<' '<< size(rotated(i)) << std::endl;
	return i;
//	assert( extensions(i) == x );
//	return i;
}

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
#endif

}}

#if _TEST_MULTI_ADAPTORS_CUFFT

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuFFT adaptor"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include <boost/timer/timer.hpp>

#include "../adaptors/cuda.hpp"
#include "../adaptors/fftw.hpp"
#include "../adaptors/cufft.hpp"

#include<complex>
#include<thrust/complex.h>

namespace multi = boost::multi;
using complex = std::complex<double>;
namespace utf = boost::unit_test;

constexpr auto I = complex{0, 1};

BOOST_AUTO_TEST_CASE(cufft_dft_1D_out_of_place, *utf::tolerance(0.00001)){

	multi::array<complex, 1> const in_cpu = {
		1. + 2.*I, 2. + 3. *I, 4. + 5.*I, 5. + 6.*I
	};
	multi::array<complex, 1> fw_cpu(size(in_cpu));
	multi::fft::dft(in_cpu, fw_cpu, multi::fft::forward);
	BOOST_REQUIRE( in_cpu[1] == +2. + 3.*I );
	BOOST_REQUIRE( fw_cpu[2] == -2. - 2.*I );

	multi::cuda::array<complex, 1> const in_gpu = in_cpu;
	multi::cuda::array<complex, 1> fw_gpu(size(in_cpu));
	multi::fft::dft( in_gpu, fw_gpu, multi::fft::forward );
	BOOST_REQUIRE( fw_gpu[3].operator complex() - fw_cpu[3] == 0. );

	multi::cuda::managed::array<complex, 1> const in_mng = in_cpu;
	multi::cuda::managed::array<complex, 1> fw_mng(size(in_cpu));
	multi::fft::dft( in_mng, fw_mng, multi::fft::forward );
	BOOST_REQUIRE( fw_mng[3] - fw_cpu[3] == 0. );
}

BOOST_AUTO_TEST_CASE(cufft_2D, *boost::unit_test::tolerance(0.0001)){
	multi::array<complex, 2> const in_cpu = {
		{ 1. + 2.*I, 9. - 1.*I, 2. + 4.*I},
		{ 3. + 3.*I, 7. - 4.*I, 1. + 9.*I},
		{ 4. + 1.*I, 5. + 3.*I, 2. + 4.*I},
		{ 3. - 1.*I, 8. + 7.*I, 2. + 1.*I},
		{ 31. - 1.*I, 18. + 7.*I, 2. + 10.*I}
	};
	multi::array<complex, 2> fw_cpu(extensions(in_cpu));
	multi::fftw::dft(in_cpu, fw_cpu, multi::fftw::forward);

	multi::cuda::array<complex, 2> const in_gpu = in_cpu;
	multi::cuda::array<complex, 2> fw_gpu(extensions(in_gpu));
	multi::fft::dft(in_gpu, fw_gpu, multi::fft::forward);

	BOOST_TEST( imag(fw_gpu[3][2].operator complex() - fw_cpu[3][2]) == 0. );

	auto fw2_gpu = multi::fft::dft(in_gpu, multi::fft::forward);
	BOOST_TEST( imag(fw2_gpu[3][1].operator complex() - fw_cpu[3][1]) == 0. );

	multi::cuda::managed::array<complex, 2> const in_mng = in_cpu;
	multi::cuda::managed::array<complex, 2> fw_mng(extensions(in_gpu));
	multi::fft::dft(in_mng, fw_mng, multi::fft::forward);

	BOOST_TEST( imag(fw_mng[3][2] - fw_cpu[3][2]) == 0. );

	auto fw2_mng = multi::fft::dft(in_mng, multi::fft::forward);
	BOOST_TEST( imag(fw2_mng[3][1] - fw_cpu[3][1]) == 0. );

}

BOOST_AUTO_TEST_CASE(cufft_3D_timing, *boost::unit_test::tolerance(0.0001)){

	{
		multi::array<complex, 3> const in_cpu({400, 400, 400}, 10.); 
		BOOST_ASSERT( in_cpu.num_elements()*sizeof(complex) < 2e9 );
		multi::array<complex, 3> fw_cpu(extensions(in_cpu), 99.);
		{
			boost::timer::auto_cpu_timer t;  // 1.041691s wall, 1.030000s user + 0.000000s system = 1.030000s CPU (98.9%)
			multi::fftw::dft(in_cpu, fw_cpu, multi::fftw::forward);

		//	BOOST_TEST( fw_cpu[8][9][10] != 99. );
		}
	}
	{
		multi::cuda::array<complex, 3> const in_gpu({400, 400, 400}, 10.); 
		multi::cuda::array<complex, 3> fw_gpu(extensions(in_gpu), 99.);
		{
			boost::timer::auto_cpu_timer t; //  0.208237s wall, 0.200000s user + 0.010000s system = 0.210000s CPU (100.8%)
			multi::cufft::dft(in_gpu, fw_gpu, multi::fftw::forward);

		//	BOOST_TEST( fw_gpu[8][9][10].operator complex() != 99. );
		}
	}

}

BOOST_AUTO_TEST_CASE(cufft_4D_many, *utf::tolerance(0.00001) ){

	auto const in_cpu = []{
		multi::array<complex, 4> ret({45, 18, 32, 16});
		std::generate(
			ret.data_elements(), ret.data_elements() + ret.num_elements(), 
			[](){return complex{std::rand()*1./RAND_MAX, std::rand()*1./RAND_MAX};}
		);
		return ret;
	}();

	multi::cuda::array<complex, 4> const in = in_cpu;
	multi::cuda::array<complex, 4>       out(extensions(in));

	multi::fft::many_dft(begin(unrotated(in)), end(unrotated(in)), begin(unrotated(out)), +1);

	multi::array<complex, 4> out_cpu(extensions(in));
	multi::fft::many_dft(begin(unrotated(in_cpu)), end(unrotated(in_cpu)), begin(unrotated(out_cpu)), +1);

	BOOST_TEST( imag( out[5][4][3][2].operator std::complex<double>() - out_cpu[5][4][3][2]) == 0. );

}

#endif
#endif

