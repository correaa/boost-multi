// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2020-2022 Alfredo A. Correa

#ifndef MULTI_ADAPTORS_CUFFTW_HPP
#define MULTI_ADAPTORS_CUFFTW_HPP

#include "../config/MARK.hpp"

#include "../adaptors/../utility.hpp"
#include "../adaptors/../array.hpp"
#include "../adaptors/../config/NODISCARD.hpp"

#include "../adaptors/cuda.hpp"

#include<numeric>

#include<tuple> // std::apply
#include<array>

#include<vector>

#include "../complex.hpp"

#include<cufft.h>

namespace boost{
namespace multi{
namespace cufft{

class sign {
	int impl_ = 0;

 public:
	sign() = default;
	constexpr sign(int i) : impl_{i} {}
	constexpr operator int() const {return impl_;}
};

constexpr sign forward{CUFFT_FORWARD};
constexpr sign none{0};
constexpr sign backward{CUFFT_INVERSE};

static_assert(forward != none and none != backward and backward != forward, "!");

template<dimensionality_type DD = -1>
struct plan {
	using complex_type = cufftDoubleComplex;
	cufftHandle h_;
	std::array<std::pair<bool, fftw_iodim64>, DD + 1> which_iodims_{};
	int first_howmany_;
	int sign_ = 0;

public:
	plan(plan&& other) :
		h_{std::exchange(other.h_, {})},
		which_iodims_{other.which_iodims_},
		first_howmany_{other.first_howmany_}
	{}

    template<
		class ILayout, class OLayout, dimensionality_type D = std::decay_t<ILayout>::rank::value,
		class=std::enable_if_t<D == std::decay_t<OLayout>::rank::value>
	>
	plan(std::array<bool, +D> which, ILayout const& in, OLayout const& out, int sign = 0) : sign_{sign} {

		assert(in.sizes() == out.sizes());

		auto const sizes_tuple   = in.sizes();
		auto const istride_tuple = in.strides();
		auto const ostride_tuple = out.strides();

		using boost::multi::detail::get;
		auto which_iodims = std::apply([](auto... elems) {
			return std::array<std::pair<bool, fftw_iodim64>, sizeof...(elems) + 1>{  // TODO(correaa) added one element to avoid problem with gcc 13 static analysis (out-of-bounds)
				std::pair<bool, fftw_iodim64>{
					get<0>(elems),
					fftw_iodim64{get<1>(elems), get<2>(elems), get<3>(elems)}
				}...,
				std::pair<bool, fftw_iodim64>{}
			};
		}, boost::multi::detail::tuple_zip(which, sizes_tuple, istride_tuple, ostride_tuple));

		std::stable_sort(which_iodims.begin(), which_iodims.end() - 1, [](auto const& a, auto const& b){return get<1>(a).is > get<1>(b).is;});

		auto const part = std::stable_partition(which_iodims.begin(), which_iodims.end() - 1, [](auto elem) {return std::get<0>(elem);});

		std::array<fftw_iodim64, D> dims{};
		auto const dims_end         = std::transform(which_iodims.begin(), part,         dims.begin(), [](auto elem) {return elem.second;});

		std::array<fftw_iodim64, D> howmany_dims{};
		auto const howmany_dims_end = std::transform(part, which_iodims.end() -1, howmany_dims.begin(), [](auto elem) {return elem.second;});

		which_iodims_ = which_iodims;
		first_howmany_ = part - which_iodims.begin();

		////////////////////////////////////////////////////////////////////////

		std::array<int, D> istrides{};
		std::array<int, D> ostrides{};
		std::array<int, D> ion{};

		auto const istrides_end = std::transform(dims.begin(), dims_end, istrides.begin(), [](auto elem) {return elem.is;});
		auto const ostrides_end = std::transform(dims.begin(), dims_end, ostrides.begin(), [](auto elem) {return elem.os;});
		auto const ion_end      = std::transform(dims.begin(), dims_end, ion.begin(),      [](auto elem) {return elem.n;});

		int istride = *(istrides_end -1);
		auto inembed = istrides; inembed.fill(0);
		int ostride = *(ostrides_end -1);
		auto onembed = ostrides; onembed.fill(0);

		for(std::size_t i = 1; i != ion_end - ion.begin(); ++i) {
			assert(ostrides[i-1] >= ostrides[i]);
			assert(ostrides[i-1]%ostrides[i]==0);
			onembed[i]=ostrides[i-1]/ostrides[i];
			assert(istrides[i-1]%istrides[i]==0);
			inembed[i]=istrides[i-1]/istrides[i];
		}


		// while(true) {
		//  if(first_howmany_ < D - 1) {
		//      int nelems = 1;
		//      for(int i = first_howmany_ + 1; i != D; ++i) {nelems *= which_iodims_[i].second.n;}
		//      if(
		//          which_iodims_[first_howmany_].second.is == nelems and
		//          which_iodims_[first_howmany_].second.os == nelems
		//      ) {
		//          which_iodims_[first_howmany_ + 1].second.n *= which_iodims_[first_howmany_].second.n;
		//          ++first_howmany_;
		//      } else {
		//          break;
		//      }
		//  } else {
		//      break;
		//  }
		// }

		if(dims_end == dims.begin()) {throw std::runtime_error{"no ffts in any dimension is not supported"};}

		if(first_howmany_ == D) {
			auto const s = ::cufftPlanMany(
				/*cufftHandle *plan*/ &h_,
				/*int rank*/          dims_end - dims.begin(),
				/*int *n*/            ion.data(),
				/*int *inembed*/      inembed.data(),
				/*int istride*/       istride,
				/*int idist*/         1, //stride(first),
				/*int *onembed*/      onembed.data(),
				/*int ostride*/       ostride,
				/*int odist*/         1, //stride(d_first),
				/*cufftType type*/    CUFFT_Z2Z,
				/*int batch*/         1 //BATCH
			);
			assert( s == CUFFT_SUCCESS );
			if(s != CUFFT_SUCCESS) {throw std::runtime_error{"cufftPlanMany failed"};}
			if(not h_) {throw std::runtime_error{"cufftPlanMany null"};}

			return;
		}
		if(first_howmany_ == D - 1) {
			auto const s = ::cufftPlanMany(
				/*cufftHandle *plan*/ &h_,
				/*int rank*/          dims_end - dims.begin(),
				/*int *n*/            ion.data(),
				/*int *inembed*/      inembed.data(),
				/*int istride*/       istride,
				/*int idist*/         which_iodims_[first_howmany_].second.is,
				/*int *onembed*/      onembed.data(),
				/*int ostride*/       ostride,
				/*int odist*/         which_iodims_[first_howmany_].second.os,
				/*cufftType type*/    CUFFT_Z2Z,
				/*int batch*/         which_iodims_[first_howmany_].second.n
			);
			assert( s == CUFFT_SUCCESS );
			if(s != CUFFT_SUCCESS) {throw std::runtime_error{"cufftPlanMany failed"};}
			if(not h_) {throw std::runtime_error{"cufftPlanMany null"};}
			++first_howmany_;
			return;
		}
		throw std::runtime_error{"cufft not implemented yet"};
	}

 private:
	plan() = default;
	plan(plan const&) = delete;
	// plan(plan&& other)
	// : idata_{std::exchange(other.idata_, nullptr)}
	// , odata_{std::exchange(other.odata_, nullptr)}
	// , direction_{std::exchange(other.direction_, 0)}
	// , h_{std::exchange(other.h_, {})}
	// {}
	void ExecZ2Z(complex_type const* idata, complex_type* odata, int direction) const{
		// ++tl_execute_count;
	//  assert(idata_ and odata_); 
	//  assert(direction_!=0);
		cufftResult r = ::cufftExecZ2Z(h_, const_cast<complex_type*>(idata), odata, direction); 
		switch(r){
			case CUFFT_SUCCESS        : break;// "cuFFT successfully executed the FFT plan."
			case CUFFT_INVALID_PLAN   : throw std::runtime_error{"The plan parameter is not a valid handle."};
		//  case CUFFT_ALLOC_FAILED   : throw std::runtime_error{"CUFFT failed to allocate GPU memory."};
		//  case CUFFT_INVALID_TYPE   : throw std::runtime_error{"The user requests an unsupported type."};
			case CUFFT_INVALID_VALUE  : throw std::runtime_error{"At least one of the parameters idata, odata, and direction is not valid."};
			case CUFFT_INTERNAL_ERROR : throw std::runtime_error{"Used for all internal driver errors."};
			case CUFFT_EXEC_FAILED    : throw std::runtime_error{"CUFFT failed to execute an FFT on the GPU."};
			case CUFFT_SETUP_FAILED   : throw std::runtime_error{"The cuFFT library failed to initialize."};
		//  case CUFFT_INVALID_SIZE   : throw std::runtime_error{"The user specifies an unsupported FFT size."};
		//  case CUFFT_UNALIGNED_DATA : throw std::runtime_error{"Unaligned data."};
		//  case CUFFT_INCOMPLETE_PARAMETER_LIST: throw std::runtime_error{"Incomplete parameter list."};
		//  case CUFFT_INVALID_DEVICE : throw std::runtime_error{"Invalid device."};
		//  case CUFFT_PARSE_ERROR    : throw std::runtime_error{"Parse error."};
		//  case CUFFT_NO_WORKSPACE   : throw std::runtime_error{"No workspace."};
		//  case CUFFT_NOT_IMPLEMENTED: throw std::runtime_error{"Not implemented."};
		//  case CUFFT_LICENSE_ERROR  : throw std::runtime_error{"License error."};
		//  case CUFFT_NOT_SUPPORTED  : throw std::runtime_error{"CUFFT_NOT_SUPPORTED"};
			default                   : throw std::runtime_error{"cufftExecZ2Z unknown error"};
		}
		cudaDeviceSynchronize();
	}

 public:
	template<class IPtr, class OPtr>
	void execute(IPtr idata, OPtr odata, int direction) {
		if(first_howmany_ == DD) {
			ExecZ2Z((complex_type const*)::thrust::raw_pointer_cast(idata), (complex_type*)::thrust::raw_pointer_cast(odata), direction);
			return;
		}
		if(first_howmany_ == DD - 1) {
			if( which_iodims_[first_howmany_].first) {throw std::runtime_error{"logic error"};}
			for(int i = 0; i != which_iodims_[first_howmany_].second.n; ++i) {
				::cufftExecZ2Z(
					h_,
					const_cast<complex_type*>((complex_type const*)::thrust::raw_pointer_cast(idata + i*which_iodims_[first_howmany_].second.is)),
					                          (complex_type      *)::thrust::raw_pointer_cast(odata + i*which_iodims_[first_howmany_].second.os) ,
					direction
				);
			}
			return;
		}
		if(first_howmany_ == DD - 2) {
			if( which_iodims_[first_howmany_ + 1].first) {throw std::runtime_error{"logic error"};}
			for(int i = 0; i != which_iodims_[first_howmany_].second.n; ++i) {
				for(int j = 0; j != which_iodims_[first_howmany_ + 1].second.n; ++j) {
					::cufftExecZ2Z(
						h_,
						const_cast<complex_type*>((complex_type const*)::thrust::raw_pointer_cast(idata + i*which_iodims_[first_howmany_].second.is + j*which_iodims_[first_howmany_ + 1].second.is)),
												  (complex_type      *)::thrust::raw_pointer_cast(odata + i*which_iodims_[first_howmany_].second.os + j*which_iodims_[first_howmany_ + 1].second.os) ,
						direction
					);
				}
			}
			return;
		}
		throw std::runtime_error{"error2"};
	}

	template<class IPtr, class OPtr>
	void execute(IPtr idata, OPtr odata) {execute(idata, odata, sign_);}

	template<class IPtr, class OPtr>
	void operator()(IPtr idata, OPtr odata, int direction) const {
		ExecZ2Z((complex_type const*)::thrust::raw_pointer_cast(idata), (complex_type*)::thrust::raw_pointer_cast(odata), direction);
	}
	template<class I, class O>
	O&& execute_dft(I&& i, O&& o, int direction) const {
		ExecZ2Z(
			const_cast<complex_type*>(reinterpret_cast<complex_type const*>(base(i))),
			const_cast<complex_type*>(reinterpret_cast<complex_type const*>(base(o))),
			direction
		);
		return std::forward<O>(o);
	}

	
	template<class I, class O>
	void execute_dft(I&& i, O&& o, int direction) const{execute_dft(std::forward<I>(i), std::forward<O>(o), direction);}
	~plan() {if(h_) cufftDestroy(h_);}
	using size_type = int;
	using ssize_type = int;

#if 0
	template<
		class ILayout, class OLayout, dimensionality_type D = std::decay_t<ILayout>::rank::value,
		class=std::enable_if_t<D == std::decay_t<OLayout>::rank::value>//,
		// class=decltype(reinterpret_cast<fftw_complex*>(/*static_cast<std::complex<double> *>*/(base(std::declval<Out&>()))))  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) interact with legacy code
	>
	auto cufft_plan_dft(std::array<bool, +D> which, ILayout const& in, OLayout const& out, int sign) -> cufftHandle {
		assert(in.sizes() == out.sizes());

		assert( (sign == -1) or (sign == +1) );

		auto const sizes_tuple   = in.sizes();
		auto const istride_tuple = in.strides();
		auto const ostride_tuple = out.strides();

		using boost::multi::detail::get;
		auto which_iodims = std::apply([](auto... elems) {
			return std::array<std::pair<bool, fftw_iodim64>, sizeof...(elems) + 1>{  // TODO(correaa) added one element to avoid problem with gcc 13 static analysis (out-of-bounds)
				std::pair<bool, fftw_iodim64>{
					get<0>(elems),
					fftw_iodim64{get<1>(elems), get<2>(elems), get<3>(elems)}
				}...,
				std::pair<bool, fftw_iodim64>{}
			};
		}, boost::multi::detail::tuple_zip(which, sizes_tuple, istride_tuple, ostride_tuple));

		std::stable_sort(which_iodims.begin(), which_iodims.end() - 1, [](auto const& a, auto const& b){return get<1>(a).is > get<1>(b).is;});

		auto const part = std::stable_partition(which_iodims.begin(), which_iodims.end() - 1, [](auto elem) {return std::get<0>(elem);});

		std::array<fftw_iodim64, D> dims{};
		auto const dims_end         = std::transform(which_iodims.begin(), part,         dims.begin(), [](auto elem) {return elem.second;});

		std::array<fftw_iodim64, D> howmany_dims{};
		auto const howmany_dims_end = std::transform(part, which_iodims.end() -1, howmany_dims.begin(), [](auto elem) {return elem.second;});

		////////////////////////////////////////////////////////////////////////

		std::array<int, D> istrides{};
		std::array<int, D> ostrides{};
		std::array<int, D> ion{};

		auto const istrides_end = std::transform(dims.begin(), dims_end, istrides.begin(), [](auto elem) {return elem.is;});
		auto const ostrides_end = std::transform(dims.begin(), dims_end, ostrides.begin(), [](auto elem) {return elem.os;});
		auto const ion_end      = std::transform(dims.begin(), dims_end, ion.begin(),      [](auto elem) {return elem.n;});

		int istride = *(istrides_end -1);
		auto inembed = istrides; inembed.fill(0);
		int ostride = *(ostrides_end -1);
		auto onembed = ostrides; onembed.fill(0);

		for(std::size_t i = 1; i != ion_end - ion.begin(); ++i) {
			assert(ostrides[i-1] >= ostrides[i]);
			assert(ostrides[i-1]%ostrides[i]==0);
			onembed[i]=ostrides[i-1]/ostrides[i];
			assert(istrides[i-1]%istrides[i]==0);
			inembed[i]=istrides[i-1]/istrides[i];
		}

		assert(ion_end - ion.begin() == D);

		::cufftHandle h;

		::cufftPlanMany(
			/*cufftHandle *plan*/ &h,
			/*int rank*/          dims_end - dims.begin(),
			/*int *n*/            ion.data(),
			/*int *inembed*/      inembed.data(),
			/*int istride*/       istride,
			/*int idist*/         1, //stride(first),
			/*int *onembed*/      onembed.data(),
			/*int ostride*/       ostride,
			/*int odist*/         1, //stride(d_first),
			/*cufftType type*/    CUFFT_Z2Z,
			/*int batch*/         1 //BATCH
		);

		assert(h == cufftSuccess );

		return h;

//      assert(ret &&"fftw lib returned a null plan, if you are using MKL check the limitations of their fftw interface"); 
		//https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/appendix-d-fftw-interface-to-intel-math-kernel-library/fftw3-interface-to-intel-math-kernel-library/using-fftw3-wrappers.html
//      return ret;
	}
#endif
#if 0
	template<class ILayout, class OLayout, //std::enable_if_t<(I::dimensionality < 4), int> =0,
		dimensionality_type D = ILayout::dimensionality//,
		// typename = decltype(raw_pointer_cast(base(std::declval<I const&>())), reinterpret_cast<complex_type*      >(raw_pointer_cast(base(std::declval<O&>()))))
	>
	plan(ILayout const& i, OLayout const& o, sign s) {
		assert( ILayout::dimensionality < 4 );
		assert( CUFFT_FORWARD == s or CUFFT_INVERSE == s or s == 0 );
		assert( sizes(i) == sizes(o) );

		auto ion      = std::apply([](auto... t){return std::array{static_cast< size_type>(t)...};}, sizes  (i));
		auto istrides = std::apply([](auto... t){return std::array{static_cast<ssize_type>(t)...};}, strides(i));
		auto ostrides = std::apply([](auto... t){return std::array{static_cast<ssize_type>(t)...};}, strides(o));

		std::array<std::tuple<int, int, int>, ILayout::dimensionality> ssn{};
		for(std::size_t i = 0; i != ssn.size(); ++i) {ssn[i] = std::make_tuple(istrides[i], ostrides[i], ion[i]);}
		std::sort(ssn.begin(), ssn.end(), std::greater<>{});

		for(std::size_t i = 0; i != ssn.size(); ++i) {
			istrides[i] = std::get<0>(ssn[i]);
			ostrides[i] = std::get<1>(ssn[i]);
			ion[i]      = std::get<2>(ssn[i]);
		}// = std::tuple<int, int, int>(istrides[i], ostrides[i], ion[i]);

		int istride = istrides.back();
		auto inembed = istrides; inembed.fill(0);
		int ostride = ostrides.back();
		auto onembed = ostrides; onembed.fill(0);
		for(std::size_t i = 1; i != onembed.size(); ++i) {
			assert(ostrides[i-1] >= ostrides[i]); // otherwise ordering is incompatible
			assert(ostrides[i-1]%ostrides[i]==0);
			onembed[i]=ostrides[i-1]/ostrides[i]; //    assert( onembed[i] <= ion[i] );
			assert(istrides[i-1]%istrides[i]==0);
			inembed[i]=istrides[i-1]/istrides[i]; //    assert( inembed[i] <= ion[i] );
		}

		// direction_ = s;
		// idata_ =                           reinterpret_cast<complex_type const*>(raw_pointer_cast(base(i))) ;
		// odata_ = const_cast<complex_type*>(reinterpret_cast<complex_type*      >(raw_pointer_cast(base(o))));

		// auto it = cache<D>().find(std::make_tuple(ion, inembed, istride, 1, onembed, ostride, 1, 1));
		// if(it != cache<D>().end()) {
		//  h_ = it->second;
		// }else{
			switch(::cufftPlanMany(
				/*cufftHandle *plan*/ &h_,
				/*int rank*/          ion.size(),
				/*int *n*/            ion.data(), //    /*NX*/      last - first,
				/*int *inembed*/      inembed.data(),
				/*int istride*/       istride,
				/*int idist*/         1, //stride(first),
				/*int *onembed*/      onembed.data(),
				/*int ostride*/       ostride,
				/*int odist*/         1, //stride(d_first),
				/*cufftType type*/    CUFFT_Z2Z,
				/*int batch*/         1 //BATCH
			)) {
				case CUFFT_SUCCESS        : break;// "cuFFT successfully executed the FFT plan."
				case CUFFT_ALLOC_FAILED   : throw std::runtime_error{"CUFFT failed to allocate GPU memory."};
				case CUFFT_INVALID_VALUE  : throw std::runtime_error{"At least one of the parameters idata, odata, and direction is not valid."};
				case CUFFT_INTERNAL_ERROR : throw std::runtime_error{"Used for all internal driver errors."};
				case CUFFT_SETUP_FAILED   : throw std::runtime_error{"The cuFFT library failed to initialize."};
				case CUFFT_INVALID_SIZE   : throw std::runtime_error{"The user specifies an unsupported FFT size."};
				default                   : throw std::runtime_error{"cufftPlanMany unknown error"};
			}
			// cache<D>().insert(std::make_pair(std::make_tuple(ion, inembed, istride, 1, onembed, ostride, 1, 1), h_));
		// }
		if(not h_) {throw std::runtime_error{"cufftPlanMany failed"};}
	}
#endif
#if 0
	template<class ILayout, class OLayout, dimensionality_type D = ILayout::dimensionality>
	static plan many(ILayout const& i, OLayout const& o)
	{
		// assert( CUFFT_FORWARD == sign or CUFFT_INVERSE == sign or sign == 0 );
		assert( sizes(i) == sizes(o) );

		auto ion      = std::apply([](auto... t){return std::array<size_type, D>{static_cast< size_type>(t)...};}, sizes  (i));

		assert(strides(*first) == strides(*last));
		auto istrides = std::apply([](auto... t){return std::array{static_cast<ssize_type>(t)...};}, strides(i));
		auto ostrides = std::apply([](auto... t){return std::array{static_cast<ssize_type>(t)...};}, strides(o));

		std::array<std::tuple<int, int, int>, D> ssn{};
		for(std::size_t i = 0; i != ssn.size(); ++i) ssn[i] = std::make_tuple(istrides[i], ostrides[i], ion[i]);
		std::sort(ssn.begin(), ssn.end(), std::greater<>{});

		for(std::size_t i = 0; i != ssn.size(); ++i){
			istrides[i] = std::get<0>(ssn[i]);
			ostrides[i] = std::get<1>(ssn[i]);
			ion[i]      = std::get<2>(ssn[i]);
		}

		int istride = istrides.back();
		auto inembed = istrides; inembed.fill(0);
		int ostride = ostrides.back();
		auto onembed = ostrides; onembed.fill(0);
		for(std::size_t i = 1; i != onembed.size(); ++i) {
			assert(ostrides[i-1] >= ostrides[i]); // otherwise ordering is incompatible
			assert(ostrides[i-1]%ostrides[i]==0);
			onembed[i]=ostrides[i-1]/ostrides[i]; //    assert( onembed[i] <= ion[i] );
			assert(istrides[i-1]%istrides[i]==0);
			inembed[i]=istrides[i-1]/istrides[i]; //    assert( inembed[i] <= ion[i] );
		}

		plan ret;
		// ret.direction_ = sign;
		// ret.idata_ =                           reinterpret_cast<complex_type const*>(raw_pointer_cast(  first.base())) ;
		// ret.odata_ = const_cast<complex_type*>(reinterpret_cast<complex_type*      >(raw_pointer_cast(d_first.base())));

		// auto it = cache<D>().find(std::make_tuple(ion, inembed, istride, stride(first), onembed, ostride, stride(d_first), last - first));
		// if(it != cache<D>().end()) {
		//  ret.h_ = it->second;
		// } else {
			switch(::cufftPlanMany(
				/*cufftHandle *plan*/ &ret.h_,
				/*int rank*/          ion.size(),
				/*int *n*/            ion.data(), //    /*NX*/      last - first,
				/*int *inembed*/      inembed.data(),
				/*int istride*/       istride,
				/*int idist*/         stride(i),
				/*int *onembed*/      onembed.data(),
				/*int ostride*/       ostride,
				/*int odist*/         stride(o),
				/*cufftType type*/    CUFFT_Z2Z,
				/*int batch*/         i.size() //BATCH
			)) {
				case CUFFT_SUCCESS        : break;// "cuFFT successfully executed the FFT plan."
			//  case CUFFT_INVALID_PLAN   : throw std::runtime_error{"The plan parameter is not a valid handle."};
				case CUFFT_ALLOC_FAILED   : throw std::runtime_error{"CUFFT failed to allocate GPU memory."};
			//  case CUFFT_INVALID_TYPE   : throw std::runtime_error{"The user requests an unsupported type."};
				case CUFFT_INVALID_VALUE  : throw std::runtime_error{"At least one of the parameters idata, odata, and direction is not valid."};
				case CUFFT_INTERNAL_ERROR : throw std::runtime_error{"Used for all internal driver errors."};
			//  case CUFFT_EXEC_FAILED    : throw std::runtime_error{"CUFFT failed to execute an FFT on the GPU."};
				case CUFFT_SETUP_FAILED   : throw std::runtime_error{"The cuFFT library failed to initialize."};
				case CUFFT_INVALID_SIZE   : throw std::runtime_error{"The user specifies an unsupported FFT size."};
			//  case CUFFT_UNALIGNED_DATA : throw std::runtime_error{"Unaligned data."};
			//  case CUFFT_INCOMPLETE_PARAMETER_LIST: throw std::runtime_error{"Incomplete parameter list."};
			//  case CUFFT_INVALID_DEVICE : throw std::runtime_error{"Invalid device."};
			//  case CUFFT_PARSE_ERROR    : throw std::runtime_error{"Parse error."};
			//  case CUFFT_NO_WORKSPACE   : throw std::runtime_error{"No workspace."};
			//  case CUFFT_NOT_IMPLEMENTED: throw std::runtime_error{"Not implemented."};
			//  case CUFFT_LICENSE_ERROR  : throw std::runtime_error{"License error."};
			//  case CUFFT_NOT_SUPPORTED  : throw std::runtime_error{"CUFFT_NOT_SUPPORTED"};
				default                   : throw std::logic_error{"cufftPlanMany unknown error"};
			}
			// cache<D>().insert(std::make_pair(std::make_tuple(ion, inembed, istride, stride(first), onembed, ostride, stride(d_first), last - first), ret.h_));
		// }
		if(not ret.h_) {throw std::runtime_error{"cufftPlanMany failed"};}
		return ret;
	}
#endif
};

template<typename In, class Out, dimensionality_type D = In::rank::value>
auto dft(std::array<bool, +D> which, In const& i, Out&& o, int s)
->decltype(cufft::plan<D>{which, i.layout(), o.layout()}(i.base(), o.base(), s), std::forward<Out>(o)) {
	return cufft::plan<D>{which, i.layout(), o.layout()}(i.base(), o.base(), s), std::forward<Out>(o); }

template<typename In, typename R = multi::array<typename In::element_type, In::dimensionality, decltype(get_allocator(std::declval<In>()))>>
NODISCARD("when first argument is const")
R dft(In const& i, int s) {
	static_assert(std::is_trivially_default_constructible<typename In::element_type>{}, "!");
	R ret(extensions(i), get_allocator(i));
	cufft::dft(i, ret, s);
	if(cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error{"Cuda error: Failed to synchronize"};
	return ret;
}

// template<typename It1, typename It2>
// auto many_dft(It1 first, It1 last, It2 d_first, sign s)
// // ->decltype(plan::many(first, last, d_first, s)(), d_first + (last - first)) {
// {   return plan<>::many(multi::ref(first, last).layout(), multi::ref(d_first, d_first + (last - first)))(first->base(), d_first->base(), s), d_first + (last - first); }

// template<typename In, class Out,  std::size_t D = In::dimensionality, std::enable_if_t<(D==1), int> = 0>
// Out&& dft(std::array<bool, +D> which, In const& i, Out&& o, int s) {
//  if(which[0]) return cufft::dft(i, std::forward<Out>(o), s);
//  else return std::forward<Out>(std::forward<Out>(o) = i);
// }

template <class Array, std::size_t... Ns>
constexpr auto array_tail_impl(Array const& t, std::index_sequence<Ns...>) {
	return std::array<typename Array::value_type, std::tuple_size<Array>{} - 1>{std::get<Ns + 1>(t)...};
}

template<class Array>
constexpr auto array_tail(Array const& t)
->decltype(array_tail_impl(t, std::make_index_sequence<std::tuple_size<Array>{} - 1>())) {
	return array_tail_impl(t, std::make_index_sequence<std::tuple_size<Array>{} - 1>()); }

#if 0
template<typename In, class Out, std::size_t D = In::dimensionality, std::enable_if_t<(D>1), int> = 0>
auto dft(std::array<bool, +D> which, In const& i, Out&& o, int s)
->decltype(many_dft(i.begin(), i.end(), o.begin(), s),std::forward<Out>(o))
{
#if 0
	assert(i.base() != o.base());
	assert(extension(i) == extension(o));
	auto ff = std::find(begin(which)+1, end(which), false);
	if(which[0] == true) {
		if(ff==end(which)) {cufft::dft(i, std::forward<Out>(o), s);}
		else {
			auto const n = ff - which.begin();
			std::rotate(begin(which), ff, end(which));
			// TODO(correaa) : make this more elegant
			switch(n) {
				case 0: dft(which, i()                            , o()                            , s); break;
				case 1: dft(which, i.rotated()                    , o.rotated()                    , s); break;
				case 2: dft(which, i.rotated().rotated()          , o.rotated().rotated()          , s); break;
				case 3: dft(which, i.rotated().rotated().rotated(), o.rotated().rotated().rotated(), s); break;
				default: assert(0);
			}
		}
	}
	else if(which[0]==false) {
		if(D==1 or std::none_of(begin(which)+1, end(which), [](auto e){return e;})){
			if(base(o) != base(i)) o() = i;
			else assert(0);
			#if 0
			else if(o.layout() != i.layout()) {
				auto tmp = +i;
				o() = tmp;
				// std::forward<Out>(o) = tmp;
			}
			#endif
		}
		#if 1
		else if(ff==end(which)) many_dft(i.begin(), i.end(), o.begin(), s);
		else{
			std::array<bool, D-1> tail = array_tail(which);
			if(which[1] == false and i.is_flattable() and o.is_flattable()) cufft::dft(tail, i.flatted(), o.flatted(), s);
			else{
				auto d_min = 0; auto n_min = size(i);
				for(auto d = 0; d != D - 1; ++d) {
					switch(d) {
						case 0: if( (size(i                              ) < n_min) and (tail[d] == false)) {n_min = size(i                              ); d_min = d;} break;
						case 1: if( (size(i.rotated()                    ) < n_min) and (tail[d] == false)) {n_min = size(i.rotated()                    ); d_min = d;} break;
						case 2: if( (size(i.rotated().rotated()          ) < n_min) and (tail[d] == false)) {n_min = size(i.rotated().rotated()          ); d_min = d;} break;
						case 3: if( (size(i.rotated().rotated().rotated()) < n_min) and (tail[d] == false)) {n_min = size(i.rotated().rotated().rotated()); d_min = d;} break;
						default: assert(0);
					}
				//  if((size(i<<d) < n_min) and (tail[d]==false)) {n_min = size(i<<d); d_min = d;}
				}
				if( d_min!=0 ) {
					std::rotate(which.begin(), which.begin()+d_min, which.end());
					switch(d_min) {
						case 0: dft(which, i(), o(), s); break;
						case 1: dft(which, i.rotated()                    , o.rotated()                    , s); break;
						case 2: dft(which, i.rotated().rotated()          , o.rotated().rotated()          , s); break;
						case 3: dft(which, i.rotated().rotated().rotated(), o.rotated().rotated().rotated(), s); break;
						default: assert(0);
					}
				//  dft(which, i<<d_min, o<<d_min, s);
				} else {
					if(i.base() == o.base() and i.layout() != o.layout()){
						auto const tmp = +i;
						for(auto idx : extension(i)) cufft::dft(tail, tmp[idx], o[idx], s);
					}else for(auto idx : extension(i)){
						MULTI_MARK_SCOPE("cufft inner loop");
						cufft::dft(tail, i[idx], o[idx], s);
					}
				}
			}
		}
		#endif
	}
#endif
	return std::forward<Out>(o);
}
#endif

template<typename In, class Out, std::size_t D = In::dimensionality, std::enable_if_t<(D>1), int> = 0>
auto dft_forward(std::array<bool, +D> which, In const& i, Out&& o)
->decltype(dft(which, i, std::forward<Out>(o), cufft::forward)) {
	return dft(which, i, std::forward<Out>(o), cufft::forward); }

template<typename In, class Out, std::size_t D = In::dimensionality, std::enable_if_t<(D>1), int> = 0>
auto dft_backward(std::array<bool, +D> which, In const& i, Out&& o)
->decltype(dft(which, i, std::forward<Out>(o), cufft::backward)) {
	return dft(which, i, std::forward<Out>(o), cufft::backward); }

template<typename In,  std::size_t D = In::dimensionality>
NODISCARD("when passing a const argument")
auto dft(std::array<bool, D> which, In const& i, int sign)->std::decay_t<decltype(
dft(which, i, typename In::decay_type(extensions(i), get_allocator(i)), sign))>{return 
dft(which, i, typename In::decay_type(extensions(i), get_allocator(i)), sign);}

template<typename In,  std::size_t D = In::dimensionality>
auto dft(std::array<bool, D> which, In&& i, int sign)
->decltype(dft(which, i, i, sign), std::forward<In>(i)){
	return dft(which, i, i, sign), std::forward<In>(i);}

template<typename Array, typename A> NODISCARD("when passing a const argument")
auto dft_forward(Array arr, A const& a) 
->decltype(cufft::dft(arr, a, cufft::forward)){
	return cufft::dft(arr, a, cufft::forward);}

// template<typename Array, dimensionality_type D> NODISCARD("when passing a const argument")
// auto dft_forward(Array arr, multi::cuda::array<std::complex<double>, D>&& a) 
// ->decltype(cufft::dft(arr, a, cufft::forward), multi::cuda::array<std::complex<double>, D>{}){//assert(0);
//  return cufft::dft(arr, a, cufft::forward), std::move(a);}

template<typename A> NODISCARD("when passing a const argument")
auto dft_forward(A const& a)
->decltype(cufft::dft(a, cufft::forward)){
	return cufft::dft(a, cufft::forward);}

template<typename... A> auto            dft_backward(A&&... a)
->decltype(cufft::dft(std::forward<A>(a)..., cufft::backward)){
	return cufft::dft(std::forward<A>(a)..., cufft::backward);}

template<typename Array, typename A> NODISCARD("when passing a const argument")
auto dft_backward(Array arr, A const& a) 
->decltype(cufft::dft(arr, a, cufft::backward)){
	return cufft::dft(arr, a, cufft::backward);}

template<typename A> NODISCARD("when passing a const argument")
auto dft_backward(A const& a)
->decltype(cufft::dft(a, cufft::backward)){
	return cufft::dft(a, cufft::backward);}

}

}}
#endif
