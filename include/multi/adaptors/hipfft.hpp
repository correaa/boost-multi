// Copyright 2020-2023 Alfredo A. Correa

#ifndef MULTI_ADAPTORS_HIPFFT_HPP
#define MULTI_ADAPTORS_HIPFFT_HPP

#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>


// #define CONCAT_(A, B) A ## B
// #define CONCAT(A, B) CONCAT_(A, B)

// #define kufft(A) CONCAT(cufft, Result)

// hipdefine(fftResult) -> #define cufftResult hipfftResult

using cudaError_t = hipError_t;
using cufftResult = hipfftResult;

#define cufftResult        hipfftResult
#define CUFFT_INVALID_PLAN HIPFFT_INVALID_PLAN
#define CUFFT_Z2Z          HIPFFT_Z2Z

template<class... As> inline auto cufftSetWorkArea(As&&... as) noexcept(noexcept(hipfftSetWorkArea(std::forward<As>(as)...))) -> decltype(hipfftSetWorkArea(std::forward<As>(as)...)) {return hipfftSetWorkArea(std::forward<As>(as)...);}
template<class... As> inline auto cufftPlanMany   (As&&... as) noexcept(noexcept(hipfftPlanMany   (std::forward<As>(as)...))) -> decltype(hipfftPlanMany   (std::forward<As>(as)...)) {return hipfftPlanMany   (std::forward<As>(as)...);}

using cufftHandle        = hipfftHandle;
using cufftDoubleComplex = hipfftDoubleComplex;

constexpr static auto const& cudaDeviceReset  = hipDeviceReset;
constexpr static auto const& cudaDeviceSynchronize  = hipDeviceSynchronize;
constexpr static auto const& cudaSuccess = hipSuccess;

#define cu2hip_fft(NamE) constexpr static auto const& cufft ## NamE  = hipfft ## NamE

cu2hip_fft(Create);
cu2hip_fft(Destroy);
// cu2hip_fft(DeviceReset);
cu2hip_fft(GetSize);
cu2hip_fft(ExecZ2Z);
cu2hip_fft(SetAutoAllocation);

#undef cu2hip_fft

#define CU2HIPFFT_(NamE) constexpr static auto const& CUFFT_ ## NamE  = HIPFFT_ ## NamE

CU2HIPFFT_(ALLOC_FAILED);
CU2HIPFFT_(BACKWARD);

constexpr static auto const& CUFFT_INVERSE = HIPFFT_BACKWARD;

CU2HIPFFT_(EXEC_FAILED);
CU2HIPFFT_(FORWARD);
CU2HIPFFT_(INCOMPLETE_PARAMETER_LIST);
CU2HIPFFT_(INTERNAL_ERROR);
CU2HIPFFT_(INVALID_DEVICE);
CU2HIPFFT_(INVALID_SIZE);
CU2HIPFFT_(INVALID_TYPE);
CU2HIPFFT_(INVALID_VALUE);
CU2HIPFFT_(NO_WORKSPACE);
CU2HIPFFT_(NOT_IMPLEMENTED);
CU2HIPFFT_(NOT_SUPPORTED);
CU2HIPFFT_(UNALIGNED_DATA);
CU2HIPFFT_(PARSE_ERROR);
CU2HIPFFT_(SETUP_FAILED);
CU2HIPFFT_(SUCCESS);

#undef CU2HIPFFT_

#include "cufft.hpp"

#endif
