// Copyright 2020-2024 Alfredo A. Correa

#ifndef BOOST_MULTI_ADAPTORS_HIPFFT_HPP
#define BOOST_MULTI_ADAPTORS_HIPFFT_HPP

#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>

using cudaError_t  = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t  = hipEvent_t;

constexpr static auto const& cudaDeviceReset  = hipDeviceReset;
constexpr static auto const& cudaDeviceSynchronize  = hipDeviceSynchronize;
constexpr static auto const& cudaSuccess = hipSuccess;

constexpr static auto const& cudaGetLastError = hipGetLastError;
constexpr static auto const& cudaFree         = hipFree;
constexpr static auto const  cudaMallocAsync  = static_cast<hipError_t (*)(void**, std::size_t, hipStream_t)>(hipMallocAsync);  // ROCm >= 5.2; cast selects the C overload
constexpr static auto const& cudaFreeAsync    = hipFreeAsync;

constexpr static auto const& cudaStreamCreate    = hipStreamCreate;
constexpr static auto const& cudaStreamDestroy   = hipStreamDestroy;
constexpr static auto const& cudaStreamWaitEvent = hipStreamWaitEvent;

constexpr static auto const& cudaEventCreateWithFlags = hipEventCreateWithFlags;
constexpr static auto const& cudaEventDestroy         = hipEventDestroy;
constexpr static auto const& cudaEventRecord          = hipEventRecord;
constexpr static auto        cudaEventDisableTiming   = hipEventDisableTiming;  // flag constant, by value

#define cu2hip_fft(TypeleafnamE) using cufft ## TypeleafnamE = hipfft ## TypeleafnamE
    cu2hip_fft(Handle);
    cu2hip_fft(DoubleComplex);
    cu2hip_fft(Result);
#undef cu2hip_fft

#define cu2hip_fft(FunctionleafnamE) constexpr static auto const& cufft ## FunctionleafnamE  = hipfft ## FunctionleafnamE
    cu2hip_fft(Create);
    cu2hip_fft(Destroy);
    cu2hip_fft(GetSize);
    cu2hip_fft(ExecZ2Z);
    cu2hip_fft(SetAutoAllocation);
    cu2hip_fft(SetWorkArea);
    cu2hip_fft(PlanMany);
    cu2hip_fft(MakePlanMany);
    cu2hip_fft(SetStream);
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
CU2HIPFFT_(INVALID_PLAN);
CU2HIPFFT_(NO_WORKSPACE);
CU2HIPFFT_(NOT_IMPLEMENTED);
CU2HIPFFT_(NOT_SUPPORTED);
CU2HIPFFT_(UNALIGNED_DATA);
CU2HIPFFT_(PARSE_ERROR);
CU2HIPFFT_(SETUP_FAILED);
CU2HIPFFT_(SUCCESS);
CU2HIPFFT_(Z2Z);

#undef CU2HIPFFT_

#include "cufft.hpp"

// namespace boost::multi{
//     namespace cufft = hipfft;
// }

#endif  // BOOST_MULTI_ADAPTORS_HIPFFT_HPP
