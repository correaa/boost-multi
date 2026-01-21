#include <boost/multi/adaptors/thrust.hpp>

#include <assert.h>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <unordered_map>
#include <vector>

// Handle cuTENSOR errors
#define HANDLE_ERROR(x)                                         \
	{                                                           \
		const auto err = x;                                     \
		if(err != CUTENSOR_STATUS_SUCCESS) {                    \
			printf("Error: %s\n", cutensorGetErrorString(err)); \
			exit(-1);                                           \
		}                                                       \
	};

#define HANDLE_CUDA_ERROR(x)                                \
	{                                                       \
		const auto err = x;                                 \
		if(err != cudaSuccess) {                            \
			printf("Error: %s\n", cudaGetErrorString(err)); \
			exit(-1);                                       \
		}                                                   \
	};

namespace boost::multi::cutensor {
template<class T> struct datatype;
template<> struct datatype<float> : std::integral_constant<cutensorDataType_t, CUTENSOR_R_32F> {};

class context {
	cutensorHandle_t handle_;

 public:
	context() {
		HANDLE_ERROR(cutensorCreate(&handle_));
	}
	context(context const&) = delete;
	~context() { HANDLE_ERROR(cutensorDestroy(handle_)); }

	auto operator&() { return handle_; }
};

class descriptor {
	cutensorTensorDescriptor_t desc_;

 public:
	template<class StridedLayout>
	descriptor(StridedLayout const& lyt, cutensorDataType_t type, uint32_t kAlignment, context& ctxt) {
		auto extents = std::apply([](auto... es) { return std::array{static_cast<int64_t>(es)...}; }, lyt.sizes());
		cutensorCreateTensorDescriptor(&ctxt, &desc_, lyt.dimensionality, extents.data(), NULL, /*stride*/
									   type, kAlignment);
	}
	descriptor(descriptor const&) = delete;
	~descriptor() { HANDLE_ERROR(cutensorDestroyTensorDescriptor(desc_)); }

	auto operator&() { return desc_; }
};

struct operation {
	cutensorOperationDescriptor_t desc_;

 public:
	~operation() {
		HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc_));
	}
	auto operator&() { return desc_; }

	auto get_scalar_type(context& ctxt) const {
		cutensorDataType_t scalarType;
		HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(&ctxt, desc_, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, (void*)&scalarType, sizeof(scalarType)));
		return scalarType;
	}

	static auto contraction(descriptor& dA, std::span<int> modeA, descriptor& dB, std::span<int> modeB, descriptor& dC, std::span<int> modeC, cutensorComputeDescriptor_t descCompute, context& ctxt) {
		operation ret;
		HANDLE_ERROR(cutensorCreateContraction(&ctxt, &ret.desc_, &dA, modeA.data(), /* unary operator A*/ CUTENSOR_OP_IDENTITY, &dB, modeB.data(), /* unary operator B*/ CUTENSOR_OP_IDENTITY, &dC, modeC.data(), /* unary operator C*/ CUTENSOR_OP_IDENTITY, &dC, modeC.data(), descCompute));
		return ret;
	}
};

struct plan_preference {
	cutensorPlanPreference_t planPref_;
	plan_preference(multi::cutensor::context& ctxt, cutensorAlgo_t const algo = CUTENSOR_ALGO_DEFAULT) {
		HANDLE_ERROR(cutensorCreatePlanPreference(
			&ctxt,
			&planPref_,
			algo,
			CUTENSOR_JIT_MODE_NONE
		));
	}
	plan_preference(plan_preference const&) = delete;
	~plan_preference() {
		HANDLE_ERROR(cutensorDestroyPlanPreference(planPref_));
	}
	auto operator&() { return planPref_; }
};

class plan {
	cutensorPlan_t plan_;

	static auto estimate_workspace_size(multi::cutensor::context& ctxt, multi::cutensor::operation& op, plan_preference& pp, cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT) {
		uint64_t ret;
		HANDLE_ERROR(cutensorEstimateWorkspaceSize(&ctxt, &op, &pp, workspacePref, &ret));
		return ret;
	}

 public:
	plan(multi::cutensor::context& ctxt, multi::cutensor::operation& op, plan_preference& pp, uint64_t workspaceSizeEstimate) {
		HANDLE_ERROR(cutensorCreatePlan(&ctxt, &plan_, &op, &pp, workspaceSizeEstimate));
	}

	plan(multi::cutensor::context& ctxt, multi::cutensor::operation& op, plan_preference& pp, cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT)
	: plan(ctxt, op, pp, estimate_workspace_size(ctxt, op, pp, workspacePref)) {}

	plan(plan const&) = delete;
	~plan() {
		HANDLE_ERROR(cutensorDestroyPlan(plan_));
	}
	auto operator&() {
		return plan_;
	}

	auto get_required_workspace(multi::cutensor::context& ctxt) {
		uint64_t actualWorkspaceSize;
		HANDLE_ERROR(cutensorPlanGetAttribute(&ctxt, plan_, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize, sizeof(actualWorkspaceSize)));
		return actualWorkspaceSize;
	}

    template<class ComputeType>
	void contract(multi::cutensor::context& ctxt, ComputeType const& alpha, void* A_d, void* B_d, ComputeType const& beta, void* C_d, void* work, uint64_t actualWorkspaceSize, cudaStream_t stream) {
		HANDLE_ERROR(cutensorContract(&ctxt, plan_, (void*)&alpha, A_d, B_d, (void*)&beta, C_d, C_d, (void*)::thrust::raw_pointer_cast(work), actualWorkspaceSize, stream));
	}
};

}  // namespace boost::multi::cutensor

namespace multi = boost::multi;

int main() {

	// Host element type definition
	using floatTypeA       = float;
	using floatTypeB       = float;
	using floatTypeC       = float;
	using floatTypeCompute = float;

	std::cout << "Include headers and define data types\n";

	/* ***************************** */

	// Create vector of modes
	std::array<int, 4> modeC{'m', 'u', 'n', 'v'};
	std::array<int, 4> modeA{'m', 'h', 'k', 'n'};
	std::array<int, 4> modeB{'u', 'k', 'v', 'h'};

	// Extents
	std::unordered_map<int, int64_t> extent = {
		{'m', 96},
		{'n', 96},
		{'u', 96},
		{'v', 64},
		{'h', 64},
		{'k', 64},
	};

	multi::thrust::cuda::array<floatTypeA, 4> A_dev({extent['m'], extent['h'], extent['k'], extent['n']});
	multi::thrust::cuda::array<floatTypeB, 4> B_dev({extent['u'], extent['k'], extent['v'], extent['h']});
	multi::thrust::cuda::array<floatTypeC, 4> C_dev({extent['m'], extent['u'], extent['n'], extent['v']});

	printf("Define modes and extents\n");

	size_t elementsA = A_dev.num_elements();
	size_t elementsB = B_dev.num_elements();
	size_t elementsC = B_dev.num_elements();

	// Allocate on device
	void* A_d = static_cast<void*>(raw_pointer_cast(A_dev.data_elements()));
	void* B_d = static_cast<void*>(raw_pointer_cast(B_dev.data_elements()));
	void* C_d = static_cast<void*>(raw_pointer_cast(A_dev.data_elements()));

	// Allocate on host
	multi::thrust::host::array<floatTypeA, 4> A_host({extent['m'], extent['h'], extent['k'], extent['n']});
	multi::thrust::host::array<floatTypeA, 4> B_host({extent['u'], extent['k'], extent['v'], extent['h']});
	multi::thrust::host::array<floatTypeA, 4> C_host({extent['m'], extent['u'], extent['n'], extent['v']});

	std::generate(A_host.elements().begin(), A_host.elements().end(), [] { return (((float)rand()) / static_cast<float>(RAND_MAX) - 0.5) * 100; });
	std::generate(B_host.elements().begin(), B_host.elements().end(), [] { return (((float)rand()) / static_cast<float>(RAND_MAX) - 0.5) * 100; });
	std::generate(C_host.elements().begin(), C_host.elements().end(), [] { return (((float)rand()) / static_cast<float>(RAND_MAX) - 0.5) * 100; });

	// Copy to device
	A_dev = A_host;
	B_dev = B_host;
	C_dev = C_host;

	printf("Allocate, initialize and transfer tensors\n");

	multi::cutensor::context ctxt;

	uint32_t const kAlignment = 128;  // Alignment of the global-memory device pointers (bytes)

	assert(uintptr_t(A_dev.data_elements()) % kAlignment == 0);
	assert(uintptr_t(B_dev.data_elements()) % kAlignment == 0);
	assert(uintptr_t(C_dev.data_elements()) % kAlignment == 0);

	multi::cutensor::descriptor descA(A_dev.layout(), multi::cutensor::datatype<decltype(A_dev)::element>::value, kAlignment, ctxt);
	multi::cutensor::descriptor descB(B_dev.layout(), multi::cutensor::datatype<decltype(B_dev)::element>::value, kAlignment, ctxt);
	multi::cutensor::descriptor descC(C_dev.layout(), multi::cutensor::datatype<decltype(C_dev)::element>::value, kAlignment, ctxt);

	printf("Initialize cuTENSOR and tensor descriptors\n");

	multi::cutensor::operation opc = multi::cutensor::operation::contraction(descA, modeA, descB, modeB, descC, modeC, CUTENSOR_COMPUTE_DESC_32F, ctxt);

	using floatTypeCompute = float;
	assert(opc.get_scalar_type(ctxt) == multi::cutensor::datatype<floatTypeCompute>::value);

	auto alpha = static_cast<floatTypeCompute>(1.1f);
	auto beta  = static_cast<floatTypeCompute>(0.0f);

	multi::cutensor::plan_preference pp(ctxt);

	multi::cutensor::plan p{ctxt, opc, pp};

	uint64_t actualWorkspaceSize = p.get_required_workspace(ctxt);

	// At this point the user knows exactly how much memory is need by the operation and
	// only the smaller actual workspace needs to be allocated
	assert(actualWorkspaceSize <= workspaceSizeEstimate);

	thrust::cuda::allocator<std::byte> alloc;

	auto work = actualWorkspaceSize ? alloc.allocate(actualWorkspaceSize) : nullptr;
	assert(work == nullptr || uintptr_t(work) % 128 == 0);  // workspace must be aligned to 128 byte-boundary

	cudaStream_t stream;
	HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

	p.contract(ctxt, alpha, A_d, B_d, beta, C_d, (void*)raw_pointer_cast(work), actualWorkspaceSize, stream);

	HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

	if(work)
		alloc.deallocate(work, actualWorkspaceSize);
}
