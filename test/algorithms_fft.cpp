// Copyright 2024-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/algorithms/fft.hpp>  // for fft_inplace, fft_plan
#include <boost/multi/array.hpp>           // for array

#include <boost/core/lightweight_test.hpp>

#include <algorithm>  // for std::max
#include <cmath>      // for std::abs
#include <complex>    // for std::complex
#include <cstddef>    // for std::size_t
#include <memory>     // for std::allocator

namespace multi = boost::multi;

// Minimal counting allocator (Allocator concept, value_type == T, no
// rebinding needed since fft_plan::execute() never asks for anything but
// T): proves execute() actually threads the caller's allocator through
// (fft.NOTES.md §9.2/§10.4(b)) rather than silently using its own
// std::allocator<T>, and that the requested size is stable across repeated
// calls on one plan (no creeping growth, no unmatched allocate/deallocate).
template<class T>
struct counting_allocator {
	using value_type = T;

	std::size_t* alloc_count;
	std::size_t* dealloc_count;
	std::size_t* last_alloc_n;

	auto allocate(std::size_t n) -> T* {
		++(*alloc_count);
		*last_alloc_n = n;
		return std::allocator<T>{}.allocate(n);
	}
	void deallocate(T* p, std::size_t n) {
		++(*dealloc_count);
		std::allocator<T>{}.deallocate(p, n);
	}
	auto operator==(counting_allocator const& other) const -> bool { return alloc_count == other.alloc_count; }
	auto operator!=(counting_allocator const& other) const -> bool { return !(*this == other); }
};

// NOLINTBEGIN(altera-id-dependent-backward-branch,altera-unroll-loops,readability-identifier-length)
// test loops iterate runtime sizes; short names (m = max difference, n = size)
// are conventional here

namespace {
using complex = std::complex<double>;

constexpr auto tol = 1e-9;

// Reference direct DFT, independent from the implementation under test.
template<class In>
auto dft_reference(In const& in, int sign) {
	auto const               nn = static_cast<std::size_t>(in.size());
	multi::array<complex, 1> out(multi::extents_t<1>{static_cast<multi::ssize_t>(nn)}, complex{});
	auto const               pi = std::acos(-1.0);
	for(std::size_t k = 0; k != nn; ++k) {
		complex sum{};
		for(std::size_t n = 0; n != nn; ++n) {
			auto const theta = static_cast<double>(sign) * 2.0 * pi * static_cast<double>(n * k) / static_cast<double>(nn);
			sum += in[static_cast<multi::ssize_t>(n)] * complex{std::cos(theta), std::sin(theta)};
		}
		out[static_cast<multi::ssize_t>(k)] = sum;
	}
	return out;
}

template<class A, class B>
auto max_abs_diff(A const& aa, B const& bb) -> double {
	double m  = 0.0;
	auto   it = bb.begin();
	for(auto const& e : aa) {
		m = std::max(m, std::abs(e - *it++));
	}
	return m;
}
}  // namespace

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	// 1D power-of-two matches a direct DFT
	{
		multi::array<complex, 1> arr = {
			{ 1.0,  0.0},
			{ 2.0, -1.0},
			{ 0.0, -1.0},
			{-1.0,  2.0},
			{ 3.0,  1.0},
			{ 0.0,  0.0},
			{-2.0,  1.0},
			{ 1.0,  1.0},
		};
		auto const ref = dft_reference(arr, multi::fft_forward);
		multi::fft_inplace(arr, multi::fft_forward);
		BOOST_TEST( max_abs_diff(arr, ref) < tol );
	}

	// 1D prime size (7) still matches a direct DFT (exercises the direct generic kernel)
	{
		multi::array<complex, 1> arr = {
			{ 1.0,  0.0},
			{ 2.0, -1.0},
			{ 0.0, -1.0},
			{-1.0,  2.0},
			{ 3.0,  1.0},
			{ 0.0,  0.0},
			{-2.0,  1.0},
		};
		auto const ref = dft_reference(arr, multi::fft_forward);
		multi::fft_inplace(arr, multi::fft_forward);
		BOOST_TEST( max_abs_diff(arr, ref) < tol );
	}

	// 1D composite non-power-of-two size (12 = 2^2 * 3)
	{
		multi::array<complex, 1> arr(multi::extents_t<1>{12}, complex{});
		for(int i = 0; i != 12; ++i) {
			arr[i] = complex{static_cast<double>(i), static_cast<double>((i * 7) % 5)};
		}
		auto const ref = dft_reference(arr, multi::fft_forward);
		multi::fft_inplace(arr, multi::fft_forward);
		BOOST_TEST( max_abs_diff(arr, ref) < tol );
	}

	// forward followed by backward recovers the input scaled by N (FFTW convention)
	{
		multi::array<complex, 1> const original = {
			{ 1.0,  0.0},
			{ 2.0, -1.0},
			{ 0.0, -1.0},
			{-1.0,  2.0},
			{ 3.0,  1.0},
		};
		auto const nn  = static_cast<double>(original.size());
		auto       arr = original;
		multi::fft_inplace(arr, multi::fft_forward);
		multi::fft_inplace(arr, multi::fft_backward);
		double m = 0.0;
		for(int i = 0; i != static_cast<int>(original.size()); ++i) {
			m = std::max(m, std::abs(arr[i] - original[i] * nn));
		}
		BOOST_TEST( m < tol );
	}

	// DC component of a forward transform is the sum of all elements
	{
		multi::array<complex, 1> arr = {
			{1.0, 0.0},
			{2.0, 0.0},
			{3.0, 0.0},
			{4.0, 0.0},
		};
		multi::fft_inplace(arr, multi::fft_forward);
		BOOST_TEST( std::abs(arr[0] - complex{10.0, 0.0}) < tol );
	}

	// 2D transform equals composition of 1D transforms along each axis
	{
		multi::array<complex, 2> arr({4, 6}, complex{});
		for(int i = 0; i != 4; ++i) {
			for(int j = 0; j != 6; ++j) {
				arr[i][j] = complex{static_cast<double>(i - j), static_cast<double>(i * j % 3)};
			}
		}

		auto reference = arr;
		// transform each row (last axis), then each column (first axis)
		for(int i = 0; i != 4; ++i) {
			auto row = dft_reference(reference[i], multi::fft_forward);
			for(int j = 0; j != 6; ++j) {
				reference[i][j] = row[j];
			}
		}
		for(int j = 0; j != 6; ++j) {
			auto col = dft_reference(reference.rotated()[j], multi::fft_forward);
			for(int i = 0; i != 4; ++i) {
				reference[i][j] = col[i];
			}
		}

		multi::fft_inplace(arr, multi::fft_forward);

		double m = 0.0;
		for(int i = 0; i != 4; ++i) {
			for(int j = 0; j != 6; ++j) {
				m = std::max(m, std::abs(arr[i][j] - reference[i][j]));
			}
		}
		BOOST_TEST( m < tol );
	}

	// 2D forward + backward round-trip recovers input scaled by total number of elements
	{
		multi::array<complex, 2> const original = {
			{{1.0, 0.0},  {2.0, 1.0}, {3.0, -1.0}},
			{{0.0, 2.0}, {-1.0, 0.0},  {4.0, 1.0}},
		};
		auto const nn  = static_cast<double>(original.num_elements());
		auto       arr = original;
		multi::fft_inplace(arr, multi::fft_forward);
		multi::fft_inplace(arr, multi::fft_backward);
		double m = 0.0;
		for(int i = 0; i != 2; ++i) {
			for(int j = 0; j != 3; ++j) {
				m = std::max(m, std::abs(arr[i][j] - original[i][j] * nn));
			}
		}
		BOOST_TEST( m < tol );
	}

	// works on a non-contiguous (strided) sub-array view: transform a sub-block in place
	{
		multi::array<complex, 2> arr({4, 4}, complex{});
		for(int i = 0; i != 4; ++i) {
			for(int j = 0; j != 4; ++j) {
				arr[i][j] = complex{static_cast<double>(i + 1), static_cast<double>(j)};
			}
		}

		// take a strided 1D fiber: column 1 (stride = row length)
		auto&&                         col = arr.rotated()[1];
		multi::array<complex, 1> const col_copy(col);
		auto const                     ref = dft_reference(col_copy, multi::fft_forward);

		multi::fft_inplace(col, multi::fft_forward);  // transform the strided column in place

		double m = 0.0;
		for(int i = 0; i != 4; ++i) {
			m = std::max(m, std::abs(arr[i][1] - ref[i]));
		}
		BOOST_TEST( m < tol );
	}

	// many 1D sizes vs a direct DFT: exercises radix-2/3/4/5/8 and generic-prime stages
	{
		for(int const n : {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 24, 30, 32, 45, 60, 64, 81, 100, 121, 125, 128, 210, 256, 512, 625, 1000, 1024, 2048}) {
			multi::array<complex, 1> arr(multi::extents_t<1>{n}, complex{});
			for(int i = 0; i != n; ++i) {
				arr[i] = complex{static_cast<double>((i * 3) % 7) - 3.0, static_cast<double>((i * 5) % 11) - 5.0};
			}
			auto const ref = dft_reference(arr, multi::fft_forward);
			multi::fft_inplace(arr, multi::fft_forward);
			BOOST_TEST( max_abs_diff(arr, ref) < 1e-8 );
		}
	}

	// large prime and prime-containing sizes vs a direct DFT (exercises the Bluestein path)
	{
		for(int const n : {67, 101, 134, 331, 1009}) {  // 67, 101, 331, 1009 prime > 64; 134 = 2 * 67
			multi::array<complex, 1> arr(multi::extents_t<1>{n}, complex{});
			for(int i = 0; i != n; ++i) {
				arr[i] = complex{static_cast<double>((i * 3) % 7) - 3.0, static_cast<double>((i * 5) % 11) - 5.0};
			}
			auto const ref = dft_reference(arr, multi::fft_forward);
			multi::fft_inplace(arr, multi::fft_forward);
			BOOST_TEST( max_abs_diff(arr, ref) < 1e-7 );
		}
	}

	// a reusable plan gives the same answers as the one-shot interface, repeatedly
	{
		multi::array<complex, 2> arr({6, 10}, complex{});
		for(int i = 0; i != 6; ++i) {
			for(int j = 0; j != 10; ++j) {
				arr[i][j] = complex{static_cast<double>(i - j), static_cast<double>((i * j) % 5)};
			}
		}

		multi::fft_plan<2, complex> const plan{arr.sizes(), multi::fft_forward};  // CTAD from a prototype array

		auto a1 = arr;
		auto a2 = arr;
		plan.execute(a1.home());
		multi::fft_inplace(a2, multi::fft_forward);
		BOOST_TEST( max_abs_diff(a1.elements(), a2.elements()) < tol );

		auto a3 = arr;  // second execution of the same plan, same result
		plan.execute(a3.home());
		BOOST_TEST( max_abs_diff(a3.elements(), a2.elements()) < tol );
	}

	// a plan built from extents (no prototype array) round-trips fwd + bwd
	{
		multi::fft_plan<2, complex> const fwd{
			multi::extents_t<2>{5, 8}
				.sizes(),
			multi::fft_forward
		};
		multi::fft_plan<2, complex> const bwd{
			multi::extents_t<2>{5, 8}
				.sizes(),
			multi::fft_backward
		};

		multi::array<complex, 2> arr({5, 8}, complex{});
		for(int i = 0; i != 5; ++i) {
			for(int j = 0; j != 8; ++j) {
				arr[i][j] = complex{static_cast<double>(i + j), static_cast<double>(i - j)};
			}
		}
		auto const original = arr;
		bwd.execute(fwd.execute(arr.home()));
		auto const nn = static_cast<double>(arr.num_elements());
		double     m  = 0.0;
		for(int i = 0; i != 5; ++i) {
			for(int j = 0; j != 8; ++j) {
				m = std::max(m, std::abs(arr[i][j] - original[i][j] * nn));
			}
		}
		BOOST_TEST( m < tol );
	}

	// the same plan applies to arrays of the same sizes but different layout (strided sub-block)
	{
		multi::array<complex, 2> big({8, 12}, complex{});
		for(int i = 0; i != 8; ++i) {
			for(int j = 0; j != 12; ++j) {
				big[i][j] = complex{static_cast<double>((i * j) % 7), static_cast<double>(i - j)};
			}
		}
		auto&& block = big({2, 6}, {3, 9});  // 4 x 6 strided view

		multi::array<complex, 2> flat{block};  // contiguous copy of the same values

		multi::fft_plan<2, complex> const plan{flat.sizes(), multi::fft_forward};
		plan.execute(flat.home());   // contiguous layout
		plan.execute(block.home());  // strided layout, same plan

		double m = 0.0;
		for(int i = 0; i != 4; ++i) {
			for(int j = 0; j != 6; ++j) {
				m = std::max(m, std::abs(block[i][j] - flat[i][j]));
			}
		}
		BOOST_TEST( m < tol );
	}

	// long fiber (six-step path): forward+backward round-trip and DC component
	{
		int const                n = 65536;
		multi::array<complex, 1> arr(multi::extents_t<1>{n}, complex{});
		complex                  sum{};
		for(int i = 0; i != n; ++i) {
			arr[i] = complex{static_cast<double>((i * 3) % 13) - 6.0, static_cast<double>((i * 7) % 11) - 5.0};
			sum += arr[i];
		}
		auto const original = arr;
		multi::fft_inplace(arr, multi::fft_forward);
		BOOST_TEST( std::abs(arr[0] - sum) / std::abs(sum) < tol );
		multi::fft_inplace(arr, multi::fft_backward);
		double m = 0.0;
		for(int i = 0; i != n; ++i) {
			m = std::max(m, std::abs(arr[i] - original[i] * static_cast<double>(n)));
		}
		BOOST_TEST( m / static_cast<double>(n) < tol );
	}

	// plan applied on a cursor (.home()): extents live in the plan, the cursor
	// supplies base + strides
	{
		multi::array<complex, 2> arr({6, 10}, complex{});
		for(int i = 0; i != 6; ++i) {
			for(int j = 0; j != 10; ++j) {
				arr[i][j] = complex{static_cast<double>(i - j), static_cast<double>((i * j) % 7)};
			}
		}
		auto reference = arr;

		multi::fft_plan<2, complex> const plan{
			multi::extents_t<2>{6, 10}
				.sizes(),
			multi::fft_forward
		};
		plan.execute(arr.home());
		plan.execute(reference.home());
		BOOST_TEST( max_abs_diff(arr.elements(), reference.elements()) < tol );
	}

	// cursor application on a strided sub-block matches the same plan on a contiguous copy
	{
		multi::array<complex, 3> big({6, 7, 8}, complex{});
		int                      c = 0;
		for(auto& e : big.elements()) {
			e = complex{static_cast<double>(c % 11) - 5.0, static_cast<double>(c % 7) - 3.0};
			++c;
		}
		auto&& blk = big({1, 5}, {2, 6}, {1, 7});  // 4 x 4 x 6 strided view

		multi::array<complex, 3> flat{blk};  // contiguous copy of the same values

		multi::fft_plan<3, complex> const plan{
			multi::extents_t<3>{4, 4, 6}
				.sizes(),
			multi::fft_forward
		};
		plan.execute(flat.home());  // contiguous cursor
		plan.execute(blk.home());   // strided cursor, same plan

		BOOST_TEST( max_abs_diff(blk.elements(), flat.elements()) < tol );
	}

	// 3D round-trip
	{
		multi::array<complex, 3> arr({3, 4, 5}, complex{});
		int                      c = 0;
		for(int i = 0; i != 3; ++i) {
			for(int j = 0; j != 4; ++j) {
				for(int k = 0; k != 5; ++k) {
					arr[i][j][k] = complex{static_cast<double>(c), static_cast<double>(-c)};
					++c;
				}
			}
		}
		auto const original = arr;
		auto const nn       = static_cast<double>(arr.num_elements());
		multi::fft_inplace(arr, multi::fft_forward);
		multi::fft_inplace(arr, multi::fft_backward);
		double m = 0.0;
		for(int i = 0; i != 3; ++i) {
			for(int j = 0; j != 4; ++j) {
				for(int k = 0; k != 5; ++k) {
					m = std::max(m, std::abs(arr[i][j][k] - original[i][j][k] * nn));
				}
			}
		}
		BOOST_TEST( m < tol );
	}

	// 4D: round-trip, and 2D-of-2D composition (each axis transformed exactly once)
	{
		multi::array<complex, 4> arr({3, 4, 5, 6}, complex{});
		int                      c = 0;
		for(auto& e : arr.elements()) {
			e = complex{static_cast<double>((c * 3) % 11) - 5.0, static_cast<double>((c * 7) % 13) - 6.0};
			++c;
		}
		auto const original = arr;

		// reference: transform axes pairwise via 2D transforms of slices
		auto ref = arr;
		for(int i = 0; i != 3; ++i) {
			for(int j = 0; j != 4; ++j) {
				multi::fft_inplace(ref[i][j], multi::fft_forward);
			}  // last two axes per [5][6] slab
		}
		auto&& r2 = ref.rotated().rotated();  // axes (2,3,0,1): last two are original 0,1
		for(int k = 0; k != 5; ++k) {
			for(int l = 0; l != 6; ++l) {
				multi::fft_inplace(r2[k][l], multi::fft_forward);
			}
		}

		multi::fft_inplace(arr, multi::fft_forward);
		BOOST_TEST( max_abs_diff(arr.elements(), ref.elements()) / 360.0 < tol );

		multi::fft_inplace(arr, multi::fft_backward);
		auto const nn = static_cast<double>(arr.num_elements());
		double     m  = 0.0;
		auto       it = original.elements().begin();
		for(auto const& e : arr.elements()) {
			m = std::max(m, std::abs(e - *it++ * nn));
		}
		BOOST_TEST( m / nn < tol );
	}
	{
		multi::array<complex, 3> arr = {
			{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
			{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
			{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
			{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}
		};

		multi::fft_inplace(arr, multi::fft_forward);
		// multi::fft_inplace(arr.flattened(), multi::fft_forward);
	}

	// fft_plan<D, TW>: TW (twiddle-table type) defaults to complex<double> and
	// is independent of T (the executed array's element type, deduced fresh
	// per execute() call) -- see fft.NOTES.md §9.2. Same-type plan/array
	// keeps today's numerics; ONE plan built with the default TW also
	// executes a complex<float> array, reused across two different arrays
	// without rebuilding any tables.
	{
		using complexf = std::complex<float>;

		constexpr multi::ssize_t nn = 256;

		multi::array<complexf, 1> arr(multi::extents_t<1>{nn}, complexf{});
		multi::array<complexf, 1> arr2(multi::extents_t<1>{nn}, complexf{});
		for(multi::ssize_t i = 0; i != nn; ++i) {
			auto const fi = static_cast<float>(i);
			arr[i]        = complexf{fi, -fi};
			arr2[i]       = complexf{-fi, fi * 0.5F};
		}

		multi::array<complex, 1> arr_d(multi::extents_t<1>{nn}, complex{});
		multi::array<complex, 1> arr2_d(multi::extents_t<1>{nn}, complex{});
		for(multi::ssize_t i = 0; i != nn; ++i) {
			arr_d[i]  = complex{arr[i].real(), arr[i].imag()};
			arr2_d[i] = complex{arr2[i].real(), arr2[i].imag()};
		}
		auto const ref  = dft_reference(arr_d, multi::fft_forward);
		auto const ref2 = dft_reference(arr2_d, multi::fft_forward);

		multi::fft_plan<1> const plan{arr.sizes(), multi::fft_forward};  // TW defaults to complex<double>
		plan.execute(arr.home());
		plan.execute(arr2.home());  // same plan, second (different) float array: no rebuild

		double m  = 0.0;
		double m2 = 0.0;
		for(multi::ssize_t i = 0; i != nn; ++i) {
			m  = std::max(m, std::abs(complex{arr[i].real(), arr[i].imag()} - ref[i]));
			m2 = std::max(m2, std::abs(complex{arr2[i].real(), arr2[i].imag()} - ref2[i]));
		}
		double const ftol = 1e-3 * static_cast<double>(nn);  // float precision, loose
		BOOST_TEST( m < ftol );
		BOOST_TEST( m2 < ftol );
	}

	// execute() threads a caller-supplied allocator through instead of
	// always using its own std::allocator<T> (fft.NOTES.md §9.2/§10.4(b)):
	// same size requested every call, exactly plan.scratch_elements().
	{
		std::size_t alloc_count = 0, dealloc_count = 0, last_n = 0;
		counting_allocator<complex> alloc{&alloc_count, &dealloc_count, &last_n};

		multi::array<complex, 1> arr(multi::extents_t<1>{1024}, complex{1.0, 0.0});
		multi::fft_plan<1> const plan{arr.sizes(), multi::fft_forward};

		plan.execute(arr.home(), alloc);
		std::size_t const n_after_1st = last_n;
		BOOST_TEST( alloc_count == 1 );
		BOOST_TEST( dealloc_count == 1 );
		BOOST_TEST( n_after_1st == plan.scratch_elements() );

		plan.execute(arr.home(), alloc);  // same plan, same allocator, again
		BOOST_TEST( alloc_count == 2 );
		BOOST_TEST( dealloc_count == 2 );
		BOOST_TEST( last_n == n_after_1st );  // stable across repeated calls
	}

	return boost::report_errors();
}

// NOLINTEND(altera-id-dependent-backward-branch,altera-unroll-loops,readability-identifier-length)
