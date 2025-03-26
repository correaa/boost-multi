// Copyright 2025 Alfredo A. Correa

// #include "../../lapack/syev.hpp"

#include <boost/multi/adaptors/blas/gemm.hpp>
#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace multi = boost::multi;

extern "C" {
void dgesvd_(char const& jobu, char const& jobvt, int const& mm, int const& nn, double* aa, int const& lda, double* ss, double* uu, int const& ldu, double* vt, int const& ldvt, double* work, int const& lwork, int& info);  // NOLINT(readability-identifier-naming)
}

namespace boost::multi::lapack {

template<class Alloc, class AArray2D, class UArray2D, class SArray1D, class VTArray2D>
void gesvd(AArray2D&& AA, UArray2D&& UU, SArray1D&& ss, VTArray2D&& VV, Alloc alloc) {
	assert( AA.size() == UU.size() );
	assert( (~AA).size() == VV.size() );

	assert((~AA).stride() == 1);
	assert(ss.stride() == 1);
	assert((~UU).stride() == 1);
	assert((~VV).stride() == 1);

	int    info;   // NOLINT(cppcoreguidelines-init-variables) init by function
	double dwork;  // NOLINT(cppcoreguidelines-init-variables) init by function

	dgesvd_(
		'A' /*all left*/, 'A' /*all right*/, 
		static_cast<int>(VV.size()), static_cast<int>(UU.size()),
		AA.base(), static_cast<int>(AA.stride()),
		ss.base(),
		VV.base(), static_cast<int>(VV.stride()),
		UU.base(), static_cast<int>(UU.stride()),
		&dwork, -1, info
	);
	if(info != 0) { throw std::runtime_error("Error in DGESVD work estimation, info: " + std::to_string(info)); }

	int const     lwork = static_cast<int>(dwork);
	double* const work  = alloc.allocate(lwork);

	dgesvd_(
		'A' /*all left*/, 'A' /*all right*/,
		static_cast<int>(VV.size()), static_cast<int>(UU.size()),
		AA.base(), static_cast<int>(AA.stride()),
		ss.base(),
		VV.base(), static_cast<int>(VV.stride()),
		UU.base(), static_cast<int>(UU.stride()),
		work, lwork, info
	);
	alloc.deallocate(work, lwork);

	if(info != 0) { throw std::runtime_error("Error in DGESVD computation, info: " + std::to_string(info)); }

	(void)std::forward<AArray2D>(AA);
}

template<
	template<typename> class AllocT = std::allocator,
	class AArray2D, class UArray2D, class SArray1D, class VTArray2D,
	class Alloc = AllocT<typename std::decay_t<AArray2D>::element_type>
>
void gesvd(AArray2D&& AA, UArray2D&& UU, SArray1D&& ss, VTArray2D&& VV) {
	return gesvd(std::forward<AArray2D>(AA), std::forward<UArray2D>(UU), std::forward<SArray1D>(ss), std::forward<VTArray2D>(VV), Alloc{});
}

}  // end namespace boost::multi::lapack

auto main() -> int {  // NOLINT(bugprone-exception-escape)

	// {
	//  multi::array<double, 2> AA = {
	//      { 2.27,  0.94, 1.07,  0.63, -2.35,  0.62},
	//      {-1.54, -0.78, 1.22,  2.93,  2.30, -7.39},
	//      { 1.15, -0.48, 0.79, -1.45,  1.03,  1.03},
	//      {-1.94, -3.09, 0.63,  2.30, -2.57, -2.57},
	//  };

	//  auto const AA_copy = AA;

	//  // Output arrays
	//  multi::array<double, 1> ss(std::min(AA.size(), (~AA).size()));  // Singular values

	//  multi::array<double, 2> UU({(~AA).size(), (~AA).size()});  // Left singular vectors
	//  multi::array<double, 2> VT({AA.size(), AA.size()});        // Right singular vectors

	//  boost::multi::lapack::gesvd(AA, UU, ss, VT);

	//  std::cout << "Original array:\n";
	//  {
	//      auto [is, js] = AA.extensions();
	//      for(auto i : is) {
	//          for(auto j : js) {  // NOLINT(altera-unroll-loops)
	//              std::cout << AA_copy[i][j] << ' ';
	//          }
	//          std::cout << '\n';
	//      }
	//  }

	//  multi::array<double, 2> SS({ss.extension(), ss.extension()}, 0.0);
	//  std::copy(ss.begin(), ss.end(), SS.diagonal().begin());

	//  // Print singular values
	//  std::cout << "Singular values:\n";
	//  for(auto i : ss.extension()) {  // NOLINT(altera-unroll-loops)
	//      std::cout << ss[i] << ' ';
	//  }
	//  std::cout << '\n';

	//  std::cout << "Singular vectors as array:\n";
	//  for(auto const& row : SS) {
	//      for(auto const& elem : row) {  // NOLINT(altera-unroll-loops)
	//          std::cout << elem << ' ';
	//      }
	//      std::cout << '\n';
	//  }

	//  // Print left singular vectors
	//  std::cout << "Left singular vectors:\n";
	//  for(auto const& row : UU) {
	//      for(auto const& elem : row) {  // NOLINT(altera-unroll-loops)
	//          std::cout << elem << ' ';
	//      }
	//      std::cout << '\n';
	//  }

	//  // Print right singular vectors
	//  std::cout << "Right singular vectors:\n";
	//  {
	//      auto [is, js] = VT.extensions();
	//      for(auto i : is) {
	//          for(auto j : js) {  // NOLINT(altera-unroll-loops)
	//              std::cout << VT[i][j] << ' ';
	//          }
	//          std::cout << '\n';
	//      }
	//  }
	//  // Singular values: s
	//  // 9.43397 4.71924 3.19716 1.88613
	//  // Left singular vectors: UU
	//  // -0.282618 -0.218724 0.114543 0.392407 0.114648 -0.831889
	//  // 0.0753653 0.370403 -0.0655296 -0.282965 0.866854 -0.146024
	//  // 0.697198 0.484235 0.360692 0.216255 -0.210349 -0.241494
	//  // 0.338129 -0.66227 0.546151 -0.338876 0.184255 -2.94997e-06
	//  // 0.558963 -0.354894 -0.725448 0.116975 0.0640779 -0.132465
	//  // 0.039885 -0.126202 0.167128 0.768545 0.391302 0.459099
	//  // Right singular vectors: VT
	//  // -0.133831 -0.393446 0.908492 0.0439558
	//  // 0.880508 0.372702 0.28873 0.0493377
	//  // -0.152352 0.213988 0.0235597 0.964595
	//  // 0.428467 -0.812713 -0.301202 0.255325

	//  // BOOST_TEST( std::abs(ss[0] - 9.43397 ) < 1e-4 );
	//  // BOOST_TEST( std::abs(ss[3] - 1.88613 ) < 1e-4 );

	//  // BOOST_TEST( std::abs(UU[0][0] - -0.282618) < 1e-4 );
	//  // BOOST_TEST( std::abs(UU[0][5] - -0.831889) < 1e-4 );
	//  // BOOST_TEST( std::abs(UU[5][0] -  0.039885) < 1e-4 );
	//  // BOOST_TEST( std::abs(UU[5][5] -  0.459099) < 1e-4 );

	//  // BOOST_TEST( std::abs(VT[0][0] - -0.133831) < 1e-4 );
	//  // BOOST_TEST( std::abs(VT[0][3] - 0.0439558) < 1e-4 );
	//  // BOOST_TEST( std::abs(VT[3][0] -  0.42846 ) < 1e-4 );
	//  // BOOST_TEST( std::abs(VT[3][3] -  0.255325) < 1e-4 );
	// }

	// {
	//  multi::array<double, 2> AA = {
	//      {0.5, 1.0},
	//      {2.0, 2.5},
	//  };

	//  auto const AA_gold = AA;

	//  // Output arrays
	//  multi::array<double, 1> ss(std::min(AA.size(), (~AA).size()));  // Singular values

	//  multi::array<double, 2> UU({(~AA).size(), (~AA).size()});  // Left singular vectors
	//  multi::array<double, 2> VT({AA.size(), AA.size()});        // Right singular vectors

	//  multi::lapack::gesvd(AA, UU, ss, VT);  // AA == VT.SS.(UU^T)

	//  multi::array<double, 2> SS({ss.extension(), ss.extension()}, 0.0);
	//  std::copy(ss.begin(), ss.end(), SS.diagonal().begin());

	//  // auto const SSUUT   = +multi::blas::gemm(1.0, SS, ~UU);
	//  auto const AA_test = +multi::blas::gemm(1.0, VT, +multi::blas::gemm(1.0, SS, ~UU));

	//  BOOST_TEST( std::abs(AA_test[0][0] - AA_gold[0][0]) < 1.0e-4 );
	//  BOOST_TEST( std::abs(AA_test[0][1] - AA_gold[0][1]) < 1.0e-4 );
	//  BOOST_TEST( std::abs(AA_test[1][0] - AA_gold[1][0]) < 1.0e-4 );
	//  BOOST_TEST( std::abs(AA_test[1][1] - AA_gold[1][1]) < 1.0e-4 );
	// }
	{
		multi::array<double, 2> AA = {
			{0.5, 1.0},
			{2.0, 2.5},
		};

		auto const AA_gold = AA;

		// Output arrays
		multi::array<double, 1> ss(std::min(AA.size(), (~AA).size()));  // Singular values

		multi::array<double, 2> UU({AA.size(), AA.size()});        // Right singular vectors
		multi::array<double, 2> VV({(~AA).size(), (~AA).size()});  // Left singular vectors

		multi::lapack::gesvd(AA, UU, ss, VV);  // AA == UU.SS.(VV^T)

		multi::array<double, 2> SS({ss.extension(), ss.extension()}, 0.0);
		std::copy(ss.begin(), ss.end(), SS.diagonal().begin());

		auto const AA_test = +multi::blas::gemm(1.0, UU, +multi::blas::gemm(1.0, SS, ~VV));
		// AA_test = UU.SS.(VV^T);

		BOOST_TEST( std::abs(AA_test[0][0] - AA_gold[0][0]) < 1.0e-4 );
		BOOST_TEST( std::abs(AA_test[0][1] - AA_gold[0][1]) < 1.0e-4 );
		BOOST_TEST( std::abs(AA_test[1][0] - AA_gold[1][0]) < 1.0e-4 );
		BOOST_TEST( std::abs(AA_test[1][1] - AA_gold[1][1]) < 1.0e-4 );
	}

	return boost::report_errors();
}
