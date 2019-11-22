#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -Wpedantic $0 -o $0x -lblas `pkg-config --libs blas`&&$0x&&rm $0x; exit
#endif

#include "../../blas.hpp"

#include "../../../array.hpp"

#include<complex>
#include<cassert>
#include<numeric>

using std::cout;
namespace multi = boost::multi;

int main(){
	{
		using Z = std::complex<double>; Z const I(0.,1.);
		multi::array<Z, 2> const A = {
			{1. + 2.*I,  2.,  3.,  4.},
			{5.,  6. + 3.*I,  7.,  8.},
			{9., 10., 11.+ 4.*I, 12.}
		};
		using multi::blas::asum;
		assert(asum(A[1]) == std::accumulate(begin(A[1]), end(A[1]), 0., [](auto&& a, auto&& b){return a + std::abs(real(b)) + std::abs(imag(b));}));
	}
}

