#ifdef COMPILATION_INSTRUCTIONS
c++ -std=c++14 -Wall -Wextra -Wpedantic -lblas -DADD_ $0 -o $0x.x  && time $0x.x $@ && rm -f $0x.x; exit
#endif

#include "../../blas.hpp"

#include "../../../array.hpp"

#include<complex>
#include<cassert>

using std::cout;
namespace multi = boost::multi;

int main(){
	{
		multi::array<double, 2> const A = {
			{1.,  2.,  3.},
			{5.,  6.,  7.},
			{9., 10., 11.}
		};
		multi::array<double, 2> const B = {
			{10.,  12.,  31.},
			{51.,  61.,  74.},
			{91., 101., 111.}
		};
		using multi::blas::gemm;
		auto C = gemm(A, B); // C = B.A // is this ok as interface?
		for(auto i = 0; i != 3; ++i){
			for(auto j = 0; j != 3; ++j){
				cout << C[i][j] <<' ';
			}
			cout << std::endl;
		}
	}
}

