#ifdef COMPILATION_INSTRUCTIONS
$CXX -O3 -std=c++17 -Wall -Wextra -Wpedantic -Werror `#-Wfatal-errors` -I$HOME/include $0 -o $0.x && $0.x $@ && rm -f $0.x; exit
#endif

#include<iostream>

#include "../array_ref.hpp"
#include "../array.hpp"

#include<iostream>
#include<vector>
#include<complex>

namespace multi = boost::multi;
using std::cout; using std::cerr;

namespace fake{
typedef double fftw_complex[2];
void fftw_plan_dft(
	int rank, const int *n, 
	fftw_complex *in, fftw_complex *out, int sign, unsigned flags){
	(void)rank, (void)n, (void)in, (void)out, (void)sign, (void)flags;
}
}

int main(){
	using std::complex;

	multi::array<complex<double>, 2> const in = {
		{150., 16., 17., 18., 19.},
		{  5.,  5.,  5.,  5.,  5.}, 
		{100., 11., 12., 13., 14.}, 
		{ 50.,  6.,  7.,  8.,  9.}  
	};
	multi::array<std::complex<double>, 2> out(extensions(in));

	assert( dimensionality(out) == dimensionality(in) );
	assert( sizes(out) == sizes(in) );

	fake::fftw_plan_dft(
		dimensionality(in), in.sizes_as<int>().data(),
		(fake::fftw_complex*)in.data_elements(), (fake::fftw_complex*)out.data_elements(), 1, 0
	);

}

