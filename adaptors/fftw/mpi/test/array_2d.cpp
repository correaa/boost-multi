#if COMPILATION_INSTRUCTIONS
$CXXX $CXXFLAGS -std=c++14 `mpicxx -showme:compile|sed 's/-pthread/ /g'` -I$HOME/prj/alf -std=c++14 $0 -o $0x `mpicxx -showme:link|sed 's/-pthread/ /g'` -lfftw3 -lfftw3_mpi&&time mpirun -n 4 $0x&&rm $0x;exit
#endif

#if 1
#include "../../../../array.hpp"
#include "../../../../config/NODISCARD.hpp"

#include<boost/mpi3/communicator.hpp>
#include<boost/mpi3/environment.hpp>

#include "../../../fftw.hpp"

#include <fftw3-mpi.h>
#endif

#include "../../../fftw/mpi.hpp"

//#include<boost/mpi3/main.hpp>
//#include<boost/mpi3/environment.hpp>
//#include<boost/mpi3/ostream.hpp>

//namespace mpi3  = boost::mpi3;
namespace multi = boost::multi;

//int mpi3::main(int, char*[], mpi3::communicator){
int main(){
#if 0
	multi::fftw::mpi::environment fenv;
	
	{
		multi::fftw::mpi::array<std::complex<double>, 2> G(world);
		assert( G.extensions() == decltype(G.extensions()){} );
		assert( G.local_cutout().empty() );
	}

	multi::fftw::mpi::array<std::complex<double>, 2> G({41, 321}, world);

	if(auto x = G.local_cutout().extensions())
		for(auto i : std::get<0>(x))
			for(auto j : std::get<1>(x))
				G.local_cutout()[i][j] = std::complex<double>(i + j, i + 2*j);

#endif

	boost::multi::array<std::complex<double>, 2, std::allocator<std::complex<double>> > L({10, 20});//(nullptr, {});// = G; // world replicas
/*
	assert( L == G );
	
	using multi::fftw::dft_forward;

	dft_forward(L, L); // dft in replicas
	dft_forward(G, G);

	if(auto x = G.local_cutout().extensions())
		for(auto i : std::get<0>(x))
			for(auto j : std::get<1>(x))
				if(not(std::abs(G.local_cutout()[i][j] - L[i][j]) < 1e-8)) std::cout<< std::abs(G.local_cutout()[i][j] - L[i][j]) << std::endl;
*/
	return 0;
}

