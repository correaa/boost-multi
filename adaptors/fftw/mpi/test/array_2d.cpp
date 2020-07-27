#if COMPILATION_INSTRUCTIONS
mpic++ -I$HOME/prj/alf $0 -o $0x -lfftw3 -lfftw3_mpi&&time mpirun -n 4 $0x&&rm $0x;exit
#endif

#include "../../../fftw/mpi.hpp"

#include<boost/mpi3/main.hpp>
#include<boost/mpi3/environment.hpp>
#include<boost/mpi3/ostream.hpp>
#include "../../../fftw.hpp"

namespace mpi3  = boost::mpi3;
namespace multi = boost::multi;

int mpi3::main(int, char*[], mpi3::communicator world){
	multi::fftw::mpi::environment fenv;

	multi::fftw::mpi::array<std::complex<double>, 2> A({41, 321}, world);

	{
		auto x = A.local().extensions();
		for(auto i : std::get<0>(x))
			for(auto j : std::get<1>(x))
				A.local()[i][j] = std::complex<double>(i + j, i + 2*j);
	}
	
	multi::array<std::complex<double>, 2> A2 = A;
	assert( A2 == A );

	A = A2;
	
	using multi::fftw::dft_forward;

	dft_forward(A , A );
	dft_forward(A2, A2);

	{
		auto x = A.local().extensions();
		for(auto i : std::get<0>(x))
			for(auto j : std::get<1>(x))
				assert(std::abs(A.local()[i][j] - A2[i][j]) < 1e-8 );
	}
	return 0;
}

