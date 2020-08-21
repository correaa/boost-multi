#if COMPILATION_INSTRUCTIONS
#mpicxx -I$HOME/prj/alf $0 -g -o $0x -lfftw3 -lfftw3_mpi &&mpirun -n 2 valgrind $0x;exit
$CXXX $CXXFLAGS -O2 -g `mpicxx -showme:compile|sed 's/-pthread/ /g'` -I$HOME/prj/alf $0 -o $0x `mpicxx -showme:link|sed 's/-pthread/ /g'` -lfftw3 -lfftw3_mpi -lboost_timer&&mpirun -n 5 $0x;exit
#endif

#ifndef MULTI_FFTW_MPI_DISTRIBUTION_HPP
#define MULTI_FFTW_MPI_DISTRIBUTION_HPP

#include <fftw3-mpi.h>

#include<boost/mpi3/communicator.hpp>

#include "../../../array_ref.hpp"

#include <experimental/tuple>

namespace boost{
namespace multi{
namespace fftw{
namespace mpi{

namespace bmpi3 = boost::mpi3;

template<std::ptrdiff_t ElementSize>
class distribution{
public:
	using difference_type = std::ptrdiff_t;
private:
	difference_type local_count_;
	difference_type local_n0_;
	difference_type local_0_start_;
	difference_type local_n1_;
	difference_type local_1_start_;
	static std::array<difference_type, 2> sizes(boost::multi::extensions_type_<2> const& ext){
		return std::experimental::apply([](auto... e){return std::array<ptrdiff_t, 2>{e.size()...};}, ext);
	}
public:
	static_assert(ElementSize%sizeof(double)==0, "!");
	distribution(
		extensions_type_<2> const& ext, boost::mpi3::communicator const& comm, 
		difference_type block0 = FFTW_MPI_DEFAULT_BLOCK, difference_type block1 = FFTW_MPI_DEFAULT_BLOCK
	) :
	local_count_{
		fftw_mpi_local_size_many_transposed(
			2, sizes(ext).data(), ElementSize/sizeof(double),
			block0, block1, comm.get(),
			&local_n0_, &local_0_start_,
			&local_n1_, &local_1_start_
		)
	/*	fftw_mpi_local_size_many(
			2, sizes(ext).data(), ElementSize/sizeof(double),
			block0, comm.get(),
			&local_n0_, &local_0_start_
		)*/
	}{
	// FFTW_MPI_DEFAULT_BLOCK = (size + comm.size - 1)/comm.size
		assert(block0*comm.size() >= std::get<0>(ext).size() or block0 == FFTW_MPI_DEFAULT_BLOCK) ;
	}
	difference_type   local_count() const{return local_count_*(ElementSize/sizeof(double));}
	multi::iextension local_extension() const{return {local_0_start_, local_0_start_ + local_n0_};}
	difference_type   local_size() const{return local_n0_;}
	bool operator==(distribution const& other) const{
		return std::tie(this->local_count_, this->local_n0_, this->local_0_start_, this->local_n1_, this->local_1_start_)
			== std::tie(other.local_count_, other.local_n0_, other.local_0_start_, other.local_n1_, other.local_1_start_);
	}
	bool operator!=(distribution const& other) const{return not operator==(other);}
};

}}}}

#if not __INCLUDE_LEVEL__

#include<boost/mpi3/main_environment.hpp>
#include<boost/mpi3/ostream.hpp>

#include "../../fftw/mpi/environment.hpp"

namespace bmpi3 = boost::mpi3;
namespace multi = boost::multi;

int bmpi3::main(int, char*[], mpi3::environment& env){
	multi::fftw::mpi::environment fenv(env);

	auto world = env.world();

	mpi3::ostream os{world};
	
//	auto block = 111;
	multi::fftw::mpi::distribution<sizeof(double)> dist({533, 13}, world, (533+world.size()-1)/world.size());//533/world.size());
	
	os<< "local count "<< dist.local_count() <<std::endl;
	os<< "local extension "<< dist.local_extension() <<' '<< dist.local_extension().size() <<std::endl;

	return 0;
}
#endif
#endif

