#if COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXXX $CXXFLAGS `mpicxx -showme:compile|sed 's/-pthread/ /g'` -I$HOME/prj/alf $0 -o $0x `mpicxx -showme:link|sed 's/-pthread/ /g'` -lfftw3 -lfftw3_mpi&&time mpirun -n 1 $0x&&rm $0x;exit
#ln -sf $0 $0.cpp;mpicxx -g -I$HOME/prj/alf $0.cpp -o $0x -lfftw3 -lfftw3_mpi&&time mpirun -n 4 valgrind --leak-check=full --track-origins=yes --show-leak-kinds=all --suppressions=$HOME/prj/alf/boost/mpi3/test/communicator_main.cpp.openmpi.supp --error-exitcode=1 $0x&&rm $0x $0.cpp;exit
#endif
// © Alfredo A. Correa 2020
// apt-get install libfftw3-mpi-dev
// compile with: mpicc simple_mpi_example.c  -Wl,-rpath=/usr/local/lib -lfftw3_mpi -lfftw3 -o simple_mpi_example */

#ifndef MULTI_ADAPTOR_FFTW_MPI_HPP
#define MULTI_ADAPTOR_FFTW_MPI_HPP

#include "../fftw/memory.hpp"

#include "../../array.hpp"

#include "../../adaptors/fftw.hpp"

#include<boost/mpi3/communicator.hpp>
#include<boost/mpi3/environment.hpp>

#include <fftw3-mpi.h>

//#include "../../config/NODISCARD.hpp"
#if 0

#include<boost/mpi3/communicator.hpp>
#include<boost/mpi3/environment.hpp>

#include "../fftw.hpp"

#include <fftw3-mpi.h>
#endif

#if 1
#if 0
namespace boost{
namespace multi{
namespace fftw{

template<typename T>
struct allocator{
	using value_type      = T;
	using pointer         = value_type*;
	using size_type 	  = std::size_t;
	using difference_type =	std::ptrdiff_t;
	using propagate_on_container_move_assignment = std::true_type;
	NODISCARD("to avoid memory leak") 
	pointer allocate(size_type n){ return static_cast<pointer>(fftw_malloc(sizeof(T)*n));}
	void deallocate(pointer data, size_type){fftw_free(data);}
};

template<> allocator<std::complex<double>>::pointer allocator<std::complex<double>>::allocate(size_type n){return reinterpret_cast<std::complex<double>*>(fftw_alloc_complex(n));}
template<> allocator<             double >::pointer allocator<             double >::allocate(size_type n){return                                         fftw_alloc_real(n)    ;}

#if 0
template<>
struct allocator<std::complex<double>>{
	using value_type      = std::complex<double>;
	using pointer         = value_type*;
	using size_type 	  = std::size_t;
	using difference_type =	std::ptrdiff_t;
	using propagate_on_container_move_assignment = std::true_type;
	NODISCARD("to avoid memory leak") 
	pointer allocate(size_type n){return reinterpret_cast<std::complex<double>*>(fftw_alloc_complex(n));}
	void deallocate(pointer data, size_type){fftw_free(data);}
};

template<>
struct allocator<double>{
	using value_type      = double;
	using pointer         = value_type*;
	using size_type 	  = std::size_t;
	using difference_type =	std::ptrdiff_t;
	using propagate_on_container_move_assignment = std::true_type;
	NODISCARD("to avoid memory leak") 
	pointer allocate(size_type n){return fftw_alloc_real(n);}
	void deallocate(pointer data, size_type){fftw_free(data);}
};
#endif

}}}


#endif

namespace boost{namespace multi{namespace fftw{namespace mpi{

struct environment{
	 environment(boost::mpi3::environment&){fftw_mpi_init();}
	~environment(){fftw_mpi_cleanup();}
};

}}}}

namespace boost{
namespace multi{
namespace fftw{
namespace mpi{

template<class T, multi::dimensionality_type D, class Alloc>// = fftw::Allocator<T>> 
struct array_transposed;

namespace bmpi3 = boost::mpi3;

#if 0
struct layout_t{
	ptrdiff_t size          = -1;
	bool      is_transposed = false;
	ptrdiff_t block         = FFTW_MPI_DEFAULT_BLOCK;
	layout_t& tranpose(){is_transposed = not is_transposed; return *this;}
};

template<class Alloc>
struct array_base{
//protected:
	mutable bmpi3::communicator comm_;
	Alloc alloc_;
	typename std::allocator_traits<Alloc>::size_type local_count_;
	typename std::allocator_traits<Alloc>::pointer   local_data_;
	bmpi3::communicator const& comm(){return comm_;}
};

template<class T, class Ptr = T*>
class basic_array{
public:
	using local_pointer_t = Ptr;
protected:
	mutable bmpi3::communicator  comm_;
	layout_t                     layout_;
	local_pointer_t              local_data_;
	multi::layout_t<2>           local_layout_;
	basic_array(layout_t layout, multi::layout_t<2> ll, local_pointer_t p) : layout_{layout}, local_layout_{ll}, local_data_{p}{}
	template<class, dimensionality_type, class> friend class array;
public:
	layout_t layout(){return layout_;}
	multi::basic_array<T, 2, Ptr> local_cutout()&{return multi::basic_array<T, 2, Ptr>(local_layout_, local_data_);}
	multi::basic_array<T, 2, typename std::pointer_traits<Ptr>::template rebind<T const>> local_cutout() const&{return multi::basic_array<T, 2, typename std::pointer_traits<Ptr>::template rebind<T const>>(local_layout_, local_data_);}

	multi::extensions_type_<2> extensions() const{
		if(layout_.is_transposed) return {std::get<1>(local_layout_.extensions()), layout_.size};
		else                      return {layout_.size, std::get<1>(local_layout_.extensions())};
	}
};

template<class T, class Alloc>
struct array_transposed<T, 2, Alloc> : array_base<Alloc>{
	using element_type = T;
	
	array_ptr<T, 2, typename std::allocator_traits<Alloc>::pointer> local_ptr_;
	multi::extensions_type_<2> ext_;
	
	auto local_2d(multi::extensions_type_<2> ext){
		ptrdiff_t local_n0, local_0_start;
		ptrdiff_t local_n1, local_1_start;
		auto count = fftw_mpi_local_size_2d_transposed(
			std::get<0>(ext).size(), std::get<1>(ext).size(), this->comm_.get(), 
			&local_n0, &local_0_start,
			&local_n1, &local_1_start
		);
		assert( count >= local_n0*local_n1 );//+ local_n1*std::get<1>(ext).size() );
		return std::make_pair(
			count,
			multi::extensions_type_<2>{
				std::get<0>(ext), 
			//	{local_0_start, local_0_start + local_n0}, 
				{local_1_start, local_1_start + local_n1}
			}
		);
	}
	typename std::allocator_traits<Alloc>::size_type 
	     local_count_2d     (multi::extensions_type_<2> ext){return local_2d(ext).first;}
	auto local_extensions_2d(multi::extensions_type_<2> ext){return local_2d(ext).second;}

	array_transposed(multi::extensions_type_<2> ext, bmpi3::communicator comm = mpi3::environment::self(), Alloc alloc = {}) :
		array_base<Alloc>{
			comm, 
			alloc,
			local_count_2d(ext),
			alloc.allocate(local_count_2d(ext))
		},
		local_ptr_{this->local_data_, local_extensions_2d(ext)},
		ext_{ext}
	{
		if(not std::is_trivially_default_constructible<element_type>{})
			adl_alloc_uninitialized_default_construct_n(this->alloc_, local_ptr_->base(), local_ptr_->num_elements());
	}
	array_transposed(array<T, 2> const& other) :
		array_base<Alloc>{
			other.comm(),
			other.get_allocator(),
			local_count_2d(other.extensions()),
			this->alloc_.allocate(local_count_2d(other.extensions()))
		},
		local_ptr_{this->local_data_, local_extensions_2d(other.extensions())},
		ext_{other.extensions()}
	{
		static_assert( std::is_trivially_assignable<T&, T>{}, "!" );
		static_assert( sizeof(T)%sizeof(double)==0, "!" );
		
		fftw_plan p = fftw_mpi_plan_many_transpose(
			std::get<0>(other.extensions()).size(), std::get<1>(other.extensions()).size(), sizeof(T)/sizeof(double), 
			FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
			reinterpret_cast<double*>(const_cast<T*>(other.local_cutout().data_elements())), 
			reinterpret_cast<double*>(this->local_cutout().data_elements()),
			this->comm_.get(), FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT
		);

/*		fftw_plan p = fftw_mpi_plan_transpose(
			std::get<0>(other.extensions()).size(), std::get<1>(other.extensions()).size(),
			const_cast<double*>(other.local_cutout().data_elements()), this->local_cutout().data_elements(),
			this->comm_.get(), FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT
		);*/
		fftw_execute(p);
		fftw_destroy_plan(p);
	}
	
	array_ref <T, 2> local_cutout()      &{return *local_ptr_;}
	array_cref<T, 2> local_cutout() const&{return *local_ptr_;}
	
	~array_transposed() noexcept{this->alloc_.deallocate(this->local_data_, this->local_count_);}
	       auto extensions()                const&      {return ext_;}
	friend auto extensions(array_transposed const& self){return self.extensions();}
	ptrdiff_t num_elements() const&{return multi::layout_t<2>(extensions()).num_elements();}
	auto local_count() const{return this->local_count_;}
	auto local_data() const&{return this->local_data_;}
};
#endif

template<class T, multi::dimensionality_type D>
class layout_t;

template<class T>
class layout_t<T, 2>{
	static constexpr ptrdiff_t default_block = FFTW_MPI_DEFAULT_BLOCK;
	multi::extensions_type_<2> global_extensions_;
	mutable bmpi3::communicator comm_            ;//= bmpi3::environment::self();
	bool      is_transposed_                     ;//= false;
	ptrdiff_t block_                             ;//= default_block;
public:
	static constexpr dimensionality_type dimensionality = 2;
	layout_t(multi::extensions_type_<2> ext, bmpi3::communicator const& comm = bmpi3::environment::self()) : 
		global_extensions_{ext},
		comm_{comm},
		block_{FFTW_MPI_DEFAULT_BLOCK}
	{}
private:
	auto local() const{
		(void)is_transposed_;
		(void)block_;
		ptrdiff_t local_n0, local_0_start;
		if(std::get<0>(global_extensions_).size()==0 or std::get<1>(global_extensions_).size()==0) 
			return std::make_pair(ptrdiff_t{0}, multi::extensions_type_<2>{});
//		auto count = fftw_mpi_local_size_2d(std::get<0>(global_extensions_).size(), std::get<1>(global_extensions_).size(), comm_.get(), &local_n0, &local_0_start);
		
		std::array<ptrdiff_t, 2> const sizes_arr = {std::get<0>(global_extensions_).size(), std::get<1>(global_extensions_).size()};
		ptrdiff_t count = fftw_mpi_local_size_many(sizes_arr.size(), sizes_arr.data(), sizeof(T)/sizeof(double),
			FFTW_MPI_DEFAULT_BLOCK, comm_.get(), &local_n0, &local_0_start
		)/(sizeof(T)/sizeof(double));
		{
		//	http://www.fftw.org/fftw3_doc/MPI-Data-Distribution-Functions.html
			
			ptrdiff_t local_n0_C, local_0_start_C;
			ptrdiff_t local_n1_C, local_1_start_C;
			ptrdiff_t count_C = fftw_mpi_local_size_many_transposed( 
				sizes_arr.size(),sizes_arr.data(), sizeof(T)/sizeof(double), 
				FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, comm_.get(),
				&local_n0_C, &local_0_start_C,
				&local_n1_C, &local_1_start_C
			);
			assert(count_C == count);
			assert(local_n0_C == local_n0);
			assert(local_0_start_C == local_0_start);
		}
		assert( count > 0 );
		assert( count >= local_n0*std::get<1>(global_extensions_).size() );
		return std::make_pair(count, multi::extensions_type_<2>{{local_0_start, local_0_start + local_n0}, std::get<1>(global_extensions_)});
	}
public:
	auto size() const{return std::get<0>(global_extensions_).size();}
	multi::extensions_type_<2> global_extensions() const{return global_extensions_;}
	ptrdiff_t                  local_count      () const{return local_2d().first  ;}
	multi::extensions_type_<2> local_extensions () const{return local_2d().second ;}
	bmpi3::communicator&       comm()              const{return comm_;}
	ptrdiff_t                  block()             const{return block_;}
	bool operator==(layout_t const& other) const{return global_extensions_==other.global_extensions_ and comm_==other.comm_ and is_transposed_==other.is_transposed_ and block_==other.block_;}
	bool operator!=(layout_t const& other) const{return not operator==(other);}
};

template<class, dimensionality_type> class gathered_layout;

template<class T> class gathered_layout<T, dimensionality_type{2}>{
	multi::extensions_type_<2> global_extensions_;
	mutable bmpi3::communicator comm_            ;//= bmpi3::environment::self();
	ptrdiff_t block_                             ;//= default_block;
public:
	gathered_layout(multi::extensions_type_<2> ext, bmpi3::communicator comm = bmpi3::environment::self()) : 
		global_extensions_{ext},
		comm_{std::move(comm)},
		block_{std::get<0>(ext).size()}
	{}
	auto local_2d() const{ static_assert(sizeof(T)%sizeof(double) == 0, "!");
		if(std::get<0>(global_extensions_).size()==0 or std::get<1>(global_extensions_).size()==0) 
			return std::make_pair(ptrdiff_t{0}, multi::extensions_type_<2>{});
		std::array<ptrdiff_t, 2> const sizes_arr = {std::get<0>(global_extensions_).size(), std::get<1>(global_extensions_).size()};
		ptrdiff_t local_n0, local_0_start;
		auto count = fftw_mpi_local_size_many(2, sizes_arr.data(), sizeof(T)/sizeof(double),
		/*std::numeric_limits<ptrdiff_t>::max()*/ sizes_arr[0], comm_.get(),
			&local_n0, &local_0_start
		)/(sizeof(T)/sizeof(double));
		assert( count > 0);
		assert( count >= local_n0*std::get<1>(global_extensions_).size() );
		return std::make_pair(count, multi::extensions_type_<2>{{local_0_start, local_0_start + local_n0}, std::get<1>(global_extensions_)});
	}
public:
	multi::extensions_type_<2> global_extensions() const{return global_extensions_;}
	auto num_elements()                            const{return multi::layout_t<2>(global_extensions_).num_elements();}
	ptrdiff_t                  local_count      () const{return local_2d().first  ;}
	multi::extensions_type_<2> local_extensions () const{return local_2d().second ;}
	bmpi3::communicator&       comm()              const{return comm_;}
	ptrdiff_t                  block()             const{return block_;}
};

template<class T, multi::dimensionality_type D, class Alloc = std::allocator<T>> 
// cannot use fftw::allocator<T> as default because it produces error in nvcc: 
// `template<class _Tp> using __pointer = typename _Tp::pointer’ is protected within this context`
class scattered_array;

template<class T, multi::dimensionality_type D, class Alloc = std::allocator<T>> 
// cannot use fftw::allocator<T> as default because it produces error in nvcc: 
// `template<class _Tp> using __pointer = typename _Tp::pointer’ is protected within this context`
class gathered_array;

template<class T>
struct array_common{
	using element_type = T;
};

template<class T, class Alloc>
class gathered_array<T, dimensionality_type{2}, Alloc> : public gathered_layout<T, 2>, public array_common<T>, public multi::array_ref<T, 2>{
public:
	using local_allocator_type = Alloc;
	using local_pointer_t      = typename std::allocator_traits<local_allocator_type>::pointer;
	using layout_type = gathered_layout<T, 2>;
	using multi::array_ref<T, 2>::num_elements;
private:
	local_allocator_type alloc_;
public:
//	local_pointer_t local_data_; //	typename boost::multi::array_ptr<T, 2, local_pointer_t>   local_ptr_;
	gathered_array(multi::extensions_type_<2> ext, bmpi3::communicator comm = mpi3::environment::self(), Alloc alloc = {}) :
		layout_type(ext, comm),
		multi::array_ref<T, 2>(std::allocator_traits<Alloc>::allocate(alloc, layout_type::local_count()), comm.root()?ext:multi::extensions_type_<2>{}),
		alloc_{alloc}
	//	local_data_{std::allocator_traits<Alloc>::allocate(alloc_, gathered_array::local_count())},
	{
	//	if(not std::is_trivially_default_constructible<typename gathered_array::element_type>{})
	//		adl_alloc_uninitialized_default_construct_n(alloc_, this->data_elements()/*local_ptr_->base()*/, this->num_elements());//local_ptr_->num_elements());
	}
	gathered_array(gathered_array const&) = delete;
	gathered_array(gathered_array&& other) = delete;// : // intel calls this function to return from a function
/*		layout_type{std::exchange(static_cast<layout_type&>(other), layout_type(multi::extensions_type_<2>{}, other.comm()))},
		multi::array_ref<T, 2>(other.data_elements(), layout_type::local_extensions()),
		alloc_      {std::move(other.alloc_)}
	{
		assert(0);
	//	assert(not other.extensions());
		assert(other.local_count() == 0 );
	}*/
	~gathered_array() noexcept{
		std::allocator_traits<local_allocator_type>::deallocate(alloc_, this->data_elements(), layout_type::local_count());
	}
	auto global_layout() const{return static_cast<gathered_layout<T, 2> const&>(*this);}
};

class scoped_barrier{
	mpi3::communicator& comm_;
public:
	scoped_barrier(mpi3::communicator& comm) : comm_{comm}{comm_.barrier();}
	scoped_barrier(scoped_barrier const&) = delete;
	~scoped_barrier(){comm_.barrier();}
};

template<class T, class Alloc>
class scattered_array<T, multi::dimensionality_type{2}, Alloc> : public array_common<T>, public layout_t<T, 2>{
public:
	using local_allocator_type = Alloc;
	using local_pointer_t      = typename std::allocator_traits<local_allocator_type>::pointer;
private:
	using layout_type = layout_t<T, 2>;
	Alloc           alloc_ ;
	local_pointer_t local_data_; //	typename boost::multi::array_ptr<T, 2, local_pointer_t>   local_ptr_;
public:
	scattered_array(multi::extensions_type_<2> ext, bmpi3::communicator comm = mpi3::environment::self(), Alloc alloc = {}) :
		layout_t<T, 2>(ext, comm),
		alloc_{alloc},
		local_data_{std::allocator_traits<Alloc>::allocate(alloc_, scattered_array::local_count())}//,
	{
		if(not std::is_trivially_default_constructible<typename scattered_array::element_type>{})
			adl_alloc_uninitialized_default_construct_n(alloc_, local_cutout().data_elements()/*local_ptr_->base()*/, local_cutout().num_elements());//local_ptr_->num_elements());
	}
	scattered_array(scattered_array const& other) :
		layout_t<T, 2> {other},
		alloc_      {other.alloc_},
		local_data_ {std::allocator_traits<Alloc>::allocate(alloc_, layout_type::local_count())}
	{
		scoped_barrier(other.comm());
		local_cutout() = other.local_cutout();
	/*
		auto p1 = fftw_mpi_plan_many_transpose(
			std::get<0>(this->extensions()).size(), std::get<1>(this->extensions()).size(), sizeof(T)/sizeof(double), 
			other.block(), this->block(),
			reinterpret_cast<double*>(const_cast<T*>(other.local_cutout().data_elements())), 
			reinterpret_cast<double*>(               this->local_cutout().data_elements() ),
			this->comm().get(), FFTW_ESTIMATE
		);
		auto p2 = fftw_mpi_plan_many_transpose(
			std::get<1>(this->extensions()).size(), std::get<0>(this->extensions()).size(), sizeof(T)/sizeof(double), 
			other.block(), this->block(),
			reinterpret_cast<double*>(               this->local_cutout().data_elements()), 
			reinterpret_cast<double*>(               this->local_cutout().data_elements()),
			this->comm().get(), FFTW_ESTIMATE
		);
		fftw_execute(p1);
		fftw_execute(p2);
		fftw_destroy_plan(p2);
		fftw_destroy_plan(p1);
	*/
	}
	scattered_array(scattered_array&& other) : // intel calls this function to return from a function
		layout_type{std::exchange(static_cast<layout_type&>(other), layout_type(multi::extensions_type_<2>{}, other.comm()))},
		alloc_     {std::move(other.alloc_)},
		local_data_{other.local_data_}
	{
		assert(not other.extensions());
		assert(other.local_count() == 0 );
	}
	
	friend std::ostream& operator<<(std::ostream& os, scattered_array const& self){
		for(int r = 0; r != self.comm().size(); ++r){
			if(self.comm().rank() == r){
				if(auto x = self.local_cutout().extensions())
					for(auto i : std::get<0>(x)){
						for(auto j : std::get<1>(x))
							os<< self.local_cutout()[i][j] <<' ';
						os<<std::endl;
					}
			}
			self.comm().barrier();
		}
		return os;
	}

	array_ref <T, 2, local_pointer_t> local_cutout()      &//{return *local_ptr_;}
		{return array_ref <T, 2, local_pointer_t>(local_data_, this->local_extensions());}
	array_cref<T, 2, local_pointer_t> local_cutout() const&//{return *local_ptr_;}
		{return array_cref<T, 2, local_pointer_t>(local_data_, this->local_extensions());}

	local_pointer_t local_data(){return local_data_;}
	typename std::pointer_traits<local_pointer_t>::template rebind<T const> local_data() const{return local_data_;}

	auto extensions() const{return this->global_extensions();}

	operator multi::array<T, 2>() const&{ 
		static_assert( std::is_trivially_copy_assignable<T>{}, "!" );
		multi::array<T, 2> ret(this->global_extensions(), 1., alloc_);
		this->comm().all_gatherv_n(local_data_, local_cutout().num_elements(), ret.data_elements());
		return ret;
	}
	
	mpi::gathered_array<T, 2> gather() const{
		mpi::gathered_array<T, 2> other(this->extensions(), this->comm());
		this->comm_.gatherv_n(local_cutout().data_elements(), local_cutout().num_elements(), other.data_elements());
		static_assert( std::is_trivially_copy_assignable<T>{} and sizeof(T)%sizeof(double)==0, "!");

	/*	{
			fftw_plan p = fftw_mpi_plan_many_transpose(
				std::get<0>(this->extensions()).size(), std::get<1>(this->extensions()).size(), sizeof(T)/sizeof(double), 
				this->block(), std::get<0>(this->extensions()).size(),
				reinterpret_cast<double*>(const_cast<T*>(local_cutout().data_elements())), 
				reinterpret_cast<double*>(ret.data_elements()),
				this->comm().get(), FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN | FFTW_MPI_TRANSPOSED_OUT
			);
			fftw_execute(p);
			fftw_destroy_plan(p);
		}*/
	
		auto p1 = fftw_mpi_plan_many_transpose(
			std::get<0>(this->extensions()).size(), std::get<1>(this->extensions()).size(), 
			sizeof(T)/sizeof(double), 
			FFTW_MPI_DEFAULT_BLOCK, this->size(),
			reinterpret_cast<double*>(const_cast<T*>(this->local_cutout().data_elements())), 
			reinterpret_cast<double*>(               other.data_elements() ),
			this->comm().get(), FFTW_ESTIMATE
		);

		auto p2 = fftw_mpi_plan_many_transpose(
			std::get<1>(this->extensions()).size(), std::get<0>(this->extensions()).size(), 
			sizeof(T)/sizeof(double), 
			other.block(), other.block(),
			reinterpret_cast<double*>(               other.data_elements()), 
			reinterpret_cast<double*>(               other.data_elements()),
			this->comm().get(), FFTW_ESTIMATE
		);
		fftw_execute(p1);
		fftw_execute(p2);
		fftw_destroy_plan(p2);
		fftw_destroy_plan(p1);

		return other;
	}

	explicit scattered_array(multi::array<T, 2> const& other, bmpi3::communicator comm = mpi3::environment::self(), Alloc alloc = {}) :
		scattered_array(other.extensions(), comm, alloc)
	{
		local_cutout() = other.stenciled(std::get<0>(local_cutout().extensions()), std::get<1>(local_cutout().extensions()));
	}
//	bool operator==(array<T, 2> const& other) const&{assert(comm()==other.comm());
//		return comm()&=(local_cutout() == other.local_cutout());
//	}
//	bool operator!=(array<T, 2> const& other) const&{return not(*this==other);}
	ptrdiff_t num_elements() const&{return multi::layout_t<2>(extensions()).num_elements();}
	layout_type layout() const{return *this;}
	~scattered_array() noexcept{if(this->local_count()) alloc_.deallocate(local_data_, this->local_count());}
	
	scattered_array& operator=(scattered_array const& other)&{
		assert(this->comm() == other.comm());
		if(this->extensions() == other.extensions()){
			fftw_plan p = fftw_mpi_plan_many_transpose(
				std::get<0>(this->extensions()).size(), std::get<1>(this->extensions()).size(), sizeof(T)/sizeof(double), 
				other.block(), this->block(),
				reinterpret_cast<double*>(const_cast<T*>(other.local_cutout().data_elements())), 
				reinterpret_cast<double*>(               this->local_cutout().data_elements() ),
				this->comm().get(), FFTW_ESTIMATE
			);
			fftw_execute(p);
			fftw_destroy_plan(p);
		}else assert(0);
		return *this;
	}
#if 0
private:
	typename std::allocator_traits<Alloc>::size_type 
	     local_count_2d    (multi::extensions_type_<2> ext){return local_2d(ext).first; }
	auto local_extension_2d(multi::extensions_type_<2> ext){return local_2d(ext).second;}
public:
	Alloc get_allocator() const{return alloc_;}
	array(bmpi3::communicator comm = mpi3::environment::self(), Alloc alloc = {}) :
		comm_{std::move(comm)},
		alloc_{alloc},
		local_count_{local_count_2d(multi::extensions_type_<2>{})},
		local_ptr_  {alloc_.allocate(local_count_), local_extension_2d(multi::extensions_type_<2>{})},
		n0_{multi::layout_t<2>(multi::extensions_type_<2>{}).size()}
	{}
	bool empty() const{return extensions().num_elements();}
	array_ref <T, 2> local_cutout()      &{return *local_ptr_;}
	array_cref<T, 2> local_cutout() const&{return *local_ptr_;}
	ptrdiff_t        local_count() const&{return  local_count_;}
	auto             local_data() const&{return local_cutout().data_elements();}
	multi::extensions_type_<2> extensions() const&{return {n0_, std::get<1>(local_cutout().extensions())};}
	friend auto extensions(array const& self){return self.extensions();}

	array& operator=(multi::array<T, 2> const& other) &{
		if(other.extensions() == extensions()) local_cutout() = other.stenciled(std::get<0>(local_cutout().extensions()), std::get<1>(local_cutout().extensions()));
		else{
			array tmp{other};
			std::swap(*this, tmp);
		}
		return *this;
	}
	template<class Array, class=std::enable_if_t<not std::is_same<Array, multi::array<T, 2>>{}> >
	array& operator=(Array const& other) &{
		assert( other.extensions() == this->extensions() );
		
		static_assert( std::is_trivially_assignable<T&, T>{}, "!" );
		static_assert( sizeof(T)%sizeof(double)==0, "!" );
		
		auto options = FFTW_ESTIMATE;
		if(other.layout_.is_transposed){
			options |= FFTW_MPI_TRANSPOSED_IN;
			n0_ = std::get<1>(other.extensions()).size();
		}
		
		fftw_plan p = fftw_mpi_plan_many_transpose(
			std::get<0>(extensions()).size(), std::get<1>(extensions()).size(), sizeof(T)/sizeof(double), 
			FFTW_MPI_DEFAULT_BLOCK, other.layout_.block,
			reinterpret_cast<double*>(const_cast<T*>(other.local_cutout().base())), 
			reinterpret_cast<double*>(this->local_cutout().data_elements()),
			this->comm_.get(), options
		);
		fftw_execute(p);
		fftw_destroy_plan(p);
		
		local_ptr_ = array_ptr<T, 2, local_pointer_t>{this->local_cutout().data_elements(), local_extension_2d(other.extensions())};
		return *this;
	}
	bool operator==(multi::array<T, 2> const& other) const&{
		if(other.extensions() != extensions()) return false;
		return comm_&=(local_cutout() == other.stenciled(std::get<0>(local_cutout().extensions()), std::get<1>(local_cutout().extensions())));
	}
	friend bool operator==(multi::array<T, 2> const& other, array const& self){
		return self.operator==(other);
	}
	bool operator==(array<T, 2> const& other) const&{assert(comm_==other.comm_);
		return comm_&=(local_cutout() == other.local_cutout());
	}
	array& operator=(array const& other)&{
		if(other.extensions() == this->extensions() and other.comm_ == other.comm_)
			local_cutout() = other.local_cutout();
		else assert(0);
		return *this;
	}
	basic_array<T, typename std::pointer_traits<local_pointer_t>::template rebind<T const>> transposed() const{
		return basic_array<T, typename std::pointer_traits<local_pointer_t>::template rebind<T const>>{
			layout_t{n0_, true, FFTW_MPI_DEFAULT_BLOCK}, this->local_cutout().layout().transpose(), this->local_cutout().data_elements()
		};
	}

#endif
};

#if 1
boost::multi::fftw::mpi::scattered_array<std::complex<double>, 2>& dft(
	boost::multi::fftw::mpi::scattered_array<std::complex<double>, 2> const& A, 
	boost::multi::fftw::mpi::scattered_array<std::complex<double>, 2>      & B, 
	fftw::sign /*s*/
){
	(void)A;
//	assert( A.extensions() == B.extensions() );
//	assert( A.comm() == B.comm() );
#if 0
	fftw_plan p = fftw_mpi_plan_dft_2d(
		std::get<0>(A.extensions()).size(), std::get<1>(A.extensions()).size(), 
		(fftw_complex *)A.local_cutout().data_elements(), (fftw_complex *)B.local_cutout().data_elements(), 
		A.comm().get(),
		s, FFTW_ESTIMATE
	);
	fftw_execute(p);
	fftw_destroy_plan(p);
#endif
	return B;
}
#endif

#if 0






array_transposed<std::complex<double>, 2>& dft(
	array<std::complex<double>, 2> const& A, 
	array_transposed<std::complex<double>, 2>& B, 
	fftw::sign s
){
// http://www.fftw.org/fftw3_doc/MPI-Plan-Creation.html
//	assert( A.extensions() == B.extensions() );
	assert( A.comm() == B.comm() );
	fftw_plan p = fftw_mpi_plan_dft_2d(
		std::get<0>(A.extensions()).size(), std::get<1>(A.extensions()).size(), 
		(fftw_complex *)A.local_cutout().data_elements(), (fftw_complex *)B.local_cutout().data_elements(), 
		A.comm().get(),
		s, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT
	);
	fftw_execute(p);
	fftw_destroy_plan(p);
	return B;
}

array<std::complex<double>, 2>& dft_forward(array<std::complex<double>, 2> const& A, array<std::complex<double>, 2>& B){
	return dft(A, B, fftw::forward);
}

array<std::complex<double>, 2> dft_forward(array<std::complex<double>,2> const& A){
	array<std::complex<double>, 2> ret(A.extensions()); dft_forward(A, ret); return ret;
}

#endif

}}}}

#endif

#if not __INCLUDE_LEVEL__

#include<boost/mpi3/main_environment.hpp>
#include<boost/mpi3/ostream.hpp>
#include "../fftw.hpp"

namespace mpi3 = boost::mpi3;
namespace multi = boost::multi;

int mpi3::main(int, char*[], mpi3::environment& menv){
	multi::fftw::mpi::environment fenv(menv);

	auto world = menv.world();
	mpi3::ostream os{world};
	
	auto const A = [&]{
		multi::fftw::mpi::scattered_array<double, 2> A({8, 15}, world);
		os<<"global sizes"<< std::get<0>(A.extensions()) <<'x'<< std::get<1>(A.extensions()) <<' '<< A.num_elements() <<std::endl;
		os<< A.local_cutout().extension() <<'x'<< std::get<1>(A.local_cutout().extensions()) <<"\t#="<< A.local_cutout().num_elements() <<" allocated "<< A.local_count() <<std::endl;
		if(auto x = A.local_cutout().extensions())
			for(auto i : std::get<0>(x))
				for(auto j : std::get<1>(x))
					A.local_cutout()[i][j] = i + j;//std::complex<double>(i + j, i + 2*j + 1)/std::abs(std::complex<double>(i + j, i + 2*j + 1));
		return A;
	}();

/*
	multi::fftw::mpi::scattered_array<std::complex<double>, 2> B(A.extensions(), world);
	
	multi::array<std::complex<double>, 2> A2 = A;
	assert( A2 == A );
	
	using multi::fftw::dft_forward;
*/
#if 0
	dft_forward(A , B );
	dft_forward(A2, A2);

	{
		auto x = B.local_cutout().extensions();
		for(auto i : std::get<0>(x))
			for(auto j : std::get<1>(x))
				if(not( std::abs(B.local_cutout()[i][j] - A2[i][j]) < 1e-12 )){
					std::cout<< B.local_cutout()[i][j] - A2[i][j] <<' '<< std::abs(B.local_cutout()[i][j] - A2[i][j]) <<'\n';
				}
	}
	
	multi::fftw::mpi::array_transposed<std::complex<double>, 2> AT(A.extensions(), world);
	os<< "global sizes" << std::get<0>(AT.extensions()) <<'x'<< std::get<1>(AT.extensions()) <<' '<< AT.num_elements() <<std::endl;
	os<< AT.local_cutout().extension() <<'x'<< std::get<1>(AT.local_cutout().extensions()) <<"\t#="<< AT.local_cutout().num_elements() <<" allocated "<< AT.local_count() <<std::endl;

	dft(A, AT, multi::fftw::forward);
	
	if(world.rank() == 0){
		if(auto x = B.local_cutout().extensions()){
			for(auto i : std::get<0>(x)){
				for(auto j : std::get<1>(x))
					std::cout<< B.local_cutout()[i][j] <<' ';
				std::cout<<'\n';
			}
		}
		
		if(auto x = AT.local_cutout().extensions()){
			for(auto i : std::get<0>(x)){
				for(auto j : std::get<1>(x))
					std::cout<< AT.local_cutout()[i][j] <<' ';
				std::cout<<'\n';
			}
		}
	}
#endif
	return 0;
}
#endif
#endif

