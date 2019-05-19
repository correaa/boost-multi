#ifdef COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && clang++ -std=c++17 -Wall -Wextra -Wfatal-errors -D_TEST_BOOST_MULTI_DETAIL_GENERIC_ALLOCATOR $0x.cpp -o $0x.x && time $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif
#ifndef BOOST_MULTI_DETAIL_GENERIC_ALLOCATOR_HPP
#define BOOST_MULTI_DETAIL_GENERIC_ALLOCATOR_HPP

#include<cassert>
#include<memory>
#include<experimental/memory_resource>

namespace boost{
namespace multi{

namespace pmr = std::experimental::pmr;

template<class T, class MemoryResource = pmr::memory_resource>
class generic_allocator{
	using memory_resource_type = MemoryResource;
	memory_resource_type* mr_;
	template<class TT, class MR2> friend class generic_allocator;
public:
	using value_type = T;
	using pointer = typename std::pointer_traits<decltype(std::declval<MemoryResource*>()->allocate(0))>::template rebind<value_type>;
	using difference_type = typename std::pointer_traits<pointer>::difference_type;
	using size_type =  std::make_unsigned_t<difference_type>;
	generic_allocator(memory_resource_type* mr) : mr_{mr}{}
	template<typename T2>
	generic_allocator(generic_allocator<T2, MemoryResource> const& other) : mr_{other.mr_}{}
	bool operator==(generic_allocator const& o) const{return mr_ == o.mr_;}
	bool operator!=(generic_allocator const& o) const{return not(o==*this);}
	pointer allocate(size_type n){
		if(n and !mr_) throw std::bad_alloc{};
		return static_cast<pointer>(mr_->allocate(n*sizeof(value_type)));
	}
	void deallocate(pointer p, size_type n){
		if(n==0 and p == nullptr) return;
		mr_->deallocate(p, n*sizeof(value_type));
	}
	template<class... Args>
	decltype(auto) construct(pointer p, Args&&... args) const{
	//	mr_->allocator().construct(p, std::forward<Args>(args)...);
		std::allocator_traits<std::decay_t<decltype(mr_->allocator())>>::construct(mr_->allocator(), p, std::forward<Args>(args)...);
	}
	decltype(auto) destroy(pointer p) const{
	//	mr_->allocator().destroy(p);
		std::allocator_traits<std::decay_t<decltype(mr_->allocator())>>::destroy(mr_->allocator(), p);
	}
};

template<class T>
class generic_allocator<T, std::experimental::pmr::memory_resource>{
	std::experimental::pmr::memory_resource* mr_;
public:
	using value_type = T;
	using pointer = typename std::pointer_traits<void*>::template rebind<value_type>;
	using size_type = std::size_t;
	generic_allocator(std::experimental::pmr::memory_resource* mr = std::experimental::pmr::get_default_resource()) : mr_{mr}{}
	pointer allocate(size_type n){
		if(n and !mr_) throw std::bad_alloc{};
		return static_cast<pointer>(mr_->allocate(n*sizeof(value_type)));
	}
	void deallocate(pointer p, size_type n){mr_->deallocate(p, n*sizeof(value_type));}
	template<class... Args>
	void construct(pointer p, Args&&... args) const{::new((void *)p) T(std::forward<Args>(args)...);}
	void destroy(pointer p) const{((T*)p)->~T();}
};

template<class T = void> 
using allocator = boost::multi::generic_allocator<T, std::experimental::pmr::memory_resource>;

}}

#ifdef _TEST_BOOST_MULTI_DETAIL_GENERIC_ALLOCATOR

#include<vector>
#include "../array.hpp"

namespace multi = boost::multi;
namespace pmr = std::experimental::pmr;

int main(){

	std::vector<double, multi::allocator<double>> v(100, 1.2);
//	std::vector v(100, 1.2, multi::allocator<double>{}); // needs C++17 CTAD
	multi::array<double, 2, multi::allocator<double> > m({2,4}, 0., pmr::get_default_resource());
//	multi::array m({2,4}, 0., pmr::get_default_resource()); // needs C++17 CTAD
	m[1][3] = 99.;
	assert( m[1][3] == 99. );

}
#endif
#endif

