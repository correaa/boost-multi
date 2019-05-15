#ifdef COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && c++ -std=c++14 -Wall -Wextra -Wfatal-errors -D_TEST_BOOST_MULTI_DETAIL_GENERIC_ALLOCATOR $0x.cpp -o $0x.x && time $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif
#ifndef BOOST_MULTI_DETAIL_GENERIC_ALLOCATOR_HPP
#define BOOST_MULTI_DETAIL_GENERIC_ALLOCATOR_HPP

#include<cassert>
#include<memory>
#include<experimental/memory_resource>

namespace boost{
namespace multi{

template<class T, class MemoryResource = std::experimental::pmr::memory_resource>
class generic_allocator{
	MemoryResource* mr_;
public:
	using value_type = T;
	using pointer = typename std::pointer_traits<typename MemoryResource::pointer>::template rebind<value_type>;
	using size_type = typename MemoryResource::size_type;
	generic_allocator(MemoryResource* mr) : mr_{mr}{}
	pointer allocate(size_type n){return static_cast<pointer>(mr_->allocate(n*sizeof(value_type)));}
	void deallocate(pointer p, size_type n){mr_->deallocate(p, n*sizeof(value_type));}
	template<class... Args>
	decltype(auto) construct(pointer p, Args&&... args) const{mr_->allocator().construct(p, std::forward<Args>(args)...);}
	decltype(auto) destroy(pointer p)   const{mr_->allocator().destroy(p);}
};

template<class T>
class generic_allocator<T, std::experimental::pmr::memory_resource>{
	std::experimental::pmr::memory_resource* mr_;
public:
	using value_type = T;
	using pointer = typename std::pointer_traits<void*>::template rebind<value_type>;
	using size_type = std::size_t;
	generic_allocator(std::experimental::pmr::memory_resource* mr = std::experimental::pmr::get_default_resource()) : mr_{mr}{}
	pointer allocate(size_type n){return static_cast<pointer>(mr_->allocate(n*sizeof(value_type)));}
	void deallocate(pointer p, size_type n){mr_->deallocate(p, n*sizeof(value_type));}
	template<class... Args>
	void construct(pointer p, Args&&... args) const{::new((void *)p) T(std::forward<Args>(args)...);}
	void destroy(pointer p) const{((T*)p)->~T();}
};

template<class T> 
using allocator = boost::multi::generic_allocator<T, std::experimental::pmr::memory_resource>;

}}

#ifdef _TEST_BOOST_MULTI_DETAIL_GENERIC_ALLOCATOR

#include<vector>

int main(){

	std::vector<double, boost::multi::allocator<double>> v;

}
#endif
#endif

