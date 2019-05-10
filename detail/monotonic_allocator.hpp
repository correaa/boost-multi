#ifdef COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && c++ -std=c++14 -Wall -Wextra `#-Wfatal-errors` -D_TEST_BOOST_MULTI_DETAIL_MONOTONIC_BUFFER $0x.cpp -o $0x.x && time $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif
#ifndef BOOST_MULTI_DETAIL_MONOTONIC_BUFFER_HPP
#define BOOST_MULTI_DETAIL_MONOTONIC_BUFFER_HPP

#include<cstddef> // max_align_t
#include<memory>

namespace boost{
namespace multi{

template<class Alloc = std::allocator<char>>
struct monotonic_buffer{
	using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<char>;
	using size_type = typename std::allocator_traits<Alloc>::size_type;
	using pointer = typename std::allocator_traits<Alloc>::pointer;
	using void_pointer = typename std::pointer_traits<pointer>::template rebind<void>;
	using difference_type = typename std::allocator_traits<Alloc>::difference_type;
private:
	allocator_type alloc_;
	pointer const  buffer_;
	size_type const size_;
	size_type position_ = 0;
public:
	long saves_ = 0;
	long misses_ = 0;
	size_type allocated_bytes_ = 0;
	size_type released_bytes_ = 0;
	allocator_type& allocator(){return alloc_;}
	monotonic_buffer(size_type bytes, allocator_type alloc = {}) : 
		alloc_{alloc}, buffer_{alloc_.allocate(bytes)}, size_{bytes}{}
	~monotonic_buffer(){alloc_.deallocate(buffer_, size_);}
	void_pointer do_allocate(size_type req_bytes, std::size_t /*alignment*/ = alignof(std::max_align_t)){
		#if 1
		if(position_ + req_bytes < size_){
			auto old_position_ = position_;
			position_ += req_bytes;
			++saves_;
			allocated_bytes_ += req_bytes;
			return buffer_ + old_position_;
		}
		#endif
		#if 0
		using std::align;
		void_pointer p = buffer_ + position_;
		size_type sz = size_ - position_;
		if(align(alignment, req_bytes, p, sz)){
			p = static_cast<pointer>(p) + req_bytes;
			sz -= req_bytes;
			position_ = size_ - sz;
			++saves_;
			allocated_bytes_ += req_bytes;
			return static_cast<pointer>(p);
		}
		#endif
		++misses_;
		auto p = alloc_.allocate(req_bytes);
		if(p) allocated_bytes_ += req_bytes;
		return p;
	}
	void do_deallocate(void_pointer p, size_type bytes){
		released_bytes_ += bytes;
		if(!(std::distance(buffer_, static_cast<pointer>(p)) < static_cast<difference_type>(size_) )) 
			alloc_.deallocate(static_cast<pointer>(p), bytes);
	}
};

template<class T, class MR>
class allocator{
	MR* mr_;
public:
	using value_type = T;
	using pointer = typename std::pointer_traits<typename MR::void_pointer>::template rebind<value_type>;
	using size_type = typename MR::size_type;
	allocator(MR* mr) : mr_{mr}{}
	pointer allocate(size_type n){return static_cast<pointer>(mr_->do_allocate(n*sizeof(value_type), alignof(value_type)));}
	void deallocate(pointer p, size_type n){mr_->do_deallocate(p, n*sizeof(value_type));}
	decltype(auto) construct(pointer p) const{mr_->allocator().construct(p);}
	decltype(auto) destroy(pointer p) const{mr_->allocator().destroy(p);}
};

template<class T = void>
using monotonic_allocator = multi::allocator<T, multi::monotonic_buffer<std::allocator<char>>>;

}}

#if _TEST_BOOST_MULTI_DETAIL_MONOTONIC_BUFFER

#include "../../multi/array.hpp"

#include<iostream>
#include<vector>

namespace multi = boost::multi;
using std::cout;

int main(){

	{
		multi::monotonic_buffer<std::allocator<char>> buf(250*sizeof(double));
		{
			multi::array<double, 2, multi::monotonic_allocator<> > A({10, 10}, &buf);
			multi::array<double, 2, multi::monotonic_allocator<> > B({10, 10}, &buf);
			multi::array<double, 2, multi::monotonic_allocator<> > C({10, 10}, &buf);
		}
		assert( buf.saves_ == 2 );
		assert( buf.misses_ == 1 );
		cout
			<<"saved: "<< buf.saves_ 
			<<"\nmisses "<< buf.misses_ 
			<<"\nallocated(bytes) "<< buf.allocated_bytes_ 
			<<"\nreleased(bytes) "<< buf.released_bytes_ 
			<< std::endl
		;
	}
	cout<<"----------"<<std::endl;
	{
		std::size_t guess = 0;
		for(int i = 0; i != 3; ++i){
			cout<<"pass "<< i << std::endl;
			multi::monotonic_buffer<std::allocator<char>> buf(guess*sizeof(double));
			{
				multi::array<double, 2, multi::monotonic_allocator<> > A({10, 10}, &buf);
				multi::array<double, 2, multi::monotonic_allocator<> > B({10, 10}, &buf);
				multi::array<double, 2, multi::monotonic_allocator<> > C({10, 10}, &buf);
			}
			cout
				<<"  save: "<< buf.saves_ 
				<<"\n  misses "<< buf.misses_ 
				<<"\n  allocated(bytes) "<< buf.allocated_bytes_ 
				<<"\n  released(bytes) "<< buf.released_bytes_ 
				<< std::endl
			;
			guess = std::max(guess, buf.allocated_bytes_);
		}
	}
	cout<<"----------"<<std::endl;
	{
		std::size_t guess_bytes = 120;
		for(int i = 0; i != 3; ++i){
			cout<<"pass "<< i << std::endl;
			multi::monotonic_buffer<std::allocator<char>> buf(guess_bytes*sizeof(double));
			{
				multi::array<double, 2, multi::monotonic_allocator<> > A({10, 10}, &buf);
				for(int i = 0; i != 3; ++i){
					multi::array<double, 2, multi::monotonic_allocator<> > B({10, 10}, &buf);
					std::vector<int, multi::monotonic_allocator<int>> v(3, &buf);
					v.push_back(33); v.push_back(33);
				}
				multi::array<double, 2, multi::monotonic_allocator<> > C({10, 10}, &buf);
			}
			cout
				<<"  saves: "<< buf.saves_ 
				<<"\n  misses "<< buf.misses_ 
				<<"\n  allocated(bytes) "<< buf.allocated_bytes_ 
				<<"\n  released(bytes) "<< buf.released_bytes_ 
				<< std::endl
			;
			guess_bytes = std::max(guess_bytes, buf.allocated_bytes_);
		}
	}
}
#endif
#endif

