#ifdef COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && c++ -std=c++14 -Wall -Wextra -Wfatal-errors -D_TEST_BOOST_MULTI_DETAIL_STACK_ALLOCATOR $0x.cpp -o $0x.x && time $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif
#ifndef BOOST_MULTI_DETAIL_STACK_ALLOCATOR_HPP
#define BOOST_MULTI_DETAIL_STACK_ALLOCATOR_HPP

#include "monotonic_allocator.hpp"

#include<stack>

#include<cassert>

namespace boost{
namespace multi{

template<class Alloc = std::allocator<char>, typename std::allocator_traits<Alloc>::size_type MaxAlignemnt = sizeof(std::max_align_t)>
class stack_buffer : private monotonic_buffer<Alloc, MaxAlignemnt>{
	using base_ = monotonic_buffer<Alloc, MaxAlignemnt>;
public:
	using typename base_::pointer;
	using typename base_::void_pointer;
	using typename base_::char_pointer;
	using typename base_::allocator_type;
	using typename base_::size_type;
	using typename base_::difference_type;
private:
	std::stack<size_type> positions_ = {};
	size_type stack_recovered_ = 0;
	size_type max_needed_ = 0;
public:
	using base_::hits;
	using base_::misses;
	using base_::allocated_bytes;
	using base_::deallocated_bytes;
	size_type max_needed() const{return max_needed_;}
	size_type stack_recovered() const{return stack_recovered_;}
	
	using base_::size;
	using base_::allocator;
	stack_buffer(size_type bytes, allocator_type alloc = {}) : monotonic_buffer<Alloc, MaxAlignemnt>{bytes, alloc}{}
	~stack_buffer(){}
	template<size_type RequiredAlignment = sizeof(std::max_align_t)>
	void_pointer allocate(size_type req_bytes, size_type al = RequiredAlignment){
		static_assert( RequiredAlignment <= this->max_alignment, "!");
		assert( al <= this->max_alignment );
		auto bytes = this->align_up(req_bytes);
		if(this->position_ + bytes <= this->size_){
			auto old_position_ = this->position_;
			positions_.push(this->position_);
			this->position_ += bytes;
			++this->hits_;
			this->allocated_bytes_ += bytes;
			max_needed_ = std::max(max_needed_, this->allocated_bytes_ - this->deallocated_bytes_);
			return this->buffer_ + old_position_;
		}
		++this->misses_;
		auto p = allocator_type::allocate(bytes/sizeof(std::max_align_t));
		if(p){
			this->allocated_bytes_ += bytes;
			max_needed_ = std::max(max_needed_, this->allocated_bytes_ - this->deallocated_bytes_);
		}
		return p;
	}
	void deallocate(void_pointer p, size_type req_bytes){
		auto bytes = this->align_up(req_bytes);
		this->deallocated_bytes_ += bytes;
		if(not this->in_buffer(static_cast<char_pointer>(p))){
			allocator_type::deallocate(static_cast<pointer>(p), bytes/sizeof(std::max_align_t));
		}else{
			if(std::distance(static_cast<char_pointer>(static_cast<void_pointer>(this->buffer_)), static_cast<char_pointer>(p)) == static_cast<difference_type>(positions_.top())){
				this->position_ -= bytes;
				positions_.pop();
				stack_recovered_ += bytes;
			}
			#ifndef _BOOST_MULTI_RELAX_STACK_CONDITION
			else{
				assert(0); // throw std::logic_error{"stack violation!"}; // careful
			}
			#endif
		}
	}
};

template<class T = void>
using stack_allocator = multi::generic_allocator<T, multi::stack_buffer<std::allocator<char>>>;

}}

#if _TEST_BOOST_MULTI_DETAIL_STACK_ALLOCATOR

#include "../../multi/array.hpp"

#include<iostream>
#include<vector>

namespace multi = boost::multi;
using std::cout;

int main(){
	cout<<"----------stack"<<std::endl;
	{
		std::size_t guess_bytes = 120;
		for(int i = 0; i != 3; ++i){
			cout<<"pass "<< i << std::endl;
			multi::stack_buffer<std::allocator<char>> buf{guess_bytes};
			{
				multi::array<double, 2, multi::stack_allocator<> > A({2, 10}, &buf);
				multi::array<double, 2, multi::stack_allocator<> > B({3, 10}, &buf);
				multi::array<double, 2, multi::stack_allocator<> > C({4, 10}, &buf);
				std::vector<int, multi::stack_allocator<int>> v(3, &buf);
			//	v.push_back(3); // can produce a runtime error because it is not using buffer as a stack
				multi::array<double, 2, multi::stack_allocator<> > D({4, 10}, &buf);
			}
			cout
				<<"  size: "<< buf.size() 
				<<"\n  hits: "<< buf.hits() 
				<<"\n  misses "<< buf.misses() 
				<<"\n  allocated(bytes) "<< buf.allocated_bytes() 
				<<"\n  deallocated(bytes) "<< buf.deallocated_bytes()
				<<"\n  max_needed(bytes) "<< buf.max_needed()
				<<"\n  stack recovered(bytes) " << buf.stack_recovered()
				<< std::endl
			;
			guess_bytes = std::max(guess_bytes, buf.max_needed());
		}
	}
}
#endif
#endif

