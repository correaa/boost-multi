// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Bug 3: move-assignment with a non-propagating, unequal allocator must move
// elements into the target's own storage -- it must NOT steal the source buffer,
// because the target's allocator cannot legally free the source allocator's
// memory.  (This is the std::pmr::polymorphic_allocator situation: POCMA == false
// and not always-equal.)  A provenance-tracking allocator detects the wrong-
// allocator deallocation without needing a sanitizer.

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

#include <cstddef>  // for std::size_t
#include <map>      // for std::map
#include <memory>   // for std::allocator
#include <utility>  // for std::move

namespace multi = boost::multi;

namespace {

std::map<void*, int> owner;            // ptr -> id of the allocator that allocated it
int                  wrong_frees = 0;  // count of "freed by an allocator that did not allocate it"

template<class T> struct tracking_allocator {
	using value_type                             = T;
	// non-propagating, like std::pmr::polymorphic_allocator
	using propagate_on_container_move_assignment = std::false_type;

	int id               = 0;
	tracking_allocator() = default;
	explicit tracking_allocator(int idx) : id{idx} {}
	template<class U> tracking_allocator(tracking_allocator<U> const& other) noexcept : id{other.id} {}  // NOLINT(google-explicit-constructor)

	auto allocate(std::size_t n) -> T* {
		auto* ptr  = std::allocator<T>{}.allocate(n);
		owner[ptr] = id;
		return ptr;
	}
	void deallocate(T* ptr, std::size_t n) {
		auto const it = owner.find(static_cast<void*>(ptr));
		if(it != owner.end()) {
			if(it->second != id) {
				++wrong_frees;
			}  // freed by a different allocator than allocated it
			owner.erase(it);
		}
		std::allocator<T>{}.deallocate(ptr, n);
	}
	template<class U> auto operator==(tracking_allocator<U> const& other) const noexcept { return id == other.id; }
	template<class U> auto operator!=(tracking_allocator<U> const& other) const noexcept { return id != other.id; }
};

}  // namespace

auto main() -> int {
	{
		multi::array<int, 2, tracking_allocator<int>> arr_a({2, 2}, 1, tracking_allocator<int>{1});
		multi::array<int, 2, tracking_allocator<int>> arr_b({2, 2}, 9, tracking_allocator<int>{2});

		arr_a = std::move(arr_b);  // POCMA == false and unequal: must move element-wise, must NOT steal arr_b's buffer

		BOOST_TEST( arr_a[0][0] == 9 );  // values transferred
	}  // destructors run here: arr_a must not free arr_b's (id 2) buffer with its own (id 1) allocator

	BOOST_TEST( wrong_frees == 0 );  // Bug 3: fails today (id-1 allocator frees an id-2 buffer)

	return boost::report_errors();
}
