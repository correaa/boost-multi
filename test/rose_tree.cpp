// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4-*-
// Copyright 2021 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi incomplete type"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

#include<variant>


template<typename T>
struct rose_tree_v : std::variant<T, std::vector<rose_tree_v<T>>> {
	using std::variant<T, std::vector<rose_tree_v<T>>>::variant;
};

namespace multi = boost::multi;

template<typename T>
struct rose_tree_arr : std::variant<T, multi::array<rose_tree_arr<T>, 1>> {
	using std::variant<T, multi::array<rose_tree_arr<T>, 1>>::variant;
};

template<typename T>
struct rose_tree_arr2 : std::variant<T, multi::array<rose_tree_arr<T>, 2>> {
	using std::variant<T, multi::array<rose_tree_arr<T>, 2>>::variant;
};

BOOST_AUTO_TEST_CASE(rose_tree_vector) {
	rose_tree_v<double> rt;
}

BOOST_AUTO_TEST_CASE(rose_tree_array) {
	rose_tree_arr<double> rt;
}

//BOOST_AUTO_TEST_CASE(rose_tree_array2) {
//	rose_tree_arr2<double> rt = multi::array<rose_tree_arr<double>, 2>{{
//		{1, 2, 3},
//		{4, 5, 6},
//		{7, 8, 9}
//	}};
//}
