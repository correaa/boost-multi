// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2023 Alfredo A. Correa

//#define BOOST_TEST_MODULE "C++ Unit Tests for Multi array diagonal"  // test title NOLINT(cppcoreguidelines-macro-usage)
#include<boost/test/unit_test.hpp>

#include <multi/array.hpp>

namespace multi = boost::multi;

template<class Array2D>
auto trace_with_indices(Array2D const& A) {
    typename Array2D::element_type sum{0};
    for(auto i : extension(A)) {sum += A[i][i];}
    return sum;
}

template<class Array2D>
auto trace_with_diagonal(Array2D const& A) {
    typename Array2D::element_type sum{0};
    for(auto aii : A.diagonal()) {sum += aii;}
    return sum;
}

BOOST_AUTO_TEST_CASE(trace_test) {
	multi::array<int, 2> A({5, 5}, 0);

    auto [is, js] = extensions(A);
    for(auto i : is) {
        for(auto j : js) {
            A[i][j] = static_cast<int>(10*i + j);
        }
    }

    auto tr = trace_with_diagonal(A);

    BOOST_REQUIRE( tr == 00 + 11 + 22 + 33 + 44 );

    BOOST_REQUIRE( trace_with_diagonal(A) == trace_with_indices(A) );
}
