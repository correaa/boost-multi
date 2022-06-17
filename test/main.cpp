// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2022 Alfredo A. Correa

// C++ Unit Tests for Multi empty main
#include "multi/array.hpp"

#include<cassert>

namespace multi = boost::multi;

auto main() -> int {

	multi::array<double, 2> A({10, 15}, 99.);

	if( A[1][2] != 99. ) {return 1;}

}
