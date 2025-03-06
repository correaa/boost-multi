// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
#include <boost/core/lightweight_test.hpp>

#include <boost/multi/array.hpp>  // for array, layout_t, subarray, range

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
// for serialization of array elements (in this case strings)
#include <boost/serialization/string.hpp>

#include<fstream>  // saving to files in example

using input_archive  = boost::archive::xml_iarchive;
using output_archive = boost::archive::xml_oarchive;

using boost::serialization::make_nvp;

namespace multi = boost::multi;

template<class Element, multi::dimensionality_type D, class IStream> 
auto array_load(IStream&& is) {
	multi::array<Element, D> value;
	auto&& vv = value();
	input_archive{is} >> make_nvp("value", vv);
	return value;
}

template<class Element, multi::dimensionality_type D, class OStream>
void array_save(OStream&& os, multi::array<Element, D> const& value) {
	auto const& vv = value();
	output_archive{os} << make_nvp("value", vv);
}

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)

	multi::array<std::string, 2> const A = {{"w", "x"}, {"y", "z"}};
	array_save(std::ofstream{"file.string2D.json"}, A);  // use std::cout to print serialization to the screen

	auto const B = array_load<std::string, 2>(std::ifstream{"file.string2D.json"});
	BOOST_TEST(A == B);


	return boost::report_errors();
}
