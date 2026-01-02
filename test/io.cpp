// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/io.hpp>

#include <boost/core/lightweight_test.hpp>

#include <sstream>
#include <type_traits>  // for is_same_v, is_same

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)

	// matlab 1d
	{
		multi::array<double, 1> arr = {1.0, 2.0, 3.0};
		std::ostringstream      oss;
		multi::detail::print(oss, arr, "\t", "\t", "\a", "\a", 0);
		std::cout << "arr = " << oss.str() << "; end" << std::endl;
	}
	// matlab 2d
	{
		multi::array<double, 2> arr = {
			{1,   3},
			{2, -10},
		};
		std::ostringstream oss;
		multi::detail::print(oss, arr, "\t", "\t", "\a", "\a", 0);
		std::cout << "arr = " << oss.str() << "; end" << std::endl;
	}
	// matlab 3d
	{
		multi::array<double, 3> arr = {
			{
             {1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0},
			 },
			{
             {7.0, 8.0, 9.0},
             {10.0, 11.0, 12.0},
			 },
		};

		std::ostringstream oss;
		multi::detail::print(oss, arr, "\t", "\t", "\a", "\a", 0);
		std::cout << "arr = " << oss.str() << "; end" << std::endl;
	}
	// fortran 2d
	{
		multi::array<double, 2> arr = {
			{1,   3},
			{2, -10},
		};
		std::ostringstream oss;
		multi::detail::print(oss, arr, " ", " ", " ", "\a", 0);
		std::cout << "fortran 2d arr = " << oss.str() << "; end" << std::endl;
	}
	// julia 1d
	{
		multi::array<double, 1> arr = {1, 2, 3};
		std::ostringstream      oss;
		multi::detail::print(oss, arr, "[", ",", "]", "\t", 0);
		std::cout << "julia 1d arr = " << oss.str() << "; end" << std::endl;
	}
	// julia 2d
	{
		multi::array<double, 2> arr = {
			{1,   3},
			{2, -10},
		};
		std::ostringstream oss;
		multi::detail::print(oss, arr, "[", ",", "]", "\a", 0);
		std::cout << "julia 2d arr = " << oss.str() << "; end" << std::endl;
	}
	// numpy 2d
	{
		multi::array<double, 2> arr = {
			{1,   3},
			{2, -10},
		};
		std::ostringstream oss;
		multi::detail::print(oss, arr, "[", " ", "]", " ", 0);
		std::cout << "numpy 2d arr = " << oss.str() << "; end" << std::endl;
	}
	{
		multi::array<double, 1> arr = {1.0, 2.0, 3.0};

		std::cout << "A1D = " << arr << "; no more, no less.\n";
	}
	{
		multi::array<double, 2> arr = {
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0}
		};

		std::cout << "A2D = " << arr << "; no more, no less " << std::endl;
	}
	{
		multi::array<double, 3> arr = {
			{
             {1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0},
			 },
			{
             {7.0, 8.0, 9.0},
             {10.0, 11.0, 12.0},
			 },
		};

		std::cout << "A3D = " << arr << "; no more, no less " << std::endl;
	}
	{
		multi::array<double, 0> arr{5.0};
		std::cout << "A0D = " << arr << "; no more, no less " << std::endl;
	}
	{
		multi::array<double, 4> arr = {
			{
             {
             {1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0},
             },
             {
             {7.0, 8.0, 9.0},
             {10.0, 11.0, 12.0},
             },
			 },
			{
             {
             {1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0},
             },
             {
             {7.0, 8.0, 9.0},
             {10.0, 11.0, 12.0},
             },
			 }
		};

		std::cout << "A4D = " << arr << "; no more, no less " << std::endl;

		std::cout << "A4D.extesion() = " << arr.extension() << std::endl;
	}


	return boost::report_errors();
}
