// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/io.hpp>

#include <boost/core/lightweight_test.hpp>

#include <iostream>
#include <sstream>

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)

	// matlab 1d
	{
		multi::array<double, 1> const arr = {1.0, 2.0, 3.0};

		std::ostringstream oss;
		multi::detail::print(oss, arr, "\t", "\t", "\a", "\a", 0);
		std::cout << "arr = " << oss.str() << "; end\n";
	}
	// matlab 2d
	{
		multi::array<double, 2> const arr = {
			{1,   3},
			{2, -10},
		};

		std::ostringstream oss;
		multi::detail::print(oss, arr, "\t", "\t", "\a", "\a", 0);
		std::cout << "arr = " << oss.str() << "; end\n";
	}
	// matlab 3d
	{
		multi::array<double, 3> const arr = {
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
		std::cout << "arr = " << oss.str() << "; end\n";
	}
	// fortran 2d
	{
		multi::array<double, 2> const arr = {
			{1,   3},
			{2, -10},
		};

		std::ostringstream oss;
		multi::detail::print(oss, arr, " ", " ", " ", "\a", 0);
		std::cout << "fortran 2d arr = " << oss.str() << "; end\n";
	}
	// julia 1d
	{
		multi::array<double, 1> const arr = {1, 2, 3};

		std::ostringstream oss;
		multi::detail::print(oss, arr, "[", ",", "]", "\t", 0);
		std::cout << "julia 1d arr = " << oss.str() << "; end\n";
	}
	// julia 2d
	{
		multi::array<double, 2> const arr = {
			{1,   3},
			{2, -10},
		};

		std::ostringstream oss;
		multi::detail::print(oss, arr, "[", ",", "]", "\a", 0);
		std::cout << "julia 2d arr = " << oss.str() << "; end\n";
	}
	// numpy 2d
	{
		multi::array<double, 2> const arr = {
			{1,   3},
			{2, -10},
		};
		std::ostringstream oss;
		multi::detail::print(oss, arr, "[", " ", "]", " ", 0);
		std::cout << "numpy 2d arr = " << oss.str() << "; end\n";
	}
	{
		multi::array<double, 1> const arr = {1.0, 2.0, 3.0};

		std::cout << "A1D = " << arr << "; no more, no less.\n";
	}
	{
		multi::array<double, 2> const arr = {
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0}
		};

		std::cout << "A2D = " << arr << "; no more, no less\n";
	}
	{
		multi::array<double, 3> const arr = {
			{
             {1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0},
			 },
			{
             {7.0, 8.0, 9.0},
             {10.0, 11.0, 12.0},
			 },
		};

		std::cout << "A3D = " << arr << "; no more, no less\n";
	}
	{
		multi::array<double, 0> const arr{5.0};
		std::cout << "A0D = " << arr << "; no more, no less\n";
	}
	{
		multi::array<double, 4> const arr = {
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

		std::cout << "A4D = " << arr << "; no more, no less " << '\n';

		std::cout << "A4D.extesion() = " << arr.extension() << '\n';
	}

	return boost::report_errors();
}
