// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2019-2023 Alfredo A. Correa

#include<boost/test/unit_test.hpp>

#include <multi/array.hpp>

#include<complex>

namespace multi = boost::multi;

// NOLINTBEGIN(fuchsia-default-arguments-calls)  // this is a defect in std::complex, not in the library
BOOST_AUTO_TEST_CASE(complex_conversion_float_to_double) {
	std::complex<float> const cee{1.0, 2.0};

	std::complex<double> const zee = cee;

	static_assert(    multi::is_explicitly_convertible_v<std::complex<float>, std::complex<double>> );
	static_assert(    multi::is_implicitly_convertible_v<std::complex<float>, std::complex<double>> );

	BOOST_TEST(cee.real() == zee.real());

	multi::static_array<std::complex<float>, 1> const CEE1(10, std::complex<float>{});  // NOLINT(fuchsia-default-arguments-calls)
	multi::static_array<std::complex<double>, 1> const ZEE1 = CEE1;
}

BOOST_AUTO_TEST_CASE(complex_conversion_double_to_float) {
	std::complex<double> const zee{1.0, 2.0};

	static_assert(    multi::is_explicitly_convertible_v<std::complex<double>, std::complex<float>>);
	static_assert(not multi::is_implicitly_convertible_v<std::complex<double>, std::complex<float>>);

	std::complex<float> const cee{zee};

	BOOST_TEST(cee.real() == zee.real());

	multi::static_array<std::complex<double>, 1> const ZEE1(10, std::complex<float>{});
	multi::static_array<std::complex<float>, 1> const CEE1{ZEE1};
}

#if 0
BOOST_AUTO_TEST_CASE(complex_assign_from_init_list) {
	std::complex<float> const                  cee(std::complex<double>{});

	multi::array<std::complex<float>, 1> const v1 = {cee, cee, cee};
	multi::array<std::complex<float>, 1> const v2 = multi::array<std::complex<double>, 1>({std::complex<double>{}, std::complex<double>{}, std::complex<double>{}})
		                                                .element_transformed([](auto zee) noexcept { return std::complex<float>(zee); });
	multi::array<std::complex<float>, 1> const v3({std::complex<double>{}, std::complex<double>{}, std::complex<double>{}});
	// multi::array<std::complex<float>, 1> v4 = {std::complex<double>{}, std::complex<double>{}, std::complex<double>{}};

	multi::array<std::complex<float>, 2> const M1 = {
		{std::complex<float>{}, std::complex<float>{}, std::complex<float>{}},
		{std::complex<float>{}, std::complex<float>{}, std::complex<float>{}},
	};

	multi::array<std::complex<float>, 2> const M2({
		{std::complex<float>{}, std::complex<float>{}, std::complex<float>{}},
		{std::complex<float>{}, std::complex<float>{}, std::complex<float>{}},
	});

	// multi::array<std::complex<float>, 2> const M3(
	//  multi::array<std::complex<double>, 2>({
	//      {std::complex<double>{}, std::complex<double>{}, std::complex<double>{}},
	//      {std::complex<double>{}, std::complex<double>{}, std::complex<double>{}},
	//  })
	//      .element_transformed([](auto zee) noexcept {return std::complex<float>(zee);})
	// );

	// multi::array<std::complex<float>, 2> M4({
	//  {std::complex<double>{}, std::complex<double>{}, std::complex<double>{}},
	//  {std::complex<double>{}, std::complex<double>{}, std::complex<double>{}},
	// });
}
#endif
// NOLINTEND(fuchsia-default-arguments-calls)
