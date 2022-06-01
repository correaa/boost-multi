// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2022 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi element transformed"
#include<boost/test/unit_test.hpp>

#include "multi/array.hpp"

#include<complex>

namespace multi = boost::multi;

using complex = std::complex<double>;
constexpr complex I{0, 1};

BOOST_AUTO_TEST_CASE(element_transformed_1D_conj_using_function_reference) {
	multi::array<complex, 1> A = { 1. + 2.*I,  3. +  4.*I};

	constexpr auto conj = static_cast<complex (&)(complex const&)>(std::conj<double>);

	auto const& Ac = A.element_transformed(conj);
	BOOST_REQUIRE( Ac[0] == conj(A[0]) );
	BOOST_REQUIRE( Ac[1] == conj(A[1]) );

	Ac[0] = 5. + 4.*I;  // this unfortunately compiles and it is confusing, this is a defect of std::complex<double>
	BOOST_REQUIRE( Ac[0] == 1. - 2.*I );
}

BOOST_AUTO_TEST_CASE(element_transformed_1D_conj_using_lambda) {
	multi::array<complex, 1> A = { 1. + 2.*I,  3. +  4.*I};

	auto const& Ac = A.element_transformed([](auto const& c) {return std::conj(c);});
	BOOST_REQUIRE( Ac[0] == std::conj(A[0]) );
	BOOST_REQUIRE( Ac[1] == std::conj(A[1]) );

	Ac[0] = 5. + 4.*I;  // this unfortunately compiles and it is confusing, this is a defect of std::complex<double>
	BOOST_REQUIRE( Ac[0] == 1. - 2.*I );
}

BOOST_AUTO_TEST_CASE(element_transformed_1D_conj_using_lambda_with_const_return) {
	multi::array<complex, 1> A = { 1. + 2.*I,  3. +  4.*I};

	auto const& Ac = A.element_transformed([](auto const& c) -> auto const {return std::conj(c);});  // NOLINT(readability-const-return-type) to disable assignment
	BOOST_REQUIRE( Ac[0] == std::conj(A[0]) );
	BOOST_REQUIRE( Ac[1] == std::conj(A[1]) );

//  Ac[0] = 5. + 4.*I;  // doesn't compile due to const return, the element is not assignable anyway
	BOOST_REQUIRE( Ac[0] == 1. - 2.*I );
}

template<typename ComplexRef> struct Conjd;

constexpr struct Conj_t {  // NOLINT(readability-identifier-naming) for testing
	template<class ComplexRef> constexpr auto operator()(ComplexRef&& z) const {return Conjd<decltype(z)>{z};}
	template<class T> constexpr auto operator()(Conjd<T> const&) const = delete;
	template<class T> constexpr auto operator()(Conjd<T> &&) const = delete;
	template<class T> constexpr auto operator()(Conjd<T> &) const = delete;
} Conj;

template<typename ComplexRef>
struct Conjd {  // NOLINT(readability-identifier-naming) for testing
	using decay_type = decltype( + std::declval<ComplexRef>() );

	constexpr operator decay_type() const {return std::conj(c_);}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

	friend constexpr auto operator==(decay_type const& other, Conjd const& self) -> bool {return std::conj(self.c_) == other;}
	friend constexpr auto operator!=(decay_type const& other, Conjd const& self) -> bool {return std::conj(self.c_) != other;}

	friend constexpr auto operator==(Conjd const& self, decay_type const& other) -> bool {return other == std::conj(self.c_);}
	friend constexpr auto operator!=(Conjd const& self, decay_type const& other) -> bool {return other != std::conj(self.c_);}

	friend constexpr auto operator==(Conjd const& self, Conjd const& other) -> bool {return other.c_ == self.c_;}
	friend constexpr auto operator!=(Conjd const& self, Conjd const& other) -> bool {return other.c_ != self.c_;}

	constexpr auto operator=(decay_type const& other) && -> Conjd& {c_ = std::conj(other); return *this;}

 private:
	constexpr explicit Conjd(ComplexRef c) : c_{c} {}
	ComplexRef c_;
	friend decltype(Conj);
};

BOOST_AUTO_TEST_CASE(element_transformed_1D_conj_using_proxy) {
	multi::array<complex, 1> const A = { 1. + 2.*I,  3. +  4.*I};

	auto const& Ac = A.element_transformed(Conj);
	BOOST_REQUIRE( std::conj(A[0]) == Ac[0] );
	BOOST_REQUIRE( std::conj(A[1]) == Ac[1] );

//  Ac[0] = 5. + 4.*I;  // not allowed, compile error, Ac is const
	BOOST_REQUIRE( Ac[0] == 1. - 2.*I );
}

BOOST_AUTO_TEST_CASE(element_transformed_1D_conj_using_mutable_proxy) {
	multi::array<complex, 1> A = { 1. + 2.*I,  3. +  4.*I};

	auto&& Ac = A.element_transformed(Conj);  // NOLINT(readability-const-return-type) to disable assignment

	BOOST_REQUIRE( std::conj(A[0]) == Ac[0] );
	BOOST_REQUIRE( std::conj(A[1]) == Ac[1] );

	Ac[0] = 5. + 4.*I;
	BOOST_REQUIRE( Ac[0] == 5. + 4.*I );
	BOOST_REQUIRE(  A[0] == 5. - 4.*I );
}

BOOST_AUTO_TEST_CASE(transform_ptr_single_value) {
	complex c = 1. + 2.*I;

	constexpr auto conj_ro = [](auto const& z) -> auto const {return std::conj(z);};  // NOLINT(readability-const-return-type,clang-diagnostic-ignored-qualifiers) to prevent assignment

	multi::transform_ptr<complex, decltype(conj_ro)> tp{&c, conj_ro};
	BOOST_REQUIRE( *tp == std::conj(1. + 2.*I) );
}

BOOST_AUTO_TEST_CASE(transform_ptr_1D_array) {
	multi::array<complex, 1> A = { 1. + 2.*I,  3. +  4.*I};

	constexpr auto conj_ro = [](auto const& z) -> auto const {return std::conj(z);};  // NOLINT(readability-const-return-type,clang-diagnostic-ignored-qualifiers) to prevent assignment

	auto const& Ac = A.element_transformed(conj_ro);
	BOOST_REQUIRE( Ac[0] == conj_ro(A[0]) );
	BOOST_REQUIRE( Ac[1] == conj_ro(A[1]) );

//  Ac[0] = 5. + 4.i;  // doesn't compile thanks to the `auto const` in the `conj` def
}
