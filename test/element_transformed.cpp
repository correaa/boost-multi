// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
// Copyright 2022 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi element transformed"
#include<boost/test/unit_test.hpp>

#include "multi/array.hpp"

#include<complex>

// namespace multi = boost::multi;

inline constexpr auto conj_read = [](auto const& c) -> auto const {return std::conj(c);};  // NOLINT(readability-const-return-type,clang-diagnostic-ignored-qualifiers) to prevent assignment

//struct conj_cref;

//struct conj_ref {
//	using base_type = std::complex<double>&;
//	explicit conj_ref(base_type c) : c_{c} {}
//	template<class Source, class = decltype(
//		std::declval<base_type&>() = std::conj(std::declval<Source&&>()))> auto operator=(Source&& v) && -> conj_ref& {
//		std::declval<base_type&>() = std::conj(std::forward<Source>(v)); return *this; }
//	operator std::decay_t<base_type>()     && {return std::conj(c_);}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
//	operator std::decay_t<base_type>()      & {return std::conj(c_);}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
//	operator std::decay_t<base_type>() const& {return std::conj(c_);}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
//	auto operator==(std::complex<double> const& other) const -> bool {return std::conj(c_) == other;}
//	friend struct conj_cref;

// private:
//	base_type c_;
//};

//struct conj_cref {
//	conj_cref(conj_ref&& other) : c_{other.c_} {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
//	explicit conj_cref(std::complex<double> const& c) : c_{c} {}
//	operator std::complex<double>()     && {return std::conj(c_);}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
//	operator std::complex<double>()      & {return std::conj(c_);}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
//	operator std::complex<double>() const& {return std::conj(c_);}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
//	auto operator==(std::complex<double> const& other) const -> bool {return std::conj(c_) == other;}

// private:
//	std::complex<double> const& c_;
//};


//inline constexpr struct conj_rw_t {
//	auto operator()(std::complex<double> const& v) const -> conj_cref {return conj_cref{v};}
//	auto operator()(std::complex<double>      & v) const -> conj_ref  {return conj_ref{v};}
//	auto operator()(std::complex<double>     && v) const -> conj_ref  {return conj_ref{v};}
//} conj_rw;

//constexpr auto conj_rw  = [](std::complex<double>& c) -> conj_ref {return conj_ref{c};};

namespace multi = boost::multi;

//BOOST_AUTO_TEST_CASE(multi_array_sliced_empty_1D_stl) {
//    using namespace std::complex_literals;
//	multi::array A = { 1. + 2.i,  3. +  4.i};
//	auto const& Ac = A.element_transformed(static_cast<std::complex<double>(*)(std::complex<double> const&)>(&std::conj<double>));
//	BOOST_REQUIRE( Ac[0] == conj(A[0]) );
//	BOOST_REQUIRE( Ac[1] == conj(A[1]) );

////  Ac[0] = 5. + 4.i;  // doesn't compile thanks to the `auto const` in the `conj` def
//}

BOOST_AUTO_TEST_CASE(transform_ptr_single) {
	std::complex<double> const I{0, 1};
	std::complex<double> c = 1. + 2.*I;
	multi::transform_ptr<std::complex<double>, decltype(conj_read)> tp{&c, conj_read};
	BOOST_REQUIRE( *tp == std::conj(1. + 2.*I) );
}

BOOST_AUTO_TEST_CASE(multi_array_sliced_empty_1D_read) {
	std::complex<double> const I{0, 1};
	multi::array<std::complex<double>, 1> A = { 1. + 2.*I,  3. +  4.*I};
	BOOST_REQUIRE( A.size() == 2 );

//	auto const& Ac = A.element_transformed(conj_read);
//	BOOST_REQUIRE( Ac[0] == conj(A[0]) );
//	BOOST_REQUIRE( Ac[1] == conj(A[1]) );

//  Ac[0] = 5. + 4.i;  // doesn't compile thanks to the `auto const` in the `conj` def
}

//template<class T> void what(T&&) = delete;

//BOOST_AUTO_TEST_CASE(multi_array_sliced_empty_1D_rw) {
//	using namespace std::complex_literals;
//	multi::array A = { 1. + 2.i,  3. +  4.i};
//	auto Ac = A.element_transformed(conj_rw);
////	what<decltype(Ac)::element_ptr>();
//	BOOST_REQUIRE( Ac[0] == std::conj(A[0]) );
//	BOOST_REQUIRE( Ac[1] == std::conj(A[1]) );

//////  Ac[0] = 5. + 4.i;  // doesn't compile thanks to the `auto const` in the `conj` def
//}



//BOOST_AUTO_TEST_CASE(multi_array_sliced_empty) {
//    using namespace std::complex_literals;
//	multi::array A = {
//		{ 1. + 2.i,  3. +  4.i},
//		{ 8. + 9.i, 10. + 11.i}
//	};
//	auto const& Ac = A.element_transformed(conj);
//}
