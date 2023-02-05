// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2023 Alfredo A. Correa

#ifndef MULTI_ADAPTORS_COMPLEX_HPP
#define MULTI_ADAPTORS_COMPLEX_HPP
#pragma once

namespace boost::multi {

template<class T> struct [[nodiscard]] complex;

template<class T> struct [[nodiscard]] imaginary;

template<class U>
constexpr auto operator+(U real, imaginary<U> imag) -> complex<U>;

template<class T>
struct [[nodiscard]] imaginary {
	T _value;

//	constexpr explicit imaginary(T value) : value_{value} {}
	template<class U>
	friend constexpr complex<U> operator+(U real, imaginary<U> imag);
	constexpr static imaginary const i{T{1}};
	friend constexpr auto operator*(T real, imaginary imag) {
		return imaginary{real*imag._value};
	}
	[[nodiscard]] constexpr auto operator*(imaginary other) const {return -_value*other._value;}
	[[nodiscard]] constexpr auto operator/(imaginary other) const {return  _value/other._value;}
	[[nodiscard]] constexpr auto operator+(imaginary other) const {return imaginary{_value + other._value};}
	[[nodiscard]] constexpr auto operator-(imaginary other) const {return imaginary{_value + other._value};}
};

namespace literals {
// constexpr imaginary<double> operator""_i(unsigned long long d) {
// 	return imaginary<double>{static_cast<double>(d)};
// }

	constexpr auto operator""  _i(long double d) {return imaginary<double>{static_cast<double>(d)};}
//	constexpr auto operator""   i(long double d) {return imaginary<double>{static_cast<double>(d)};}

//  constexpr auto operator"" f_i(long double d) {return imaginary<float >{static_cast<float >(d)};}
	constexpr auto operator""_f_i(long double d) {return imaginary<float >{static_cast<float >(d)};}
	constexpr auto operator"" _if(long double d) {return imaginary<float >{static_cast<float >(d)};}

template<char... chars>
constexpr auto operator""_FI() noexcept {}

}  // namespace literals

template<class T>
struct [[nodiscard]] complex {
	T _real;
	T _imag;

	template<class U>
	friend constexpr complex<U> operator+(U real, imaginary<U> imag);
	[[nodiscard]] friend constexpr T real(complex self) {return self._real;}
	[[nodiscard]] friend constexpr T imag(complex self) {return self._imag;}
};

template<class U>
constexpr complex<U> operator+(U real, imaginary<U> imag) {return {real, imag._value};}

}  // end namespace boost::multi
#endif
