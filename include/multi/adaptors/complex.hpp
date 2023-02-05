// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2023 Alfredo A. Correa

#ifndef MULTI_ADAPTORS_COMPLEX_HPP
#define MULTI_ADAPTORS_COMPLEX_HPP
#pragma once

#include<complex>  // to define its traits

namespace boost::multi {

template<class T> struct [[nodiscard]] complex;

template<class T> struct [[nodiscard]] imaginary;

template<class U>
constexpr auto operator+(U real, imaginary<U> imag) -> complex<U>;

template<class T>
struct [[nodiscard]] imaginary {
	T _value;  // NOLINT(misc-non-private-member-variables-in-classes) I want the class to be an aggregate

	//	constexpr explicit imaginary(T value) : value_{value} {}
	template<class U>
	friend constexpr auto operator+(U real, imaginary<U> imag) -> complex<U>;
	//  constexpr static imaginary i{T{1}};  // NOLINT(clang-diagnostic-error) "constexpr variable cannot have non-literal type"?
	friend constexpr auto operator*(T real, imaginary imag) {
		return imaginary{real * imag._value};
	}
	[[nodiscard]] constexpr auto operator*(imaginary other) const { return -_value * other._value; }
	[[nodiscard]] constexpr auto operator/(imaginary other) const { return _value / other._value; }
	[[nodiscard]] constexpr auto operator+(imaginary other) const { return imaginary{_value + other._value}; }
	[[nodiscard]] constexpr auto operator-(imaginary other) const { return imaginary{_value + other._value}; }
};

template<>
struct [[nodiscard]] imaginary<void> {
	template<class T>
	friend constexpr auto operator*(T real, imaginary /*self*/) { return imaginary<T>{real}; }
	template<class T>
	[[nodiscard]] constexpr auto operator*(imaginary<T> other) const { return -other._value; }
	template<class T>
	[[nodiscard]] constexpr auto operator/(imaginary<T> other) const { return T{1} / other._value; }
};

constexpr imaginary<void> I{};  // NOLINT(readability-identifier-length) imaginary unit

namespace literals {
// constexpr imaginary<double> operator""_i(unsigned long long d) {
// 	return imaginary<double>{static_cast<double>(d)};
// }

constexpr auto operator"" _i(long double value) { return imaginary<double>{static_cast<double>(value)}; }
//	constexpr auto operator""   i(long double value) {return imaginary<double>{static_cast<double>(value)};}
constexpr auto operator"" _I(long double value) { return imaginary<double>{static_cast<double>(value)}; }

//  constexpr auto operator"" f_i(long double value) {return imaginary<float >{static_cast<float >(value)};}
constexpr auto operator""_f_i(long double value) { return imaginary<float>{static_cast<float>(value)}; }
constexpr auto operator"" _if(long double value) { return imaginary<float>{static_cast<float>(value)}; }
constexpr auto operator""_F_I(long double value) { return imaginary<float>{static_cast<float>(value)}; }
constexpr auto operator"" _IF(long double value) { return imaginary<float>{static_cast<float>(value)}; }

template<char... Chars>
constexpr auto operator""_FI() noexcept {}

}  // namespace literals

template<class T>
struct [[nodiscard]] complex {
	using real_type = T;

	real_type _real;  // NOLINT(misc-non-private-member-variables-in-classes) complex should be an aggregate class
	real_type _imag;  // NOLINT(misc-non-private-member-variables-in-classes) complex should be an aggregate class

	template<class U>
	friend constexpr auto operator+(U real, imaginary<U> imag) -> complex<U>;
	friend constexpr auto operator*(T scale, complex self) { return complex{scale * self._real, scale * self._imag}; }
	friend constexpr auto operator+(T real, complex self) { return complex{real + self._real, self._imag}; }
	friend constexpr auto operator-(T real, complex self) { return complex{real - self._real, self._imag}; }

	[[nodiscard]] friend constexpr auto real(complex self) -> T { return self._real; }
	[[nodiscard]] friend constexpr auto imag(complex self) -> T { return self._imag; }
	friend constexpr auto               conj(complex self) { return complex{self._real, -self._imag}; }

	auto operator==(complex const& other) const {return _real == other._real and _imag == other._imag;}
	auto operator!=(complex const& other) const {return _real != other._real or  _imag != other._imag;}

	[[nodiscard]] constexpr auto real() const -> T { return _real; }
	[[nodiscard]] constexpr auto imag() const -> T { return _imag; }
};

template<class U>
constexpr auto operator+(U real, imaginary<U> imag) -> complex<U> { return {real, imag._value}; }

template<
	class Complex,
	class RealType = typename Complex::real_type,
	class IU = decltype(Complex{0, 1})
>
struct complex_traits {
	using real_type = typename Complex::real_type;
	constexpr static auto imaginary_unit() { return Complex{0, 1}; }
};

template<class T>
struct complex_traits<
	std::complex<T>, void, void
> {
	using real_type = typename std::complex<T>::value_type;
	constexpr static auto imaginary_unit() { return std::complex<T>{0, 1}; }
};

}  // end namespace boost::multi
#endif
