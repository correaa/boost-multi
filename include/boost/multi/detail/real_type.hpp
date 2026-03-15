// Copyright 2026 Amlal El Mahrouss
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_DETAIL_REAL_TYPE_HPP
#define BOOST_MULTI_DETAIL_REAL_TYPE_HPP

#include <algorithm>
#include <utility>

/// @brief Implements an operation for ID and OP.
/// @param ID This could be /=.
/// @param Op This could be /.
#define BOOST_MULTI_DECL_OPERATOR(ID, OP)                                                 \
	friend PrecisionType& operator ID(basic_real_type& lhs, const PrecisionType& rhs) {   \
		lhs.pv_ ID rhs;                                                                   \
		return lhs.pv_;                                                                   \
	}                                                                                     \
                                                                                          \
	friend PrecisionType& operator OP(basic_real_type& lhs, const PrecisionType& rhs) {   \
		return lhs.pv_ OP rhs;                                                            \
	}                                                                                     \
	friend PrecisionType& operator ID(const PrecisionType& rhs, basic_real_type& lhs) {   \
		lhs.pv_ ID rhs;                                                                   \
		return lhs.pv_;                                                                   \
	}                                                                                     \
                                                                                          \
	friend PrecisionType& operator OP(const PrecisionType& rhs, basic_real_type& lhs) {   \
		return lhs.pv_ OP rhs;                                                            \
	}                                                                                     \
	friend PrecisionType& operator ID(basic_real_type& lhs, const basic_real_type& rhs) { \
		lhs.pv_ ID rhs.pv_;                                                               \
		return lhs.pv_;                                                                   \
	}                                                                                     \
                                                                                          \
	friend PrecisionType& operator OP(basic_real_type& lhs, const basic_real_type& rhs) { \
		return lhs.pv_ OP rhs.pv_;                                                        \
	}

namespace boost::multi {

/// @brief This container represents a numeric type for the multi library. e.g: float, double, simd128, simd256...
/// @author Amlal El Mahrouss
template<typename PrecisionType>
class basic_real_type final {
	PrecisionType pv_{};

 public:
	auto const width() const {
		return sizeof(PrecisionType);
	}

	auto const& get() const {
		return pv_;
	}

	basic_real_type(PrecisionType const& v) : pv_(v) {}

	basic_real_type()  = default;
	~basic_real_type() = default;

	basic_real_type& operator=(basic_real_type const&) = default;
	basic_real_type(basic_real_type const&)            = default;

 public:
	BOOST_MULTI_DECL_OPERATOR(/=, /)
	BOOST_MULTI_DECL_OPERATOR(*=, *)
	BOOST_MULTI_DECL_OPERATOR(+=, +)
	BOOST_MULTI_DECL_OPERATOR(-=, -)
	BOOST_MULTI_DECL_OPERATOR(<=, <)
	BOOST_MULTI_DECL_OPERATOR(>=, >)

 public:
	friend bool operator==(basic_real_type const& lhs, PrecisionType const& rhs) {
		return lhs.pv_ == rhs;
	}

	friend bool operator!=(basic_real_type const& lhs, PrecisionType const& rhs) {
		return lhs.pv_ != rhs;
	}

	friend bool operator==(basic_real_type const& lhs, basic_real_type const& rhs) {
		return lhs.pv_ == rhs.pv_;
	}

	friend bool operator!=(basic_real_type const& lhs, basic_real_type const& rhs) {
		return lhs.pv_ != rhs.pv_;
	}

	friend std::ofstream& operator<<(std::ofstream& os, basic_real_type const& ft) {
		os << ft.pv_;
		return os;
	}

	friend std::ostream& operator<<(std::ostream& os, basic_real_type const& ft) {
		os << ft.pv_;
		return os;
	}
};

using float_type  = basic_real_type<float>;
using double_type = basic_real_type<double>;

inline constexpr auto get(::boost::multi::float_type const& ft) {
	return ft.get();
}

inline constexpr auto get(::boost::multi::double_type const& dt) {
	return dt.get();
}

}  // namespace boost::multi

namespace std {

inline constexpr auto size(::boost::multi::float_type const& ft) {
	return ft.width();
}

inline constexpr auto size(::boost::multi::double_type const& dt) {
	return dt.width();
}

}  // namespace std

#endif  // BOOST_MULTI_DETAIL_REAL_TYPE_HPP
