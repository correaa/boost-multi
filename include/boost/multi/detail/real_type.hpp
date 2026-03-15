// Copyright 2026 Amlal El Mahrouss
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_DETAIL_REAL_TYPE_HPP
#define BOOST_MULTI_DETAIL_REAL_TYPE_HPP

#include <algorithm>
#include <utility>

namespace boost::multi {

/// @brief This container represents a numeric type for the multi library. e.g: float, double, simd128, simd256, etc...
template <typename PrecisionType>
class basic_real_type final {
	PrecisionType pv_{};

public:
  auto size() const {
		return sizeof(PrecisionType);
	}

  auto& get() const {
    return pv_;
  }

  basic_real_type(const PrecisionType& v) : pv_(v) {}

  basic_real_type() = default;
  ~basic_real_type() = default;

  basic_real_type& operator=(const basic_real_type&) = default;
  basic_real_type(const basic_real_type&) = default;

	friend PrecisionType operator/=(basic_real_type& lhs, const PrecisionType& rhs) {
    lhs.pv_ /= rhs;
    return lhs.pv_;
	}

	friend PrecisionType& operator/(basic_real_type& lhs, const PrecisionType& rhs) {
    return lhs.pv_ / rhs;
	}

	friend PrecisionType& operator/=(basic_real_type& lhs, const basic_real_type& rhs) {
    lhs.pv_ /= rhs.pv_;
    return lhs.pv_;
	}

	friend PrecisionType& operator/(basic_real_type& lhs, const basic_real_type& rhs) {
    return lhs.pv_ / rhs.pv_;
	}

  friend PrecisionType& operator*=(basic_real_type& lhs, const basic_real_type& rhs) {
    lhs.pv_ *= rhs.pv_;
    return lhs.pv_;
	}

	friend PrecisionType& operator*(basic_real_type& lhs, const basic_real_type& rhs) {
    return lhs.pv_ * rhs.pv_;
	}

  friend PrecisionType& operator*=(basic_real_type& lhs, const PrecisionType& rhs) {
    lhs.pv_ *= rhs;
    return lhs;
	}

	friend PrecisionType& operator*(basic_real_type& lhs, const PrecisionType& rhs) {
    return lhs.pv_ * rhs;
	}

  friend PrecisionType& operator-=(basic_real_type& lhs, const PrecisionType& rhs) {
    lhs.pv_ -= rhs;
    return lhs;
	}

	friend PrecisionType& operator-(basic_real_type& lhs, const PrecisionType& rhs) {
    return lhs.pv_ - rhs;
	}

  friend PrecisionType& operator+=(basic_real_type& lhs, const PrecisionType& rhs) {
    lhs.pv_ += rhs;
    return lhs;
	}

	friend PrecisionType& operator+(basic_real_type& lhs, const PrecisionType& rhs) {
    return lhs.pv_ + rhs;
	}

	friend bool operator==(const basic_real_type& lhs, const PrecisionType& rhs) {
		return lhs.pv_ == rhs;
	}

	friend bool operator!=(const basic_real_type& lhs, const PrecisionType& rhs) {
		return lhs.pv_ != rhs;
	}

	friend bool operator==(const basic_real_type& lhs, const basic_real_type& rhs) {
		return lhs.pv_ == rhs.pv_;
	}

	friend bool operator!=(const basic_real_type& lhs, const basic_real_type& rhs) {
		return lhs.pv_ != rhs.pv_;
	}

  friend std::ofstream& operator<<(std::ofstream& os, const basic_real_type& ft) {
    os << ft.pv_;
    return os;
  }

  friend std::ostream& operator<<(std::ostream& os, const basic_real_type& ft) {
    os << ft.pv_;
    return os;
  }

};

using float_type = basic_real_type<float>;
using double_type = basic_real_type<double>;

} // end boost::multi

namespace std {

inline auto size(const ::boost::multi::float_type& ft) {
    return ft.size();
}

inline auto size(const ::boost::multi::double_type& dt) {
    return dt.size();
}

}

#endif // BOOST_MULTI_DETAIL_REAL_TYPE_HPP
