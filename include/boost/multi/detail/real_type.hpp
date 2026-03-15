// Copyright 2026 Amlal El Mahrouss
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_DETAIL_REAL_TYPE_HPP
#define BOOST_MULTI_DETAIL_REAL_TYPE_HPP

#include <algorithm>
#include <utility>

#define BOOST_MULTI_DECL_OPERATOR(ID, OP) \
  friend PrecisionType& operator ID (basic_real_type& lhs, const PrecisionType& rhs) { \
lhs.pv_ ID rhs; \
return lhs.pv_; \
} \
 \
friend PrecisionType& operator OP (basic_real_type& lhs, const PrecisionType& rhs) { \
  return lhs.pv_ OP rhs; \
} \
 friend PrecisionType& operator ID (basic_real_type& lhs, const basic_real_type& rhs) { \
   lhs.pv_ ID rhs.pv_;                                                      \
   return lhs.pv_;                                                      \
 }                                                                      \
                                                                        \
 friend PrecisionType& operator OP (basic_real_type& lhs, const basic_real_type& rhs) { \
   return lhs.pv_ OP rhs.pv_;                                               \
 }



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

  public:
    BOOST_MULTI_DECL_OPERATOR(/=, /)
    BOOST_MULTI_DECL_OPERATOR(*=, *)
    BOOST_MULTI_DECL_OPERATOR(+=, +)
    BOOST_MULTI_DECL_OPERATOR(-=, -)

  public:

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


  inline constexpr auto get(const ::boost::multi::float_type& ft) {
    return ft.get();
  }

  inline constexpr auto get(const ::boost::multi::double_type& dt) {
    return dt.get();
  }

} // end boost::multi

namespace std {

  inline constexpr auto size(const ::boost::multi::float_type& ft) {
    return ft.size();
  }

  inline constexpr auto size(const ::boost::multi::double_type& dt) {
    return dt.size();
  }

}

#endif // BOOST_MULTI_DETAIL_REAL_TYPE_HPP
