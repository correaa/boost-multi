// Copyright 2026 Amlal El Mahrouss
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_DETAIL_REAL_HPP
#define BOOST_MULTI_DETAIL_REAL_HPP

#include <algorithm>
#include <utility>

namespace boost::multi {

template <typename Precision>
struct basic_real_type final {
	Precision v{};

	const auto size() const {
		return sizeof(v);
	}

    basic_real_type() = default;
    ~basic_real_type() = default;

	friend bool operator==(const basic_real_type& lhs, const Precision& rhs) {
		return lhs.v == rhs;
	}

	friend bool operator==(const basic_real_type& lhs, const basic_real_type& rhs) {
		return lhs.v == rhs.v;
	}
};

using float_type = basic_real_type<double>;
using double_type = basic_real_type<double>;

} // end boost::multi

inline std::ofstream& operator<<(std::ofstream& os, const boost::multi::float_type& ft) {
    os << ft.v;
    return os;
}

inline std::ofstream& operator<<(std::ofstream& os, const boost::multi::double_type& ft) {
    os << ft.v;
    return os;
}

namespace std {

inline auto& size(const ::boost::multi::float_type& ft) {
    return ft.size();
}

inline auto& size(const ::boost::multi::double_type& ft) {
    return ft.size();
}

}

#endif // BOOST_MULTI_DETAIL_REAL_HPP