#include <iostream>
#include "boost/multi/utility.hpp"

namespace boost::multi {

namespace detail {

template<class Array, std::enable_if_t<!has_dimensionality<Array>::value, int> = 0>
void print(std::ostream& os, Array const& arr, std::string_view /*open*/, std::string_view /*sep*/, std::string_view /*close*/, std::string_view /*tag*/, int /*indent*/) {
	os << arr;
}

template<class Array, std::enable_if_t<has_dimensionality<Array>::value && (Array::dimensionality == 0), int> = 0>
void print(std::ostream& os, Array const& arr, std::string_view /*open*/, std::string_view /*sep*/, std::string_view /*close*/, std::string_view /*tag*/, int /*indent*/) {
    assert(!arr.empty());
    os << static_cast<typename Array::element_cref>(arr);
}

template<class Array, std::enable_if_t<has_dimensionality<Array>::value && (Array::dimensionality > 0), int> = 0>
void print(std::ostream& os, Array const& arr, std::string_view open, std::string_view sep, std::string_view close, std::string_view tab, int indent) {
	for(auto count = 0; count != indent; ++count) {
		os << tab;
	}
	os << open[0];
	if(Array::dimensionality > 1) {
		os << '\n';
	}
	if(arr.size()) {
		multi::detail::print(os, arr.front(), open.size() == 1 ? open : open.substr(1), sep.size() == 1 ? sep : sep.substr(1), close.size() == 1 ? close : close.substr(1), tab.size() == 1 ? tab : tab.substr(1), indent + 1);
		for(auto const& item : arr.dropped(1)) {
			os << sep[0] << ' ';
			if(Array::dimensionality > 1) {
				os << '\n';
			}
			multi::detail::print(os, item, open.size() == 1 ? open : open.substr(1), sep.size() == 1 ? sep : sep.substr(1), close.size() == 1 ? close : close.substr(1),  tab.size() == 1 ? tab : tab.substr(1), indent + 1);
		}
	}
	if(Array::dimensionality > 1) {
		os << sep[0] << ' ' << '\n';
        for(auto count = 0; count != indent; ++count) {
			os << tab;
		}
	}
	os << close[0];
}

}

template<class Array, std::enable_if_t<Array::dimensionality >= 0, int> = 0>
auto operator<<(std::ostream& os, Array const& arr) -> std::ostream& {
	multi::detail::print(os, arr, "{", ",", "}", "\t", 0);
	return os;
}

}  // namespace boost::multi
