// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2021-2022 Alfredo A. Correa

#ifndef MULTI_TUPLE_ZIP
#define MULTI_TUPLE_ZIP

#include<cassert>
#include<utility>

#include<tuple>

namespace boost::multi {  // NOLINT(modernize-concat-nested-namespaces) keep c++14 compat
namespace detail {

template<class... Ts> class tuple;

template<class T0, class... Ts> class tuple<T0, Ts...> {
	T0 t0_;
	tuple<Ts...> sub_;

 public:
	constexpr explicit tuple(T0 t0, Ts... ts)
	: t0_{std::move(t0)}, sub_{std::move(ts)...} {}
};

template<> class tuple<> {};

//template<class... Ts>
//constexpr auto make_tuple(Ts... ts) -> tuple<std::decay_t<Ts>...> {
//	return tuple<std::decay_t<Ts>...>(std::move(ts)...) }

template<class Tuple1, std::size_t... Indices>
auto tuple_zip_impl(Tuple1&& t1, std::index_sequence<Indices...> /*012*/) {
	return make_tuple(
		make_tuple(
			std::get<Indices>(std::forward<Tuple1>(t1))
		)...
	);
}

template<class Tuple1, class Tuple2, std::size_t... Is>
auto tuple_zip_impl(Tuple1&& t1, Tuple2&& t2, std::index_sequence<Is...> /*012*/) {
	return make_tuple(
		make_tuple(
			std::get<Is>(std::forward<Tuple1>(t1)),
			std::get<Is>(std::forward<Tuple2>(t2))
		)...
	);
}

template<class Tuple1, class Tuple2, class Tuple3, std::size_t... Is>
auto tuple_zip_impl(Tuple1&& t1, Tuple2&& t2, Tuple3&& t3, std::index_sequence<Is...> /*012*/) {
	return	make_tuple(
		make_tuple(
			std::get<Is>(std::forward<Tuple1>(t1)),
			std::get<Is>(std::forward<Tuple2>(t2)),
			std::get<Is>(std::forward<Tuple3>(t3))
		)...
	);
}

template<class Tuple1, class Tuple2, class Tuple3, class Tuple4, std::size_t... Is>
auto tuple_zip_impl(Tuple1&& t1, Tuple2&& t2, Tuple3&& t3, Tuple4&& t4, std::index_sequence<Is...> /*012*/) {
	return make_tuple(
		make_tuple(
			std::get<Is>(std::forward<Tuple1>(t1)),
			std::get<Is>(std::forward<Tuple2>(t2)),
			std::get<Is>(std::forward<Tuple3>(t3)),
			std::get<Is>(std::forward<Tuple4>(t4))
		)...
	);
}

template<class Tuple1, class... Tuples>
auto tuple_zip(Tuple1&& t1, Tuples&&... ts) {
	return detail::tuple_zip_impl(
		std::forward<Tuple1>(t1), std::forward<Tuples>(ts)...,
		std::make_index_sequence<std::tuple_size<typename std::decay<Tuple1>::type>::value>()
	);
}

}  // end namespace detail
}  // end namespace boost::multi

#endif
