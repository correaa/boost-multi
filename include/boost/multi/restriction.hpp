// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_RESTRICTION_HPP
#define BOOST_MULTI_RESTRICTION_HPP

#include "boost/multi/detail/layout.hpp"  // IWYU pragma: export
#include "boost/multi/utility.hpp"

#ifdef __NVCC__
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define BOOST_MULTI_HD_LAMBDA(BodY)     \
	{                                   \
		return [=] BOOST_MULTI_HD BodY; \
	}                                   \
	()

namespace boost::multi::detail {

#ifdef __NVCC__
template<class Fun>
struct device : Fun {
	using Fun::operator();
};
#endif

template<class Fun> struct function_system {
	using type = void;
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif

template<class T>
constexpr auto make_restriction(std::initializer_list<T> const& il) {
	return [il](multi::index i0) { return il.begin()[i0]; } ^ multi::extensions(il);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

template<class T>
constexpr auto make_restriction(std::initializer_list<std::initializer_list<T>> const& il) {
	return [il](multi::index i0, multi::index i1) { return il.begin()[i0].begin()[i1]; } ^ multi::extensions(il);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

template<class T>
constexpr auto make_restriction(std::initializer_list<std::initializer_list<std::initializer_list<T>>> const& il) {
	return [il](multi::index i0, multi::index i1, multi::index i2) { return il.begin()[i0].begin()[i1].begin()[i2]; } ^ multi::extensions(il);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<class T, dimensionality_type D>
struct init_list {
	using type = std::initializer_list<typename init_list<T, D - 1>::type>;
};

template<class T>
struct init_list<T, 1> {
	using type = std::initializer_list<T>;
};

template<class T>
struct init_list<T, 0> {
	using type = T;
};

template<class T, dimensionality_type D>
using init_list_t = typename init_list<T, D>::type;

}  // namespace boost::multi::detail

namespace boost::multi {

template<class T, dimensionality_type D>
using restriction_idl = decltype(detail::make_restriction(std::declval<detail::init_list_t<T, D>>()));

template<class T, dimensionality_type D>
class initializer_array : public restriction_idl<T, D> {
	using base_ = restriction_idl<T, D>;
	// detail::init_list_t<T, D> ild_;
 public:
	// cppcheck-suppress noExplicitConstructor ;
	constexpr initializer_array(detail::init_list_t<T, D> ild)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: base_(detail::make_restriction(ild)) {}
};

template<class Proj>
struct bind_transposed_t {
	Proj proj_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

	template<class T1, class T2, class... Ts>
	BOOST_MULTI_HD constexpr auto operator()(T1 ii, T2 jj, Ts... rest) const noexcept -> decltype(auto) /*element*/ { return proj_(jj, ii, rest...); }
};

template<class Proj>
struct bind_front_t {
	multi::index idx_;
	Proj         proj_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

	// bind_front_t(multi::index idx, Proj& proj) : idx_{idx}, proj_{proj} {}
	template<class... Args>
	BOOST_MULTI_HD constexpr auto operator()(Args&&... rest) const noexcept { return proj_(idx_, std::forward<Args>(rest)...); }
};

template<dimensionality_type D, class Proj>
class restriction;

template<dimensionality_type D, class Proj>
	class restriction_iterator {
		typename extensions_t<D>::iterator it_;
		Proj const*                        Pproj_;

		restriction_iterator(typename extensions_t<D>::iterator it, Proj const* Pproj) : it_{it}, Pproj_{Pproj} {}

		template<dimensionality_type, class>
		friend class restriction;

		struct bind_front_t {
			multi::index idx_;
			Proj         proj_;
			
			using system = typename multi::detail::function_system<Proj>::type;

			template<class... Args>
			BOOST_MULTI_HD /*BOOST_MULTI_DEV*/ constexpr auto operator()(Args&&... rest) const noexcept { return proj_(idx_, std::forward<Args>(rest)...); }
		};

	 public:
		constexpr restriction_iterator() = default;  // cppcheck-suppress uninitMemberVar ; partially formed
		// constexpr iterator() {}  // = default;  // NOLINT(hicpp-use-equals-default,modernize-use-equals-default) TODO(correaa) investigate workaround

		restriction_iterator(restriction_iterator const& other) = default;
		restriction_iterator(restriction_iterator&&) noexcept   = default;

		auto operator=(restriction_iterator&&) noexcept -> restriction_iterator& = default;
		auto operator=(restriction_iterator const&) -> restriction_iterator&     = default;

		~restriction_iterator() = default;

		using system = typename multi::detail::function_system<Proj>::type;

		using difference_type = std::ptrdiff_t;
		using value_type      = std::conditional_t<(D != 1), restriction<D - 1, bind_front_t>, decltype(apply_(std::declval<Proj>(), std::declval<typename extensions_t<D>::element>()))>;

		using pointer = void;

		using reference = std::conditional_t<(D != 1), restriction<D - 1, bind_front_t>, decltype(apply_(std::declval<Proj&>(), std::declval<typename extensions_t<D>::element>()))>;

		using iterator_category = std::random_access_iterator_tag;

		constexpr auto operator++() -> auto& {
			++it_;
			return *this;
		}
		constexpr auto operator--() -> auto& {
			--it_;
			return *this;
		}

		constexpr auto operator+=(difference_type dd) -> auto& {
			it_ += dd;
			return *this;
		}
		constexpr auto operator-=(difference_type dd) -> auto& {
			it_ -= dd;
			return *this;
		}

		constexpr auto operator++(int) {
			restriction_iterator ret{*this};
			++(*this);
			return ret;
		}
		constexpr auto operator--(int) {
			restriction_iterator ret{*this};
			--(*this);
			return ret;
		}

		friend constexpr auto operator-(restriction_iterator const& self, restriction_iterator const& other) { return self.it_ - other.it_; }
		friend constexpr auto operator+(restriction_iterator const& self, difference_type n) {
			restriction_iterator ret{self};
			return ret += n;
		}
		friend constexpr auto operator-(restriction_iterator const& self, difference_type n) {
			restriction_iterator ret{self};
			return ret -= n;
		}

		friend constexpr auto operator+(difference_type n, restriction_iterator const& self) { return self + n; }

		friend constexpr auto operator==(restriction_iterator const& self, restriction_iterator const& other) noexcept -> bool { return self.it_ == other.it_; }
		friend constexpr auto operator!=(restriction_iterator const& self, restriction_iterator const& other) noexcept -> bool { return self.it_ != other.it_; }

		friend auto operator<=(restriction_iterator const& self, restriction_iterator const& other) noexcept -> bool { return self.it_ <= other.it_; }
		friend auto operator<(restriction_iterator const& self, restriction_iterator const& other) noexcept -> bool { return self.it_ < other.it_; }
		friend auto operator>(restriction_iterator const& self, restriction_iterator const& other) noexcept -> bool { return self.it_ > other.it_; }
		friend auto operator>=(restriction_iterator const& self, restriction_iterator const& other) noexcept -> bool { return self.it_ > other.it_; }

		BOOST_MULTI_HD constexpr auto operator*() const -> decltype(auto) {
			if constexpr(D != 1) {
				using std::get;
				// auto ll = [idx = get<0>(*it_), proj = proj_](auto... rest) { return proj(idx, rest...); };
				return restriction<D - 1, bind_front_t>(extensions_t<D - 1>((*it_).tail()), bind_front_t{get<0>(*it_), *Pproj_});
			} else {
				using std::get;
				return (*Pproj_)(get<0>(*it_));
			}
		}

		BOOST_MULTI_HD auto operator[](difference_type dd) const { return *((*this) + dd); }  // TODO(correaa) use ra_iterator_facade
	};



template<dimensionality_type D, class Proj>
class restriction : std::conditional_t<std::is_reference_v<Proj>, detail::non_copyable_base, detail::copyable_base> {
	extensions_t<D> xs_;
	Proj            proj_;

	using system = typename multi::detail::function_system<std::decay_t<Proj>>::type;

	template<class Fun, class Tup>
	static BOOST_MULTI_HD constexpr auto std_apply_(Fun&& fun, Tup&& tup) -> decltype(auto) {
		using std::apply;
		return apply(std::forward<Fun>(fun), std::forward<Tup>(tup));
	}

 public:
	static constexpr dimensionality_type dimensionality = D;
	constexpr static dimensionality_type rank_v         = D;

	using difference_type = typename extensions_t<D>::difference_type;
	using index           = typename extensions_t<D>::index;

	BOOST_MULTI_HD constexpr restriction(extensions_t<D> xs, Proj proj) : xs_{xs}, proj_{std::move(proj)} {}

	using element = decltype(std_apply_(std::declval<Proj>(), std::declval<typename extensions_t<D>::element>()));

	using value_type = std::conditional_t<
		(D == 1),
		element,
		array<element, D - 1>>;

 private:
	// struct bind_front_t {
	// 	multi::index idx_;
	// 	Proj         proj_;

	// 	// bind_front_t(multi::index idx, Proj& proj) : idx_{idx}, proj_{proj} {}
	// 	template<class... Args>
	// 	BOOST_MULTI_HD constexpr auto operator()(Args&&... rest) const noexcept { return proj_(idx_, std::forward<Args>(rest)...); }
	// };

	template<class Fun, class... Args>
	static BOOST_MULTI_HD constexpr auto apply_(Fun&& fun, Args&&... args) {
		using std::apply;
		return apply(std::forward<Fun>(fun), std::forward<Args>(args)...);
	}

 public:
	using reference = std::conditional_t<(D != 1), restriction<D - 1, bind_front_t<Proj const&>>, decltype(apply_(std::declval<Proj>(), std::declval<typename extensions_t<D>::element>()))  // (std::declval<index>()))
										 >;

#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)
	template<class... Indices>
	BOOST_MULTI_HD constexpr auto operator[](index idx, Indices... rest) const {
		return operator[](idx)[rest...];
	}
	BOOST_MULTI_HD constexpr auto operator[]() const -> decltype(auto) { return proj_(); }
#endif

	BOOST_MULTI_HD constexpr auto operator[](index idx) && -> decltype(auto) {
		// assert( extension().contains(idx) );
		if constexpr(D != 1) {
			// auto ll = [idx, proj = proj_](auto... rest) { return proj(idx, rest...); };
			// return restriction<D - 1, decltype(ll)>(extensions_t<D - 1>(xs_.base().tail()), ll);
			// return [idx, proj = proj_](auto... rest) noexcept { return proj(idx, rest...); } ^ extensions_t<D - 1>(xs_.base().tail());
			return bind_front_t<Proj>{idx, std::move(proj_)} ^ extensions_t<D - 1>(xs_.base().tail());
		} else {
			return proj_(idx);
		}
	}

	BOOST_MULTI_HD constexpr auto operator[](index idx) const& -> decltype(auto) {
		// assert( extension().contains(idx) );
		if constexpr(D != 1) {
			// auto ll = [idx, proj = proj_](auto... rest) { return proj(idx, rest...); };
			// return restriction<D - 1, decltype(ll)>(extensions_t<D - 1>(xs_.base().tail()), ll);
			// return [idx, proj = proj_](auto... rest) noexcept { return proj(idx, rest...); } ^ extensions_t<D - 1>(xs_.base().tail());
			return bind_front_t<Proj const&>{idx, proj_} ^ extensions_t<D - 1>(xs_.base().tail());
		} else {
			return proj_(idx);
		}
	}

	constexpr auto operator+() const { return multi::array<element, D>(*this); }

	// struct bind_transposed_t {
	// 	Proj proj_;
	// 	template<class T1, class T2, class... Ts>
	// 	BOOST_MULTI_HD constexpr auto operator()(T1 ii, T2 jj, Ts... rest) const noexcept -> element { return proj_(jj, ii, rest...); }
	// };

	BOOST_MULTI_HD constexpr auto transposed() && {
		return bind_transposed_t<Proj>{std::move(proj_)} ^ layout_t<D>(extensions()).transpose().extensions();
		// return [proj = proj_](auto i, auto j, auto... rest) { return proj(j, i, rest...); } ^ layout_t<D>(extensions()).transpose().extensions();
	}

	BOOST_MULTI_HD constexpr auto transposed() const& -> restriction<D, bind_transposed_t<Proj const&>> {
		return bind_transposed_t<Proj const&>{proj_} ^ layout_t<D>(extensions()).transpose().extensions();
		// return [proj = proj_](auto i, auto j, auto... rest) { return proj(j, i, rest...); } ^ layout_t<D>(extensions()).transpose().extensions();
	}

	struct bind_diagonal_t {
		Proj proj_;
		template<class T1, class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(T1 ij, Ts... rest) const noexcept -> element { return proj_(ij, ij, rest...); }
	};

	BOOST_MULTI_HD constexpr auto diagonal() const -> restriction<D - 1, bind_diagonal_t> {
		static_assert(D > 1);
		using std::get;  // needed for C++17
		return bind_diagonal_t{proj_} ^ (std::min(get<0>(sizes()), get<1>(sizes())) * extensions().sub().sub());
		// return [proj = proj_](auto i, auto j, auto... rest) { return proj(j, i, rest...); } ^ layout_t<D>(extensions()).transpose().extensions();
	}

	BOOST_MULTI_HD constexpr auto operator~() && { return std::move(*this).transposed(); }
	BOOST_MULTI_HD constexpr auto operator~() const& { return transposed(); }

	struct bind_repeat_t {
		Proj proj_;
		template<class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(multi::index /*unused*/, Ts... rest) const noexcept -> element { return proj_(rest...); }
	};

	BOOST_MULTI_HD auto repeated(multi::size_t n) const -> restriction<D + 1, bind_repeat_t> {
		return bind_repeat_t{proj_} ^ (n * extensions());
	}

	struct bind_partitioned_t {
		Proj      proj_;
		size_type nn_;
		template<class T1, class T2, class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(T1 ii, T2 jj, Ts... rest) const noexcept -> element { return proj_((ii * nn_) + jj, rest...); }
	};

	BOOST_MULTI_HD constexpr auto partitioned(size_type nn) const noexcept -> restriction<D + 1, bind_partitioned_t> {
		return bind_partitioned_t{proj_, size() / nn} ^ layout_t<D>(extensions()).partition(nn).extensions();
	}

	struct bind_reversed_t {
		Proj      proj_;
		size_type size_m1;
		template<class T1, class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(T1 ii, Ts... rest) const noexcept -> element { return proj_(size_m1 - ii, rest...); }
	};

	BOOST_MULTI_HD constexpr auto reversed() const { return bind_reversed_t{proj_, size() - 1} ^ extensions(); }

	struct bind_rotated_t {
		Proj      proj_;
		size_type size_;
		template<class T1, class T2, class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(T1 ii, Ts... rest) const noexcept { return proj_(rest..., ii); }
	};

	BOOST_MULTI_HD constexpr auto rotated() const { return bind_rotated_t{proj_, size()} ^ extensions(); }

	template<class Proj2>
	struct bind_element_transformed_t {
		Proj  proj_;
		Proj2 proj2_;
		template<class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(Ts... rest) const noexcept -> element { return proj2_(proj_(rest...)); }
	};

	template<class Proj2>
	BOOST_MULTI_HD auto element_transformed(Proj2 proj2) const -> restriction<D, bind_element_transformed_t<Proj2>> {
		return bind_element_transformed_t<Proj2>{proj_, proj2} ^ extensions();
	}

	template<class Proj2>
	class bind_transform_t {
		restriction proj_;
		Proj2       proj2_;
		friend restriction;
		bind_transform_t(restriction proj, Proj2 proj2) : proj_{std::move(proj)}, proj2_{std::move(proj2)} {}

	 public:
		BOOST_MULTI_HD constexpr auto operator()(restriction::index idx) const noexcept { return proj2_(proj_[idx]); }
	};

	template<class Proj2, dimensionality_type One = 1 /*workaround for MSVC*/>
	BOOST_MULTI_HD auto transformed(Proj2 proj2) const -> restriction<1, bind_transform_t<Proj2>> {
		return bind_transform_t<Proj2>{*this, proj2} ^ multi::extensions_t<One>({extension()});
	}

	template<class Cursor, dimensionality_type DD = D>
	class cursor_t {
		Proj const* Pproj_;
		Cursor      cur_;
		friend class restriction;
		explicit constexpr cursor_t(Proj const* Pproj, Cursor cur) : Pproj_{Pproj}, cur_{cur} {}

	 public:
		using difference_type = restriction::difference_type;

		BOOST_MULTI_HD constexpr auto operator[](difference_type n) const -> decltype(auto) {
			if constexpr(DD != 1) {
				auto cur = cur_[n];
				return cursor_t<decltype(cur), DD - 1>{Pproj_, cur};
			} else {
				return apply_(*Pproj_, cur_[n]);
			}
		}
	};

	auto home() const {
		auto cur = extensions().home();
		return cursor_t<decltype(cur), D>{&proj_, cur};
	}

	class iterator {
		typename extensions_t<D>::iterator it_;
		Proj const*                        Pproj_;

		iterator(typename extensions_t<D>::iterator it, Proj const* Pproj) : it_{it}, Pproj_{Pproj} {}

		friend restriction;

		struct bind_front_t {
			multi::index idx_;
			Proj         proj_;
			
			using system = typename multi::detail::function_system<Proj>::type;

			template<class... Args>
			BOOST_MULTI_HD /*BOOST_MULTI_DEV*/ constexpr auto operator()(Args&&... rest) const noexcept { return proj_(idx_, std::forward<Args>(rest)...); }
		};

	 public:
		constexpr iterator() = default;  // cppcheck-suppress uninitMemberVar ; partially formed
		// constexpr iterator() {}  // = default;  // NOLINT(hicpp-use-equals-default,modernize-use-equals-default) TODO(correaa) investigate workaround

		iterator(iterator const& other) = default;
		iterator(iterator&&) noexcept   = default;

		auto operator=(iterator&&) noexcept -> iterator& = default;
		auto operator=(iterator const&) -> iterator&     = default;

		~iterator() = default;

		using system = typename multi::detail::function_system<Proj>::type;

		using difference_type = std::ptrdiff_t;
		using value_type      = std::conditional_t<(D != 1), restriction<D - 1, bind_front_t>, decltype(apply_(std::declval<Proj>(), std::declval<typename extensions_t<D>::element>()))>;

		using pointer = void;

		using reference = std::conditional_t<(D != 1), restriction<D - 1, bind_front_t>, decltype(apply_(std::declval<Proj&>(), std::declval<typename extensions_t<D>::element>()))>;

		using iterator_category = std::random_access_iterator_tag;

		constexpr auto operator++() -> auto& {
			++it_;
			return *this;
		}
		constexpr auto operator--() -> auto& {
			--it_;
			return *this;
		}

		constexpr auto operator+=(difference_type dd) -> auto& {
			it_ += dd;
			return *this;
		}
		constexpr auto operator-=(difference_type dd) -> auto& {
			it_ -= dd;
			return *this;
		}

		constexpr auto operator++(int) -> iterator {
			iterator ret{*this};
			++(*this);
			return ret;
		}
		constexpr auto operator--(int) -> iterator {
			iterator ret{*this};
			--(*this);
			return ret;
		}

		friend constexpr auto operator-(iterator const& self, iterator const& other) { return self.it_ - other.it_; }
		friend constexpr auto operator+(iterator const& self, difference_type n) {
			iterator ret{self};
			return ret += n;
		}
		friend constexpr auto operator-(iterator const& self, difference_type n) {
			iterator ret{self};
			return ret -= n;
		}

		friend constexpr auto operator+(difference_type n, iterator const& self) { return self + n; }

		friend constexpr auto operator==(iterator const& self, iterator const& other) noexcept -> bool { return self.it_ == other.it_; }
		friend constexpr auto operator!=(iterator const& self, iterator const& other) noexcept -> bool { return self.it_ != other.it_; }

		friend auto operator<=(iterator const& self, iterator const& other) noexcept -> bool { return self.it_ <= other.it_; }
		friend auto operator<(iterator const& self, iterator const& other) noexcept -> bool { return self.it_ < other.it_; }
		friend auto operator>(iterator const& self, iterator const& other) noexcept -> bool { return self.it_ > other.it_; }
		friend auto operator>=(iterator const& self, iterator const& other) noexcept -> bool { return self.it_ > other.it_; }

		BOOST_MULTI_HD constexpr auto operator*() const -> decltype(auto) {
			if constexpr(D != 1) {
				using std::get;
				// auto ll = [idx = get<0>(*it_), proj = proj_](auto... rest) { return proj(idx, rest...); };
				return restriction<D - 1, bind_front_t>(extensions_t<D - 1>((*it_).tail()), bind_front_t{get<0>(*it_), *Pproj_});
			} else {
				using std::get;
				return (*Pproj_)(get<0>(*it_));
			}
		}

		BOOST_MULTI_HD auto operator[](difference_type dd) const { return *((*this) + dd); }  // TODO(correaa) use ra_iterator_facade
	};

	constexpr auto begin() const { return iterator{xs_.begin(), &proj_}; }
	constexpr auto end() const { return iterator{xs_.end(), &proj_}; }

	constexpr auto size() const { return xs_.size(); }
	constexpr auto sizes() const { return xs_.sizes(); }

	constexpr auto extension() const { return xs_.extension(); }
	constexpr auto extensions() const { return xs_; }

	constexpr auto front() const { return *begin(); }
	constexpr auto back() const { return *(begin() + (size() - 1)); }

	class elements_t {
		typename extensions_t<D>::elements_t elems_;
		Proj                                 proj_;

		elements_t(typename extensions_t<D>::elements_t elems, Proj proj) : elems_{elems}, proj_{std::move(proj)} {}
		friend class restriction;

	 public:
		BOOST_MULTI_HD constexpr auto operator[](index idx) const -> decltype(auto) {
			using std::apply;
			return apply(proj_, elems_[idx]);
		}

		using difference_type = restriction::difference_type;

		class iterator : ra_iterable<iterator> {
			typename extensions_t<D>::elements_t::iterator it_;
			BOOST_MULTI_NO_UNIQUE_ADDRESS Proj             proj_;

		 public:
			iterator(typename extensions_t<D>::elements_t::iterator it, Proj proj) : it_{it}, proj_{std::move(proj)} {}

			auto operator++() -> auto& {
				++it_;
				return *this;
			}
			auto operator--() -> auto& {
				--it_;
				return *this;
			}

			constexpr auto operator+=(difference_type dd) -> auto& {
				it_ += dd;
				return *this;
			}
			constexpr auto operator-=(difference_type dd) -> auto& {
				it_ -= dd;
				return *this;
			}

			friend constexpr auto operator-(iterator const& self, iterator const& other) { return self.it_ - other.it_; }

			constexpr auto operator*() const -> decltype(auto) {
				using std::apply;
				return apply(proj_, *it_);
			}

			using system = typename detail::function_system<std::decay_t<Proj> >::type;

			using difference_type   = elements_t::difference_type;
			using value_type        = difference_type;
			using pointer           = void;
			using reference         = value_type;
			using iterator_category = std::random_access_iterator_tag;

			friend auto operator==(iterator const& self, iterator const& other) -> bool { return self.it_ == other.it_; }
			friend auto operator!=(iterator const& self, iterator const& other) -> bool { return self.it_ != other.it_; }

			friend auto operator<=(iterator const& self, iterator const& other) -> bool { return self.it_ <= other.it_; }
			friend auto operator<(iterator const& self, iterator const& other) -> bool { return self.it_ < other.it_; }

			BOOST_MULTI_HD constexpr auto operator[](difference_type dd) const { return *((*this) + dd); }  // TODO(correaa) use ra_iterator_facade
		};

		auto begin() const { return iterator{elems_.begin(), proj_}; }
		auto end() const { return iterator{elems_.end(), proj_}; }

		auto size() const { return elems_.size(); }
	};

	constexpr auto elements() const { return elements_t{xs_.elements(), proj_}; }
	constexpr auto num_elements() const { return xs_.num_elements(); }
};

#ifdef __cpp_deduction_guides
template<dimensionality_type D, typename Fun>
restriction(multi::extensions_t<D>, Fun) -> restriction<D, Fun>;

template<typename Fun> restriction(extensions_t<0>, Fun) -> restriction<0, Fun>;
template<typename Fun> restriction(extensions_t<1>, Fun) -> restriction<1, Fun>;
template<typename Fun> restriction(extensions_t<2>, Fun) -> restriction<2, Fun>;
template<typename Fun> restriction(extensions_t<3>, Fun) -> restriction<3, Fun>;
template<typename Fun> restriction(extensions_t<4>, Fun) -> restriction<4, Fun>;
template<typename Fun> restriction(extensions_t<5>, Fun) -> restriction<5, Fun>;
template<typename Fun> restriction(extensions_t<6>, Fun) -> restriction<6, Fun>;
#endif

template<dimensionality_type D, typename F>
auto restricted(F&& fun, extensions_t<D> const& ext) {  // nvc++ has 'restrict' reserved
	return restriction<D, F>(ext, std::forward<F>(fun));
}

template<class F, dimensionality_type D>
BOOST_MULTI_HD constexpr auto operator^(F fun, extensions_t<D> const& xs) {
	return restriction<D, F>(xs, std::move(fun));
}

template<class F, dimensionality_type D>
BOOST_MULTI_HD constexpr auto operator->*(extensions_t<D> const& xs, F fun) {
	return restriction<D, F>(xs, std::move(fun));
}

}  // namespace boost::multi

#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L) && !defined(_MSC_VER)
namespace std::ranges {  // NOLINT(cert-dcl58-cpp) to enable borrowed, nvcc needs namespace
template<class Fun, ::boost::multi::dimensionality_type D>
[[maybe_unused]] constexpr bool enable_borrowed_range<::boost::multi::restriction<D, Fun>> = true;  // NOLINT(misc-definitions-in-headers)
}  // end namespace std::ranges
#endif

#undef BOOST_MULTI_HD

#endif  // BOOST_MULTI_RESTRICTION_HPP
