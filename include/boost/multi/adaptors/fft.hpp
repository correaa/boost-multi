// Copyright 2020-2024 Alfredo A. Correa

#ifndef BOOST_MULTI_ADAPTORS_FFT_HPP
#define BOOST_MULTI_ADAPTORS_FFT_HPP

#include "../adaptors/fftw.hpp"

#if defined(__CUDA__) || defined(__NVCC__)
#include "../adaptors/cufft.hpp"
#elif defined(__HIPCC__)
#include "../adaptors/hipfft.hpp"
#endif

#define BOOST_MULTI_DECLRETURN_(ExpR) -> decltype(ExpR) {return ExpR;}  // NOLINT(cppcoreguidelines-macro-usage) saves a lot of typing

namespace boost::multi::fft{

	static inline constexpr int forward = static_cast<int>(fftw::forward);
	static inline constexpr int none = static_cast<int>(fftw::none);
	static inline constexpr int backward = static_cast<int>(fftw::backward);

	static_assert( forward != none && none != backward && backward != forward );

	template<std::size_t I> struct priority : std::conditional_t<I==0, std::true_type, struct priority<I-1>>{};

	template<class... Args> auto dft_aux(priority<0> /*unused*/, Args&&... args) BOOST_MULTI_DECLRETURN_(                  fftw::dft_backward(std::forward<Args>(args)...))
	template<class... Args> auto dft_aux(priority<1> /*unused*/, Args&&... args) BOOST_MULTI_DECLRETURN_(::boost::multi::cufft ::dft_backward(std::forward<Args>(args)...))
	template<class... Args> auto dft(Args&&... args) BOOST_MULTI_DECLRETURN_(dft_backward_aux_(priority<1>{}, std::forward<Args>(args)...))
	template<class In, class... Args> auto dft(std::array<bool, std::decay_t<In>::dimensionality> which, In const& in, Args&&... args) -> decltype(auto) {return dft_aux(priority<1>{}, which, in, std::forward<Args>(args)...);}

	template<class... Args> auto dft_forward_aux(priority<0> /*unused*/, Args&&... args) BOOST_MULTI_DECLRETURN_(  fftw::dft_forward(std::forward<Args>(args)...))
	template<class... Args> auto dft_forward_aux(priority<1> /*unused*/, Args&&... args) BOOST_MULTI_DECLRETURN_(cufft ::dft_forward(std::forward<Args>(args)...))
	template<class In, class... Args> auto dft_forward(std::array<bool, In::dimensionality> which, In const& in, Args&&... args) -> decltype(auto) {return dft_forward_aux(priority<1>{}, which, in, std::forward<Args>(args)...);}

	template<class... Args> auto dft_backward_aux(priority<0> /*unused*/, Args&&... args) BOOST_MULTI_DECLRETURN_(  fftw::dft_backward(std::forward<Args>(args)...))
	template<class... Args> auto dft_backward_aux(priority<1> /*unused*/, Args&&... args) BOOST_MULTI_DECLRETURN_(cufft ::dft_backward(std::forward<Args>(args)...))
	template<class In, class... Args> auto dft_backward(std::array<bool, In::dimensionality> which, In const& in, Args&&... args) -> decltype(auto) {return dft_backward_aux(priority<1>{}, which, in, std::forward<Args>(args)...);}

	template<class In, class Direction>
	class DFT_range {
	 public:
		static constexpr auto dimensionality = std::decay_t<In>::dimensionality;

	 private:
		std::array<bool, dimensionality> which_;
		In const& in_;
		Direction dir_;

		struct const_iterator : private In::const_iterator {
			bool do_;
			std::array<bool, dimensionality - 1> sub_which_;
			Direction dir_;

			const_iterator(
				typename In::const_iterator it,
				bool doo, std::array<bool, In::dimensionality - 1> sub_which,
				Direction dir
			) : In::const_iterator{it}, do_{doo}, sub_which_{sub_which}, dir_{dir} {}
			// using In::const_iterator::operator-;
			friend auto operator-(const_iterator const& lhs, const_iterator const& rhs) {
				return static_cast<typename In::const_iterator>(lhs) - static_cast<typename In::const_iterator>(rhs);
			}

			using typename In::const_iterator::difference_type;
			using typename In::const_iterator::value_type;
			using pointer = void*;
			using reference = DFT_range<typename In::const_iterator::reference, Direction>;
			using iterator_category = std::random_access_iterator_tag;

			auto operator*() const -> reference {
				return DFT_range<reference, Direction>(sub_which_, *static_cast<typename In::const_iterator>(*this), dir_);
			}

			template<class It>
			friend auto copy(const_iterator const& first, const_iterator const& last, It const& first_d) -> decltype(auto) {
				auto const n = last - first;
				dft(
					std::apply([doo = first.do_](auto... es) { return std::array<bool, dimensionality + 1>{doo, es...}; }, first.sub_which_),
					const_subarray(static_cast<typename In::const_iterator>(first), static_cast<typename In::const_iterator>(last)),
					subarray(first_d, first_d + n)
				);
				return first_d + n;
			}
		};

	 public:
		DFT_range(std::array<bool, std::decay_t<In>::dimensionality> which, In const& in, Direction dir) : which_{which}, in_(in), dir_{dir} {}
		auto begin() const { return const_iterator(in_.begin(), which_[0], std::apply([](auto /*e0*/, auto... es) { return std::array<bool, dimensionality - 1>{es...}; }, which_), dir_); }
		auto end  () const { return const_iterator(in_.end  (), which_[0], std::apply([](auto /*e0*/, auto... es) { return std::array<bool, dimensionality - 1>{es...}; }, which_), dir_); }

		auto extensions() const { return in_.extensions(); }
	};

	template<class In, class Direction>
	auto DFT(std::array<bool, In::dimensionality> which, In const& in, Direction dir) {
		return DFT_range(which, in, dir);
	}

}  // end namespace boost::multi::fft

#undef BOOST_MULTI_DECLRETURN_

#endif  // BOOST_MULTI_ADAPTORS_FFT_HPP
