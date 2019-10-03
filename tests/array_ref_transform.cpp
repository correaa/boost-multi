#ifdef COMPILATION_INSTRUCTIONS
$CXX -O3 -std=c++14 -Wall -Wfatal-errors -DBOOST_RESULT_OF_USE_DECLTYPE $0 -o $0.x && $0.x $@ && rm -f $0.x; exit
#endif

#include "../array_ref.hpp"
#include "../array.hpp"

#include<complex>
#include<iostream>
#include<boost/iterator/transform_iterator.hpp>

using std::cout; using std::cerr;
namespace multi = boost::multi;

template<class F, class It>
struct bitransformer{
	It it_;
	F f_;
	bitransformer(It it, F f) : it_{std::move(it)}, f_{std::move(f)}{}
	using difference_type = typename std::iterator_traits<It>::difference_type;
	using value_type = typename std::iterator_traits<It>::value_type;
	using pointer  = typename std::iterator_traits<It>::pointer;
	struct reference{
		typename std::iterator_traits<It>::reference r_;
		F f_;
		reference(typename std::iterator_traits<It>::reference r, F f) : r_{r}, f_{std::move(f)}{}
		operator typename std::iterator_traits<It>::value_type()&&{return f_(r_);}
		template<class T, typename = decltype(*(std::declval<It>()) = std::declval<T>())> 
		reference&& operator=(T&& t)&&{r_ = inverse_function(f_)(std::forward<T>(t)); return std::move(*this);}
	};
	using iterator_category = typename std::iterator_traits<It>::iterator_category;
	reference operator*() const{return {*it_, f_};}
	bitransformer operator+(std::ptrdiff_t n) const{return {it_ + n, f_};}
};

auto neg = [](auto&& x){return -x;};
#if __has_cpp_attribute(maybe_unused)
[[maybe_unused]]
#endif
auto inverse_function(decltype(neg)){return [](auto&& x){return -x;};}

template<class P = std::complex<double>*>
struct indirect_real : std::iterator_traits<typename std::pointer_traits<P>::element_type::value_type*>{
	P impl_;
	indirect_real(P const& p) : impl_{p}{}
    auto operator+(std::ptrdiff_t n) const{return indirect_real{impl_ + n};}
	double& operator*() const{return reinterpret_cast<double(&)[2]>(*impl_)[0];}
};

int main(){

	double const d2D[4][5] {
		{ 0,  1,  2,  3,  4}, 
		{ 5,  6,  7,  8,  9}, 
		{10, 11, 12, 13, 14}, 
		{15, 16, 17, 18, 19}
	};
	double const md2D[4][5] {
		{ -0,  -1,  -2,  -3,  -4}, 
		{ -5,  -6,  -7,  -8,  -9}, 
		{-10, -11, -12, -13, -14}, 
		{-15, -16, -17, -18, -19}
	};
#if __cpp_deduction_guides
	multi::array_ref d2DA({4, 5}, boost::transform_iterator{&d2D[0][0], neg});
	multi::array_ref d2DB({4, 5}, &md2D[0][0]);
#else
	auto d2DA = multi::make_array_ref(
		boost::make_transform_iterator(&d2D[0][0], neg), 
		{4, 5}
	);
	auto d2DB = multi::make_array_ref(&md2D[0][0], {4, 5});
#endif
//	d2DA[0][0] = 4.;
	assert( d2DA == d2DB );

{
#if __cpp_deduction_guides
	double Z[4][5] {
		{ 0,  1,  2,  3,  4}, 
		{ 5,  6,  7,  8,  9}, 
		{10, 11, 12, 13, 14}, 
		{15, 16, 17, 18, 19}
	};
	auto d2DC = multi::make_array_ref(bitransformer<decltype(neg), decltype(&Z[0][0])>{&Z[0][0], neg}, {4, 5});
//	multi::array_ref d2DC{bitransformer<decltype(neg), decltype(&Z[0][0])>{&Z[0][0], neg}, {4, 5}};
	cout<< d2DC[1][1] <<'\n';
	d2DC[1][1] = -66;
	cout<< d2DC[1][1] <<'\n';
	assert( Z[1][1] == 66 );
#endif

	{
		using complex = std::complex<double>;
		multi::array<complex, 2> d2D = {
			{ {0, 3}, {1, 9}, {2, 4},  3,  4}, 
			{  5    , {6, 3}, {7, 5},  8,  9}, 
			{ {1, 4}, {9, 1}, 12    , 13, 14}, 
			{  15   ,  16   , 17    , 18, 19}
		};

		using multi::reinterpret_array_cast;
		auto&& d2Dreal = reinterpret_array_cast<double>(d2D);
		assert( d2Dreal[2][1] == 9. );
		d2Dreal[2][1] = 12.;
		assert( d2D[2][1] == complex(12., 1.) ); 

		auto&& d2DrealT = reinterpret_array_cast<double>(rotated(d2D));
		assert( d2DrealT[2][1] == 7. );

		multi::array<double, 2> d2real_copy = d2Dreal;

		using multi::static_array_cast;
		auto&& d2Dreal2 = static_array_cast<double, indirect_real<>>(d2D);
		assert( d2Dreal2[2][1] == 12. );

		struct indirect_imag{
			std::complex<double>* underlying; using element_type = double;
			indirect_imag(std::complex<double>* underlying) : underlying{underlying}{}
			indirect_imag operator+(std::ptrdiff_t n) const{return {underlying + n};}
			double& operator*() const{return reinterpret_cast<double(&)[2]>(*underlying)[1];}
			operator double*() const{return &(*(*this));}
			using difference_type = std::ptrdiff_t;
			using value_type = double;
			using pointer = double*;
			using reference = double&;
			using iterator_category = std::random_access_iterator_tag;
		};
		auto&& d2imag2 = static_array_cast<double, indirect_imag>(d2D);
		assert( d2imag2[2][1] == 1. );
		double* p = d2imag2.base();
		assert( *p == 3 );

	}
}

}

