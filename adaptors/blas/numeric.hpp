#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&clang++ -std=c++14 -D_TEST_MULTI_ADAPTORS_BLAS_NUMERIC $0.cpp -o $0x \
`pkg-config --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// Alfredo A. Correa 2019 ©

#ifndef MULTI_ADAPTORS_BLAS_NUMERIC_HPP
#define MULTI_ADAPTORS_BLAS_NUMERIC_HPP

//#include "../blas/core.hpp"
#include "../../array_ref.hpp"

#include<functional> // negate
#include<complex> // conj

namespace boost{
namespace multi{namespace blas{

template<class A, typename C = typename std::decay_t<A>::element_type, typename T = typename C::value_type>
decltype(auto) real(A&& a){
	struct Complex_{T real_; T imag_;};
	auto&& Acast = multi::reinterpret_array_cast<Complex_>(a);
	return multi::member_array_cast<T>(Acast, &Complex_::real_);
}

template<class A, typename Complex = typename std::decay_t<A>::element_type, typename T = typename Complex::value_type>
decltype(auto) imag(A&& a){
	struct Complex_{T real_; T imag_;};
	auto&& Acast = multi::reinterpret_array_cast<Complex_>(a);
	return multi::member_array_cast<T>(Acast, &Complex_::imag_);
}

template<class It, class F> class involuter;

template<class Ref, class Involution>
class involuted{
protected:
	Ref r_; // [[no_unique_address]] 
	Involution f_;
public:
	using decay_type =std::decay_t<decltype(std::declval<Involution>()(std::declval<Ref>()))>;
	explicit involuted(Ref r, Involution f = {}) : r_{std::forward<Ref>(r)}, f_{f}{}
	involuted& operator=(involuted const& other)=delete;//{r_ = other.r_; return *this;}
public:
	involuted(involuted const&) = delete;
	involuted(involuted&&) = default; // for C++14
	operator decay_type() const&{return f_(r_);}
	decltype(auto) operator&()&&{return involuter<decltype(&std::declval<Ref>()), Involution>{&r_, f_};}
//	template<class DecayType>
//	auto operator=(DecayType&& other)&&
//	->decltype(r_=f_(std::forward<DecayType>(other)), *this){
//		return r_=f_(std::forward<DecayType>(other)), *this;}
	template<class DecayType>
	auto operator=(DecayType&& other)&
	->decltype(r_=f_(std::forward<DecayType>(other)), *this){
		return r_=f_(std::forward<DecayType>(other)), *this;}
//	template<class OtherRef>
//	auto operator=(involuted<OtherRef, Involution> const& o)&
//	->decltype(r_=f_==o.f_?std::forward<decltype(o.r_)>(o.r_):f_(o), *this){
//		return r_=f_==o.f_?std::forward<decltype(o.r_)>(o.r_):f_(o), *this;}
	template<class DecayType>
	auto operator==(DecayType&& other) const
	->decltype(this->operator decay_type()==other){
		return this->operator decay_type()==other;}
	template<class DecayType, class Involuted /**/>
	friend auto operator==(DecayType&& other, Involuted const& self)
	->decltype(other == self.operator decay_type()){
		return other == self.operator decay_type();}

	template<class Any> friend auto operator<<(Any&& a, involuted const& self)->decltype(a << std::declval<decay_type>()){return a << self.operator decay_type();}
};

#if __cpp_deduction_guides
template<class T, class F> involuted(T&&, F)->involuted<T const, F>;
//template<class T, class F> involuted(T&, F)->involuted<T&, F>;
//template<class T, class F> involuted(T const&, F)->involuted<T const&, F>;
#endif

template<class It, class F>
class involuter : public std::iterator_traits<It>{
	It it_; // [[no_unique_address]] 
	F f_;
public:
	explicit involuter(It it, F f = {}) : it_{std::move(it)}, f_{std::move(f)}{}
	involuter(involuter const& other) = default;
	using reference = involuted<typename std::iterator_traits<It>::reference, F>;
	auto operator*() const{return reference{*it_, f_};}
	bool operator==(involuter const& o) const{return it_==o.it_;}
	bool operator!=(involuter const& o) const{return it_!=o.it_;}
	involuter& operator+=(typename involuter::difference_type n){it_+=n; return *this;}
	auto operator+(typename involuter::difference_type n) const{return involuter{it_+n, f_};}
	decltype(auto) operator->() const{
		return involuter<typename std::iterator_traits<It>::pointer, F>{&*it_, f_};
	}
	auto operator-(involuter const& other) const{return it_-other.it_;}
	explicit operator bool() const{return it_;}
	friend It underlying(involuter const& self){return self.it_;}
	operator It() const{return underlying(*this);}
};

template<class It2, class F>
class involuter<involuter<It2, F>, F> : public std::iterator_traits<involuter<It2, F>>{
	using It = involuter<It2, F>;
	It it_; // [[no_unique_address]] 
	F f_;
public:
	explicit involuter(It it, F f = {}) : it_{std::move(it)}, f_{std::move(f)}{}
	involuter(involuter const& other) = default;
	using reference = involuted<typename std::iterator_traits<It>::reference, F>;
	auto operator*() const{return reference{*it_, f_};}
	bool operator==(involuter const& o) const{return it_==o.it_;}
	bool operator!=(involuter const& o) const{return it_!=o.it_;}
	involuter& operator+=(typename involuter::difference_type n){it_+=n; return *this;}
	auto operator+(typename involuter::difference_type n) const{return involuter{it_+n, f_};}
	decltype(auto) operator->() const{
		return involuter<typename std::iterator_traits<It>::pointer, F>{&*it_, f_};
	}
	auto operator-(involuter const& other) const{return it_-other.it_;}
	explicit operator bool() const{return it_;}
	friend It underlying(involuter const& self){return self.it_;}
	operator It() const{return underlying(*this);}
	operator It2() const{return underlying(underlying(*this));}
};


template<class Ref> using negated = involuted<Ref, std::negate<>>;
template<class It>  using negater = involuter<It, std::negate<>>;

struct conjugate{
	template<class T>
	auto operator()(T const& a) const{using std::conj; return conj(a);}
};

namespace detail{
template<class Ref> using conjugated = involuted<Ref, conjugate>;
template<class It>  using conjugater = involuter<It, conjugate>;

template<class It> conjugater<It> make_conjugater(It it){return {it};}
template<class It> It make_conjugater(conjugater<It> it){return underlying(it);}

}

#if 0
constexpr auto conj = [](auto const& a){using std::conj; return conj(std::forward<decltype(a)>(a));};

template<class ComplexRef> struct conjd : involuted<ComplexRef, decltype(conj)>{
	conjd(ComplexRef r) : involuted<ComplexRef, decltype(conj)>(r){}
	decltype(auto) real() const{return this->r_.real();}
	decltype(auto) imag() const{return negated<decltype(this->r_.imag())>(this->r_.imag());}//-this->r_.imag();}//negated<std::decay_t<decltype(this->r_.imag())>>(this->r_.imag());} 
	friend decltype(auto) real(conjd const& self){using std::real; return real(static_cast<typename conjd::decay_type>(self));}
	friend decltype(auto) imag(conjd const& self){using std::imag; return imag(static_cast<typename conjd::decay_type>(self));}
};
template<class T> conjd(T&&)->conjd<T>;

template<class Complex> using conjr = involuter<Complex, decltype(conj)>;

#endif

}


}}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#if _TEST_MULTI_ADAPTORS_BLAS_NUMERIC

#include "../blas/gemm.hpp"

#include "../../array.hpp"
#include "../../utility.hpp"

#include <boost/timer/timer.hpp>

#include<complex>
#include<cassert>
#include<iostream>
#include<numeric>
#include<algorithm>

using std::cout;

namespace multi = boost::multi;

template<class M> decltype(auto) print(M const& C){
	using boost::multi::size;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j) cout << C[i][j] << ' ';
		cout << std::endl;
	}
	return cout << std::endl;
}

template<class... T> void what(T&&...);

int main(){
	using multi::blas::gemm;
	using multi::blas::herk;
	using complex = std::complex<double>;
	constexpr auto const I = complex{0., 1.};

	multi::array<double, 2> A = {
		{1., 3., 4.}, 
		{9., 7., 1.}
	};
	multi::array<complex, 2> Acomplex = A;
	multi::array<complex, 2> B = {
		{1. - 3.*I, 6.  + 2.*I},
		{8. + 2.*I, 2. + 4.*I},
		{2. - 1.*I, 1. + 1.*I}
	};
	multi::array<double, 2> Breal = {
		{1., 6.},
		{8., 2.},
		{2., 1.}
	};
	multi::array<double, 2> Bimag = {
		{-3., +2.},
		{+2., +4.},
		{-1., +1.}
	};
	using multi::blas::real;
	using multi::blas::imag;

	assert( Breal == real(B) );
	assert( real(B) == Breal );
	assert( imag(B) == Bimag );

	multi::array_ref<double, 2> rB(reinterpret_cast<double*>(data_elements(B)), {size(B), 2*size(*begin(B))});

	auto&& Bconj = multi::static_array_cast<complex, multi::blas::detail::conjugater<complex*>>(B);
	assert( size(Bconj) == size(B) );
	assert( conj(B[1][2]) == Bconj[1][2] );

//	auto&& BH = multi::blas::hermitized(B);
//	assert( BH[1][2] == conj(B[2][1]) );
//	std::cout << BH[1][2] << " " << B[2][1] << std::endl;

	auto&& BH1 = multi::static_array_cast<complex, multi::blas::detail::conjugater<complex*>>(rotated(B));
	auto&& BH2 = rotated(multi::static_array_cast<complex, multi::blas::detail::conjugater<complex*>>(B));

//	what( BH1, BH2 );
//	using multi::blas::imag;

//	assert( real(A)[1][2] == 1. );
//	assert( imag(A)[1][2] == -3. );

//	print(A) <<"--\n";
//	print(real(A)) <<"--\n";
//	print(imag(A)) <<"--\n";

	multi::array<complex, 2> C({2, 2});
	multi::array_ref<double, 2> rC(reinterpret_cast<double*>(data_elements(C)), {size(C), 2*size(*begin(C))});

	
//	gemm('T', 'T', 1., A, B, 0., C);
//	gemm('T', 'T', 1., A, B, 0., C);
//	gemm('T', 'T', 1., real(A), B, 0., C);
}

#endif
#endif

