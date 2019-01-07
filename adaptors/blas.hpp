#ifdef COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && c++ `#-DNDEBUG` -std=c++14 -Wall -Wextra  -I$HOME/prj -D_TEST_MULTI_ADAPTORS_BLAS -DADD_ $0x.cpp -o $0x.x -lblas && time $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif

#ifndef MULTI_ADAPTORS_BLAS_HPP
#define MULTI_ADAPTORS_BLAS_HPP

#include<iostream>
//#include <cblas/cblas.h>

#include<cassert>
#include<complex>
#include "../utility.hpp"
#include "../array_ref.hpp"
#include<numeric> // inner_product
//#include<algorithm>
#include<iostream>

#ifdef CBLAS_H
#define BLAS(NamE) cblas_##NamE
#else
#define BLAS(NamE) NamE##_
extern "C"{
#define s float
#define d double
#define c std::complex<s>
#define z std::complex<d>
#define v void 
#define C _Complex s
#define Z _Complex d
#define INT int
#define INTEGER INT const&
#define N INTEGER n
#define INCX INTEGER incx
#define INCY INTEGER incy
#define xROTG(T1, T2)     v BLAS(   T1##rotg)(T1 const*, T1 const*, T2*, T1*)
#define xROTMG(T)         v BLAS(   T##rotmg)(T*, T*, T*, T const&, T(&param)[5])
#define xROT(TT, T, S)    v BLAS(  TT##rot  )(N,              T       *x, INCX, T       *y, INCY, S const&, S const&)
#define xROTM(T)          v BLAS(   T##rotm )(N, T* x, INCX, T* y, INCY, T const(&p)[5])
#define xSWAP(T)          v BLAS(   T##swap )(N,              T       *x, INCX, T       *y, INCY)
#define xSCAL(TT, TA, TX) v BLAS(  TT##scal )(N, TA const& a, TX      *x, INCX                  )
#define xCOPY(T)          v BLAS(   T##copy )(N,              T const *x, INCX, T       *y, INCY) 
#define xAXPY(T)          v BLAS(   T##axpy )(N,  T const& a, T const *x, INCX, T       *y, INCY)
#define xDOT(R, TT, T)    R BLAS(  TT##dot  )(N,              T const *x, INCX, T const *y, INCY)
#define xDOTU(R, T)       R BLAS(   T##dotu )(N,              T const *x, INCX, T const *y, INCY)
#define xDOTC(R, T)       R BLAS(   T##dotc )(N,              T const *x, INCX, T const *y, INCY)
#define xxDOT(TT, T)      T BLAS(  TT##dot  )(N,  T const& a, T const *x, INCX, T const *y, INCY)
#define xNRM2(R, TT, T)   R BLAS(  TT##nrm2 )(N,              T const *x, INCX                  )   
#define xASUM(R, TT, T)   R BLAS(  TT##asum )(N,              T const *x, INCX                  )
#define IxAMAX(T)       INT BLAS(i##T##amax )(N,              T const* x, INCX                  )
xROTG(s, s); xROTG(d,d);// MKL extension xROTG(c, s); xROTG(z, d);
xROTMG(s); xROTMG(d);
xROT(s, s, s); xROT(d, d, d); xROT(cs, c, s); xROT(zd, z, d);
xROTM(s); xROTM(d);
xSWAP(s); xSWAP(d); xSWAP(c); xSWAP(z);
xSCAL(s, s, s); xSCAL( d, d, d); xSCAL( c, c, c); xSCAL(zd, d, z); xSCAL(cs, s, c);
xCOPY(s); xCOPY(d); xCOPY(c); xCOPY(z);
xAXPY(s); xAXPY(d); xAXPY(c); xAXPY(z);
xDOT(s, s, s); xDOT(d, d, d); xDOT(d, sd, s);
xDOTU(C, c); xDOTU(Z, z); 
xDOTC(C, c); xDOTC(Z, z); 
xxDOT(sds, s);
xNRM2(s, s, s); xNRM2(d, d, d); xNRM2(s, sc, c); xNRM2(d, dz, z);
xASUM(s, s, s); xASUM(d, d, d); xASUM(s, sc, c); xASUM(d, dz, z);
IxAMAX(s); IxAMAX(d); IxAMAX(c); IxAMAX(z);

#define TRANS const char& trans
#define NR const int& nr
#define NC const int& nc
#define LDA const int& lda
#define xGEMV(T) void BLAS(T##gemv)(TRANS, NR, NC, T const& a, T const* A, LDA, T const* X, INCX, T const& beta, T* Y, INCY);
xGEMV(s); xGEMV(d); xGEMV(c); xGEMV(z);
#undef xROTG
#undef xROTMG
#undef xROT
#undef xROTM
#undef xSCAL
#undef xSWAP
#undef xCOPY
#undef xAXPY
#undef xDOT
#undef xDOTU
#undef xDOTC
#undef xxDOT
#undef xNRM2
#undef xASUM
#undef IxAMAX 
#undef s
#undef d
#undef c
#undef z
#undef C
#undef Z
#undef v
#undef INTEGER
#undef N
#undef INCX
#undef INCY
}
#endif

namespace boost{
namespace multi{
namespace blas{

using s = float;
using d = double;
using c = std::complex<s>;
using z = std::complex<d>;
using v = void;

#define xrotg(T1, T2)                       v   rotg (T1 const& a, T1 const& b, T2& cc, T1& ss){BLAS(T1##rotg )(const_cast<T1*>(&a), const_cast<T1*>(&b), &cc, &ss);}
#define xrotmg(T)                           v   rotmg(T& d1, T& d2, T& A, T const& B, T(&p)[5]){BLAS(T##rotmg )(&d1, &d2, &A, B, p);}
#define xrot(T, TT, CS)   template<class S> v   rot  (S n,       T       *x, S incx, T       *y, S incy, CS const& c, CS const& s){       BLAS(TT##rot )(n,    x, incx, y, incy, c, s);}
#define xrotm(T)          template<class S> v   rotm (S n,       T       *x, S incx, T       *y, S incy, T const(&p)[5]){BLAS(T##rotm)(n, x, incx, y, incy, p);}
#define xswap(T)          template<class S> v   swap (S n,       T       *x, S incx, T       *y, S incy){       BLAS( T##swap)(n,    x, incx, y, incy);} 
#define xscal(XX, TA, TX) template<class S> TX* scal (S n, TA a, TX      *x, S incx                    ){       BLAS(XX##scal)(n, a, x, incx         ); return x+n*incx;}
#define xcopy(T)          template<class S> v   copy (S n,       T const *x, S incx, T       *y, S incy){       BLAS( T##copy)(n,    x, incx, y, incy);} 
#define xaxpy(T)          template<class S> T*  axpy (S n, T  a, T const *x, S incx, T       *y, S incy){       BLAS( T##axpy)(n, a, x, incx, y, incy); return y+n*incy;}
#define xdot(R, TT, T)    template<class S> R   dot  (S n,       T const *x, S incx, T const *y, S incy){return BLAS(TT##dot )(n,    x, incx, y, incy);                 }
xrotg(s, s); xrotg(d, d); //MKL extension xrotg(c, s); xrotg(z, d);
xrotmg(s); xrotmg(d);
xrot(s, s, s); xrot(d, d, d); xrot(c, cs, s); xrot(z, zd, d);
xrotm(s); xrotm(d);
xswap(s); xswap(d); xswap(c); xswap(z);
xscal(s, s, s); xscal(d, d, d); xscal(c, c, c); xscal(z, z, z); xscal(zd, d, z); xscal(cs, s, c);
xcopy(s); xcopy(d); xcopy(c); xcopy(z);
xaxpy(s); xaxpy(d); xaxpy(c); xaxpy(z);
xdot(s, s, s); xdot(d, d, d); xdot(d, sd, s);
#undef xrotg
#undef xrot
#undef xswap
#undef xscal
#undef xcopy
#undef xaxpy
#undef xdot

#ifndef CBLAS_H
#define xdotu(T) template<class S> T dotu(S n, T const* x, S incx, T const* y, S incy){return BLAS(T##dotu)(n, x, incx, y, incy);}
#define xdotc(T) template<class S> T dotc(S n, T const* x, S incx, T const* y, S incy){return BLAS(T##dotc)(n, x, incx, y, incy);}
xdotu(c); xdotu(z);
xdotc(c); xdotc(z);
#undef xdotu
#undef xdotc
#else
#define xdotu(T) template<class S> T dotu(S n, T const* x, S incx, T const* y, S incy){T ret; BLAS(T##dotu_sub)(n, x, incx, y, incy, &ret); return ret;}
#define xdotc(T) template<class S> T dotc(S n, T const* x, S incx, T const* y, S incy){T ret; BLAS(T##dotc_sub)(n, x, incx, y, incy, &ret); return ret;}
xdotu(c); xdotu(z);
xdotc(c); xdotc(z);
#undef xdotu
#undef xdotc
#endif

template<class S> s apdot(S n, s const& a, s const* x, S incx, s const* y, S incy){return BLAS(sdsdot)(n, a, x, incx, y, incy);}

#define xnrm2(R, T, TT) template<class S>    R nrm2 (S n, T const* x, S incx){return BLAS(TT##nrm2  )(n, x, incx);}
#define xasum(T, TT)    template<class S> auto asum (S n, T const* x, S incx){return BLAS(TT##asum  )(n, x, incx);}
#define ixamax(T)       template<class S> auto iamax(S n, T const* x, S incx){return BLAS(i##T##amax)(n, x, incx);}
xnrm2(s, s, s); xnrm2(d, d, d); xnrm2(s, c, sc); xnrm2(d, z, dz);
xasum(s, s) ; xasum (d, d); xasum (c, sc); xasum (z, dz);
ixamax(s); ixamax(d); ixamax(c); ixamax(z);
#undef xnrm2
#undef xasum
#undef ixamax

///////////////////////////////////////////////////////////////////////////////
// LEVEL2
#define xgemv(T) template<class C, class S> v gemv(C trans, S m, S n, T const& a, T const* A, S lda, T const* X, S incx, T beta, T* Y, S incy){BLAS(T##gemv)(trans, m, n, a, A, lda, X, incx, beta, Y, incy);}
xgemv(s); xgemv(d); xgemv(c); xgemv(z);

}}}
///////////////////////////////////////////////////////////////////////////////

namespace boost{
namespace multi{
namespace blas{

template<class T> struct cs{
	T c; T s;
	operator multi::array<T, 2>() const{return {{c, s}, {-s, c}};}
};
template<class T> struct ab{T a; T b; using value_type = T;};
template<class T> struct modified_rotation{
	T data_[5];
	int flag() const{return data_[0];}
	multi::array<T, 2> h() const{
		switch(flag()){
			case -1: return {{data_[1], data_[2]}, {data_[3], data_[4]}};
			case  0: return {{T{+1}   , data_[2]}, {data_[3], T{+1}   }};
			case  1: return {{data_[1], T{+1}   }, {T{-1}   , data_[4]}};
			case -2: return {{T{+1}   , T{ 0}   }, {T{ 0}   , T{+1}   }};
			default: assert(0); return {};
		}
	}
};

template<class T>
auto rotg(T& a, T& b){
	cs<T> ret;
//	using blas::rotg;
	rotg(a, b, ret.c, ret.s );
	return ret;
}

template<class T>
modified_rotation<T> rotmg(T& d1, T& d2, T& x1, T const& y1){
	modified_rotation<T> ret;
	rotmg(d1, d2, x1, y1, ret.data_);
	return ret;
}

//template<class T>
//auto rotmg(T& d1, T& d2, T& b1, T const& b2){
//	modified_rotation<T> ret;
//	rotmg(d1, d2, b1, b2, ret);
//	return ret;
//}

template<class X1D, class Y1D, class T>
auto rot(X1D&& x, Y1D&& y, T const& c, T const& s){
	assert( size(x) == size(y) );
	assert( offset(x) == 0 and offset(y) == 0 );
//	using blas::rot;
	rot(size(x), origin(x), stride(x), origin(y), stride(y), c, s);
	return std::tie(x, y);
}
template<class X1D, class Y1D, class CS>
auto rot(X1D&& x, Y1D&& y, CS const& cs){
	return rot(std::forward<X1D>(x), std::forward<Y1D>(y), cs.c, cs.s);
}
template<class X1D, class Y1D, class M>
auto rotm(X1D&& x, Y1D&& y, M const& param){
	using boost::multi::size;
	assert( size(x) == size(y) );
	assert( offset(x) == 0 and offset(y) == 0);
	rotm(size(x), origin(x), stride(x), origin(y), stride(y), param.data_);
}

template<class X1D, class Y1D>
auto swap(X1D&& x, Y1D&& y)
->decltype(swap(size(x), origin(x), stride(x), origin(y), stride(y)))
{
	assert( size(x) == size(y) );
	assert( offset(x) == 0 and offset(y) == 0 );
//	using blas::swap;
	return swap(size(x), origin(x), stride(x), origin(y), stride(y));
}

template<class T, class X1D>//, typename = decltype(T{}*std::declval<X1D>()[0])>
auto scal(T a, X1D&& m)
->decltype(     size(m),    origin(m), stride(m) , m){
	assert( offset(m) == 0 );
	using blas::scal;
	return scal(size(m), a, origin(m), stride(m)), m;}

template<class X1D, class Y1D>
void copy(X1D const& x, Y1D&& y){
	assert( size(x) == size(y) );
	assert( offset(x) == 0 and offset(y) == 0 );
	using blas::copy;
	return copy(size(x), origin(x), stride(x), origin(y), stride(y));
}

template<class T, class X1D, class Y1D>
auto axpy(T a, X1D const& x, Y1D&& y)
->decltype(     size(x),    origin(x), stride(x), origin(y), stride(y) , y){
	assert( size(x) == size(y) );
	assert( not offset(x) and not offset(y) );
	using blas::axpy;
	return axpy(size(x), a, origin(x), stride(x), origin(y), stride(y)), y;}

template<class X1D, class Y1D>
auto dot(X1D const& x, Y1D const& y){
	assert( size(x) == size(y) );
	assert( not offset(x) and not offset(y) );
	using blas::dot;
	return dot(size(x), origin(x), stride(x), origin(y), stride(y));
}

template<class X1D, class Y1D>
auto dotu(X1D const& x, Y1D const& y)
->decltype(dotu(size(x), origin(x), stride(x), origin(y), stride(y))){
	assert( size(x) == size(y) );
	assert( not offset(x) and not offset(y) );
//	using blas::dotu;
	return dotu(size(x), origin(x), stride(x), origin(y), stride(y));
}

template<class X1D, class Y1D>
auto dotc(X1D const& x, Y1D const& y){
	assert( size(x) == size(y) );
	assert( not offset(x) and not offset(y) );
	using blas::dotc;
	return dotc(size(x), origin(x), stride(x), origin(y), stride(y));
}

template<class T, class X1D, class Y1D> 
auto xdot(T a, X1D const& x, Y1D const& y){
	assert( size(x) == size(y) );
	assert( not offset(x) and not offset(y) );
//	using blas::apdot;
	return apdot(size(x), a, origin(x), stride(x), origin(y), stride(y));
}

template<class X1D> 
auto nrm2(X1D const& x){
	assert( not offset(x) );
//	using blas::nrm2;
	return nrm2(size(x), origin(x), stride(x));
}

template<class X1D> 
auto asum(X1D const& x){
	assert( not offset(x) );
//	using blas::asum;
	return asum(size(x), origin(x), stride(x));
}

template<class X1D> 
auto iamax(X1D const& x){
	assert( not offset(x) );
//	using blas::iamax;
	return iamax(size(x), origin(x), stride(x))
	#ifndef CBLAS_H
		- 1
	#endif
	;
}

}}

namespace multi{
namespace blas{

struct trans{enum : char{N='N', T='T', C='C'};};

template<class Trans, class T, class A2D, class X1D, class Y1D
//	,typename = typename std::enable_if<A2D::dimensionality == 2 and 1DX::dimensionality == 1 and std::decay_t<Y1D>::dimensionality == 1>::type
>
Y1D gemv(Trans IN, T a, A2D const& A, X1D const& x, T beta, Y1D&& y){
    assert( (IN == 'N') || (IN == 'T') || (IN == 'H')  );
	if(IN=='T' or IN=='H') assert( size(x)==std::get<1>(A.shape()) and size(y)==std::get<0>(A.shape()));
	else if(IN == 'N') assert( size(x) == std::get<0>(A.shape()) and size(y) == std::get<1>(A.shape()));
	assert( std::get<1>(strides(A)) == 1 ); // gemv is not implemented for arrays with non-leading stride != 1
	auto M = std::get<1>(shape(A));
	auto N = std::get<0>(shape(A));
	gemv(IN, M, N, a, origin(A), stride(A), origin(x), stride(x), beta, origin(y), stride(y));
	return std::forward<Y1D>(y);
} //y := alpha*A*x + beta*y,

template<class A, class B, class RowIt, class ConstIt, class It>
It gemv(A const& a, RowIt M_first, RowIt M_last, ConstIt X_first, B const& b, It Y_first){
	using std::transform; using std::inner_product; using std::begin; using std::end;
	return transform(M_first, M_last, Y_first, Y_first, [&](auto const& r, auto const& e){
		return a*inner_product(begin(r), end(r), X_first, typename std::iterator_traits<It>::value_type{0}) + b*e;
	});
}

struct conj{template<class T> auto operator()(T const& t) const{using std::conj; return conj(t);}};

template<class A, class B, class RowIt, class ConstIt, class It, class Conj>
It gemv(A const& a, RowIt M_first, RowIt M_last, ConstIt X_first, B const& b, It Y_first, Conj&& /*conj*/){
	std::cout<< __LINE__ <<std::endl;
	using std::transform; using std::inner_product; using std::begin; using std::end;
	return transform(M_first, M_last, Y_first, Y_first, [&](auto&& r, auto&& e){
		return a*inner_product(begin(r), end(r), X_first, typename std::iterator_traits<It>::value_type{0}/*, std::plus<>{}, [&](auto const& a, auto const& b){return conj(a)*b;}*/) + b*e;
	});
}

template<class AB, class RowIt, class ConstIt, class It>
It gemv(AB const& a, RowIt M_first, RowIt M_last, ConstIt X_first, AB const& b, It Y_first){
	assert( stride(M_first) == stride(M_last) );
	std::cout<< __LINE__ <<std::endl;
	using std::distance;
#ifndef NO_BLAS
	     if(stride(*M_first) == 1){std::cout<< __LINE__ <<std::endl; gemv(trans::T, M_first->size(), std::distance(M_first, M_last), a, base(M_first), stride( M_first), base(X_first), stride(X_first), b, base(Y_first), stride(Y_first));}
	else if(stride( M_first) == 1){std::cout<< __LINE__ <<std::endl; gemv(trans::N, std::distance(M_first, M_last),M_first->size(), a, base(M_first), stride(*M_first), base(X_first), stride(X_first), b, base(Y_first), stride(Y_first));}
	else
#endif
#ifdef NO_GENERICBLAS
		assert(0);
#else
		gemv<AB, AB>(a, M_first, M_last, X_first, b, Y_first);
#endif
	return Y_first + std::distance(M_first, M_last);
}

template<class RowIt, class ConstIt, class It>
It gemv(std::complex<double> const& a, RowIt M_first, RowIt M_last, ConstIt X_first, std::complex<double> const& b, It Y_first, blas::conj&&){
	using AB = std::complex<double>;
	std::cout<< __LINE__ <<std::endl;
	assert( stride(M_first) == stride(M_last) );
	using std::distance;
	
	if(stride( M_first) == 1){
	 	std::cout<< __LINE__ << " " << stride(*M_first) << " " << std::distance(M_first, M_last) << " " << M_first->size() << std::endl;
     	gemv(trans::C, std::distance(M_first, M_last), M_first->size(), a, base(M_first), stride(*M_first), base(X_first), stride(X_first), b, base(Y_first), stride(Y_first));
     	std::cout<< __LINE__ << " " << stride(*M_first) << " " << std::distance(M_first, M_last) << " " << M_first->size() << std::endl;
//			assert(0);
     }else{
     	gemv<AB, AB>(a, M_first, M_last, X_first, b, Y_first, blas::conj{});
     }

#if 0
	
#ifndef NO_BLAS
	     if(stride( M_first) == 1){
	     	std::cout<< __LINE__ << " " << stride(*M_first) << " " << std::distance(M_first, M_last) << " " << M_first->size() << std::endl;
	     	gemv('C', std::distance(M_first, M_last), M_first->size(), a, base(M_first), stride(*M_first), base(X_first), stride(X_first), b, base(Y_first), stride(Y_first));
	     	std::cout<< __LINE__ << " " << stride(*M_first) << " " << std::distance(M_first, M_last) << " " << M_first->size() << std::endl;
//			assert(0);
	     }
	else
#endif
#ifdef NO_GENERICBLAS
	assert(0);
#else
	gemv<AB, AB>(a, M_first, M_last, X_first, b, Y_first, blas::conj{});
#endif
#endif
 	std::cout<< __LINE__ << " " << stride(*M_first) << " " << std::distance(M_first, M_last) << " " << M_first->size() << std::endl;
	return Y_first;// + std::distance(M_first, M_last);
}

template<class T, class A2D, class X1D, class Y1D>
Y1D gemv(T const& a, A2D const& A, X1D const& x, T const& b, Y1D&& y){
	std::cout<< __LINE__ <<std::endl;
	assert( size(x)==std::get<1>(shape(A)) and size(y)==std::get<0>(shape(A)) );
	auto last = gemv(a, begin(A), end(A), begin(x), b, begin(y));
	assert( last == end(y) );
	return std::forward<Y1D>(y);
	
//	else if(IN == 'N') 
//	assert( std::get<1>(strides(A)) == 1 ); // gemv is not implemented for arrays with non-leading stride != 1
	auto m = std::get<1>(shape(A));
	auto n = std::get<0>(shape(A));
	if(std::get<1>(strides(A)) == 1){
	//	if(IN=='T' or IN=='H') 
		assert( size(x)==std::get<1>(A.shape()) and size(y)==std::get<0>(A.shape()));
		gemv(trans::T, m, n, a, origin(A), std::get<0>(strides(A)), origin(x), stride(x), b, origin(y), stride(y));
	}else if(std::get<0>(strides(A)) == 1){
		assert( size(x) == std::get<0>(A.shape()) and size(y) == std::get<1>(A.shape()));
		gemv(trans::N, m, n, a, origin(A), std::get<1>(strides(A)), origin(x), stride(x), b, origin(y), stride(y));
	}else{assert(0);}
	return std::forward<Y1D>(y);
} //y := alpha*A*x + beta*y,

template<class T, class A2D, class X1D, class Y1D, class Conj>
Y1D&& gemv(T const& a, A2D const& A, X1D const& x, T const& b, Y1D&& y, Conj&& c){
	std::cout<<__LINE__ <<std::endl;
	assert( size(x)==std::get<1>(shape(A)) and size(y)==std::get<0>(shape(A)) );
//	auto last = 
	gemv(a, begin(A), end(A), begin(x), b, begin(y), std::forward<Conj>(c));
	std::cout<< __LINE__ <<std::endl;
//	assert( last == end(y) );
	return std::forward<Y1D>(y);
} //y := alpha*A*x + beta*y,

//template<class T, class A2D, class X1D, class Y1D = typename A2D::value_type>
//Y1D gemv(T const& a, A2D const& A, X1D const& x){
//	return gemv(a, A, x, 0., Y1D(std::get<1>(extensions(A))));
//}
//template<class A2D, class X1D, class Y1D>
//Y1D gemv(A2D const& A, X1D const& x, Y1D&& y){
//	return gemv(1., A, x, 0., std::forward<Y1D>(y));
//} //y := alpha*A*x
//	gemm<'T', 'T'>(1., A, B, 0., C); // C = T(A*B) = T(B)*T(A) or T(C) = A*B
//	gemm<'N', 'N'>(1., A, B, 0., C); // C = B*A = T(T(A)*T(B)) or T(C) = T(A)*T(B)
//	gemm<'T', 'N'>(1., A, B, 0., C); // C = T(A*T(B)) = B*T(A) or T(C) = A*T(B)
//	gemm<'N', 'T'>(1., A, B, 0., C); // C =  T(T(A)*B) = T(B)*A or T(C) = T(A)*B

template<class A, class B, class M2D, class X1D, class Y1D, class Conj>
Y1D gemv(A const& a, M2D const& M, X1D const& X, B const& b, Y1D&& Y, Conj&& c){
//	for(auto i : extension(Y2)){
//		decltype(b*Y2[i]/a) Mxi{0};
//		for(auto j : extension(X)) Mxi += c(M[i][j])*X[j];
//		Y2[i] = a*Mxi + b*Y2[i];
//	}
	assert( std::make_tuple(size(Y), size(X)) == shape(M) ); //	assert( size(X)==std::get<1>(shape(M)) and size(Y)==std::get<0>(shape(M)) );
	gemv<A, B>(a, begin(M), end(M), begin(X), b, begin(Y), std::forward<Conj>(c));
	return std::forward<Y1D>(Y);
}

template<class A, class B, class M2D, class X1D, class Y1D>
Y1D gemv(A const& a, M2D const& M, X1D const& X, B const& b, Y1D&& Y){
	return gemv(a, M, X, b, std::forward<Y1D>(Y), [](auto&& e){return std::forward<decltype(e)>(e);});
};

}}

}

#if _TEST_MULTI_ADAPTORS_BLAS

#include "../array.hpp"
#include "../utility.hpp"

#include<complex>
#include<cassert>
#include<iostream>
#include<numeric>
#include<algorithm>

using std::cout;
namespace multi = boost::multi;

int main(){

	multi::array<double, 2> const CA = {
		{1.,  2.,  3.,  4.},
		{5.,  6.,  7.,  8.},
		{9., 10., 11., 12.}
	};
	{
		double const a0 = 2./3.; 
		double const b0 = 4./5.;
		double a = a0, b = b0;
		double c, s;
		using multi::blas::rotg;
		rotg(a, b, c, s);
		using std::abs; using std::sqrt;
		assert( abs(c - 5./sqrt(61.)) < 1e-15 );
		assert( abs(s - 6./sqrt(61.)) < 1e-15 );
		assert( abs(a - (b0>0?1:-1)*sqrt(a0*a0 + b0*b0)) < 1e-15 );
		assert( abs(  c*c  + s*s  - 1 ) < 1e-15 );
		assert( abs(  c*a0 + s*b0 - a ) < 1e-15 );
		assert( abs( -s*a0 + c*b0     ) < 1e-15 );
	}
	{
		using multi::blas::rotmg;
		double const x0 = 2./3.; 
		double const y0 = 4./5.;
		double const D1 = 1.;
		double const D2 = 1.;
		{
			double d1 = D1;
			double d2 = D2;
			double x1 = x0;
			double const y1 = y0;
			auto m = rotmg(d1, d2, x1, y1);
			assert( std::abs(x1 -( m.h()[0][0]*x0*std::sqrt(D1) + m.h()[0][1]*y0*std::sqrt(D2) )) < 1e-15 );
			assert( std::abs(      m.h()[1][0]*x0*std::sqrt(D1) + m.h()[1][1]*y0*std::sqrt(D2)  ) < 1e-15 );
		}
		{
			double x1 = x0;
			double const y1 = y0;
			double d1 = D1;
			double d2 = D2;
			multi::array<double, 1> X0 = {x0*std::sqrt(D1)};
			multi::array<double, 1> Y0 = {y0*std::sqrt(D2)};
			multi::array<double, 1> X1 = X0;
			multi::array<double, 1> Y1 = Y0;
			rotm(X1, Y1, rotmg(d1, d2, x1, y1));
			assert( std::abs( X1[0] - x1 ) <1e-15 );
			assert( Y1[0] == 0. );
		}
	}
	{
		multi::array<double, 1> X = CA[0];
		multi::array<double, 1> Y = CA[2];
		using multi::blas::rot;
		using std::cos; using std::sin;
		rot(X, Y, cos( 1.2), sin( 1.2));
		assert(X[1] == CA[0][1]*cos(1.2) + CA[2][1]*sin(1.2));
		assert(Y[1] == CA[2][1]*cos(1.2) - CA[0][1]*sin(1.2));
	}
	{
		multi::array<double, 1> const a0 = {2./3.};
		multi::array<double, 1> const b0 = {4./5.};
		using multi::blas::rotg;
		{
			double a = a0[0], b = b0[0];
			auto cs = rotg(a, b);
			multi::array<double, 1> a1 = a0;
			multi::array<double, 1> b1 = b0;
			rot(a1, b1, cs);
			assert( std::abs(a1[0] - a) < 1e-15 );
			assert( std::abs(b1[0]    ) < 1e-15 );
		}
		{
			double a = a0[0], b = b0[0];
			multi::array<double, 1> a1 = a0;
			multi::array<double, 1> b1 = b0;
			rot(a1, b1, rotg(a, b));
			assert( std::abs(a1[0] - a) < 1e-15 );
			assert( std::abs(b1[0]    ) < 1e-15 );
		}
	}
	{
		multi::array<double, 2> A = CA;
		using multi::blas::swap;
		swap(A.rotated(1)[1], A.rotated(1)[3]);
	}
	{
		multi::array<double, 2> A = CA;
		using multi::blas::scal;
		auto&& S = scal(2., A.rotated(1)[1]);
		assert( A[2][1] == 20. );
		assert( S[0] == 4. );
	}
	{
		multi::array<double, 1> const A = {1., 2., 3., 4.};
		multi::array<double, 1> B = {5., 6., 7., 8.};
		using multi::blas::copy;
		copy(A, B);
		assert( B == A );
	}
	{
		multi::array<double, 2> A = CA;
		multi::array<double, 1> const B = CA[2];
		using multi::blas::axpy;
		axpy(2., B, A[1]);
		assert( A[1][2] == 2.*B[2] + CA[1][2] );
	}
	{
		using multi::blas::dot;
		auto d = dot(CA[1], CA[2]);
		assert(d == std::inner_product(begin(CA[1]), begin(CA[2]), end(CA[1]), 0.));
	}
	using dcomplex = std::complex<double>;
	{
		multi::array<dcomplex, 2> A = CA;
		A[1][1] += dcomplex{1.1, 2.1};
		A[2][1] -= dcomplex{1.1, 2.1};
		using multi::blas::dotu;
		using multi::blas::dotc;
		using multi::blas::nrm2;
		using multi::blas::asum;
		assert(dotu(A[1], A[2]) == std::inner_product(begin(A[1]), begin(A[2]), end(A[1]), dcomplex{}, std::plus<>{}, [](auto&& a, auto&& b){return a*b;}));
		assert(dotc(A[1], A[2]) == std::inner_product(begin(A[1]), begin(A[2]), end(A[1]), dcomplex{}, std::plus<>{}, [](auto&& a, auto&& b){return conj(a)*b;}));
		assert(nrm2(A[1]) == std::sqrt(dotc(A[1], A[1])));
		assert(dotu(A[1], A[2]) == std::inner_product(begin(A[1]), begin(A[2]), end(A[1]), dcomplex{}, std::plus<>{}, [](auto&& a, auto&& b){return a*b;}));
		assert(asum(A[1]) == std::accumulate(begin(A[1]), end(A[1]), 0., [](auto&& a, auto&& b){return a + std::abs(real(b)) + std::abs(imag(b));}));
	}
	{
		auto const& A = CA.rotated(1)[1]; (void)A;
		using multi::blas::iamax;
		assert(iamax(A) == std::distance(begin(A), std::max_element(begin(A), end(A), [](auto&& a, auto&& b){
			return std::abs(a) < std::abs(b);
		})));
	}
	
///////////////////////////////////////////////////////////////////////////////
	{
		multi::array<double, 2> const M = {
			{ 9., 24., 30., 9.},
			{ 4., 10., 12., 7.},
			{14., 16., 36., 1.}
		};
		assert( M[2][0] == 14. );
		multi::array<double, 1> const X = {1.1,2.1,3.1, 4.1};
		multi::array<double, 1> Y = {4.,5.,6.};
		multi::array<double, 1> Y2 = Y;
		multi::array<double, 1> Y3 = {214.02, 106.43, 188.37};
		double a = 1.1, b = 1.2;
		multi::blas::gemv(a, M, X, b, Y);
		multi::blas::gemv<double, double>(a, M, X, b, Y2);
		assert( Y == Y2 );
		assert( std::abs(Y[1] - Y3[1]) < 1e-14 );
	}
	{
		multi::array<double, 2> const M = {
			{ 9., 24., 30., 9.},
			{ 4., 10., 12., 7.},
			{14., 16., 36., 1.}
		};
		assert( M[2][0] == 14. );
		multi::array<double, 1> const X = {1.1,2.1,3.1};
		multi::array<double, 1> Y = {4.,5.,6., 7.};
		multi::array<double, 1> Y2 = Y;
		multi::array<double, 1> Y3 = {72.67, 112.7, 193.98, 38.87};
		double a = 1.1, b = 1.2;
		multi::blas::gemv(a, M.rotated(1), X, b, Y);
		multi::blas::gemv<double, double>(a, M.rotated(1), X, b, Y2);
		assert( std::abs(Y[1] - Y2[1]) < 1e-13 );
		assert( std::abs(Y[1] - Y3[1]) < 1e-13 );
	}
	auto const I = dcomplex{0.,1.};
	{
		multi::array<dcomplex, 2> const M = {
			{ 9. + 1.*I, 24. + 2.*I, 30. + 3.*I, 9. + 1.*I}, 
			{ 4. + 1.*I, 10. + 1.*I, 12. - 2.*I, 7. + 2.*I}, 
			{14. + 3.*I, 16. - 4.*I, 36. + 1.*I, 1. - 2.*I}
		};
		multi::array<dcomplex, 1> const X = {1.1+I*2., 2.1+I*1.1, 3.1+I*8. , 4.1+I*1.2};
		multi::array<dcomplex, 1> Y = {4.+I*3.1,5.-I*9.,6.+I*1.};
		multi::array<dcomplex, 1> Y2 = Y;
		multi::array<dcomplex, 1> const Y3 = {-486.81+698.69*I, -125.08+359.44*I, -504.21+707.01*I};
		dcomplex a = 1.1+I*2.1, b = 1.2+I*3.;
		cout<<">>"<<__LINE__ <<std::endl;
		multi::blas::gemv(a, M, X, b, Y);
		cout<<">>"<<__LINE__ <<std::endl;
		multi::blas::gemv<dcomplex, dcomplex>(a, M, X, b, Y2);
		using std::abs;
		assert( abs(Y[0] - Y3[0]) < 1e-12 && abs(Y[1] - Y3[1]) < 1e-12 && abs(Y[2] - Y3[2]) < 1e-12 );
	}
	{
		multi::array<dcomplex, 2> const M = {
			{9. + 1.*I, 4. + 1.*I, 14. + 3.*I}, 
			{24. + 2.*I, 10. + 1.*I, 16. - 4.*I}, 
			{30. + 3.*I, 12. - 2.*I, 36. + 1.*I}, 
			{9. + 1.*I,   7. + 2.*I, 1. - 2.*I}
		};
		multi::array<dcomplex, 1> const X = {1.1+I*2., 2.1+I*1.1, 3.1+I*8. , 4.1+I*1.2};
		multi::array<dcomplex, 1> Y = {4.+I*3.1,5.-I*9.,6.+I*1.};
		multi::array<dcomplex, 1> Y2 = Y;
		multi::array<dcomplex, 1> const Y3 = {-486.81+698.69*I, -125.08+359.44*I, -504.21+707.01*I};
		std::complex<double> a = 1.1+I*2.1, b = 1.2+I*3.;
		cout<<">>"<<__LINE__ <<std::endl;
		multi::blas::gemv(a, M.rotated(), X, b, Y);
		cout<<">>"<<__LINE__ <<std::endl;
		multi::blas::gemv<dcomplex, dcomplex>(a, M.rotated(), X, b, Y2);
		assert( abs(Y[0] - Y3[0]) < 1e-12 && abs(Y[1] - Y3[1]) < 1e-12 && abs(Y[2] - Y3[2]) < 1e-12 );
		assert( abs(Y[0] - Y2[0]) < 1e-12 && abs(Y[1] - Y2[1]) < 1e-12 && abs(Y[2] - Y2[2]) < 1e-12 );
	}
	{
		multi::array<dcomplex, 2> const M = {
			{9. + 1.*I, 4. + 1.*I, 14. + 3.*I}, 
			{24. + 2.*I, 10. + 1.*I, 16. - 4.*I}, 
			{30. + 3.*I, 12. - 2.*I, 36. + 1.*I}, 
			{9. + 1.*I,   7. + 2.*I, 1. - 2.*I}
		};
		multi::array<dcomplex, 1> const X = {1.1+I*2., 2.1+I*1.1, 3.1+I*8. , 4.1+I*1.2};
		multi::array<dcomplex, 1> Y = {4.+I*3.1,5.-I*9.,6.+I*1.};
	//	multi::array<dcomplex, 1> Y2 = Y;
	//	multi::array<dcomplex, 1> const Y3 = {-486.81+698.69*I, -125.08+359.44*I, -504.21+707.01*I};
		std::complex<double> a = 1.1+I*2.1, b = 1.2+I*3.;
		cout<<">>"<<__LINE__ <<std::endl;
		multi::blas::gemv(a, M.rotated(), X, b, Y, multi::blas::conj{});
	//	cout<<">>"<<__LINE__ <<std::endl;
	//	multi::blas::gemv<dcomplex, dcomplex>(a, M.rotated(), X, b, Y2);
		cout<< Y[0] <<' '<< Y[1] <<' '<< Y[2] <<std::endl;
	//	assert( abs(Y[0] - Y3[0]) < 1e-12 && abs(Y[1] - Y3[1]) < 1e-12 && abs(Y[2] - Y3[2]) < 1e-12 );
	//	assert( abs(Y[0] - Y2[0]) < 1e-12 && abs(Y[1] - Y2[1]) < 1e-12 && abs(Y[2] - Y2[2]) < 1e-12 );
	}
	cout<<__LINE__ <<std::endl;
	return 0;
	{
//		multi::array<dcomplex, 2> const M = {
//			{ 9.+I*1., 24.+I*2., 30.+I*3.},
//			{ 4.+I*1., 10.+I*1., 12.-I*2.},
//			{14.+I*3., 16.-I*4., 36.+I*1.},
//			{ 9.+I*1.,  7.+I*2.,  1.-I*2.}
//		};
		multi::array<dcomplex, 2> const M = {
			{ 9. + 1.*I,  4. + 1.*I, 14. + 3.*I, 9. + 1.*I}, 
			{24. + 2.*I, 10. + 1.*I, 16. - 4.*I, 7. + 2.*I}, 
			{30. + 3.*I, 12. - 2.*I, 36. + 1.*I, 1. - 2.*I}
		};
		multi::array<dcomplex, 1> const X = {1.1+I*2., 2.1+I*1.1, 3.1+I*8., 4.1+I*1.2};
		multi::array<dcomplex, 1> Y = {4.+I*3.1,5.-I*9.,6.+I*1.};
		multi::array<dcomplex, 1> Y2 = Y;
		multi::array<dcomplex, 1> const Y3 = {-134.97+423.67*I, -265.81+431.55*I, -567.81+809.37*I};
		dcomplex const a = 1.1+I*2.1, b = 1.2+I*3.;
		cout<< "708" <<std::endl;
	//	multi::blas::gemv(a, M.rotated(), X, b, Y, multi::blas::conj<>{});
		zgemv_('N', std::get<0>(M.shape()), std::get<1>(M.shape()), a, M.base(), M.stride(), X.base(), X.stride(), b, Y.base(), Y.stride());
	//	zgemv_('T', std::get<1>(M.shape()), std::get<0>(M.shape()), a, M.base(), 2*std::get<0>(M.strides()), X.base(), stride(X), b, Y.base(), stride(Y));
		multi::blas::gemv<std::complex<double>, std::complex<double>>(a, M, X, b, Y2);
	//	multi::blas::gemv<std::complex<double>, std::complex<double>>(a, M.rotated(1), X, b, Y2);

	//	multi::blas::gemv<dcomplex, dcomplex>(a, M.rotated(), X, b, Y2, multi::blas::conj<>{});
		cout << Y[0] <<' '<< Y[1] <<' '<< Y[2] <<std::endl;
		cout << Y2[0] <<' '<< Y2[1] <<' '<< Y2[2] <<std::endl;
		cout << "finished" << std::endl;
	//	assert( std::abs(Y[1] - Y3[1]) < 1e-12 );
	//	assert( Y[1] == Y2[1] );
	}
}

#endif
#endif

