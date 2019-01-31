#ifdef COMPILATION_INSTRUCTIONS
$CXX -O3 -std=c++14 -Wall -Wextra -Wpedantic `#-Wfatal-errors` $0 -o $0.x && time $0.x $@ && rm -f $0.x; exit
#endif

#include "../array_ref.hpp"
#include "../array.hpp"

#include<cassert>
#include<iostream>
#include<vector>
#include<complex>

using std::cout; using std::cerr;
namespace multi = boost::multi;

#include<iostream>

template<class T>
struct A{
	A(){}
	A(T*){}
	void m() const{};
private:
	friend void F(A const&){}
};
struct B{};

template<class T, typename = decltype(F(std::declval<T>()))>
void g(T const&){std::cout << "F exists" << std::endl;}
void g(...){std::cout << "F doesn't exists" << std::endl;}

int main(){
	A<double> a;
	g(a);
	B b;
	g(b);
	return 0;

	static_assert(std::is_same<std::pointer_traits<std::complex<double>*>::element_type, std::complex<double>>{}, "!");
	static_assert(std::is_same<std::pointer_traits<std::complex<double>*>::rebind<std::complex<double> const>, std::complex<double> const*>{}, "!");

	std::vector<std::complex<double>> d(100);
	std::vector<std::complex<double>> const dc(100);

	multi::array_ref<std::complex<double>, 2> A2D(d.data(), multi::iextensions<2>{10, 10});
	multi::array_ref<std::complex<double>, 2, std::complex<double>*> B2D(d.data(), {10, 10});
	
	assert( &A2D[3][4] == &B2D[3][4] );
	
//	multi::array_ref<std::complex<double>, 2> C2D(dc.data(), {10, 10}); // error double const* -> double*
	multi::array_ref<std::complex<double>, 2, std::complex<double> const*> D2D(dc.data(), {10, 10});
	multi::array_cref<std::complex<double>, 2> E2D(dc.data(), {10, 10});
	multi::array_cref<std::complex<double>, 2> F2D(d.data(), {10, 10});
//	F2D[3][4] = 4.; // error, not assignable
	A2D[7][8] = 3.;
	assert( F2D[7][8] == 3. );
	assert( &A2D[7][8] == &F2D[7][8] );

#if __cpp_deduction_guides
	multi::array_ref G2D(dc.data(), {10, 10}); assert( G2D == D2D );
#endif
	auto&& H2D = multi::make_array_ref<2>(dc.data(), {10, 10}); assert( H2D == D2D );

}

