#ifdef COMPILATION_INSTRUCTIONS
clang++ -O3 -std=c++17 -Wall $0 -o$0x && $0x && rm $0x; exit
#endif

#include "../array_ref.hpp"
#include "../array.hpp"

//#include<boost/iterator/transform_iterator.hpp>

#include<complex>
//#include<iostream>
//#include<numeric>

//using std::cout; using std::cerr;
namespace multi = boost::multi;

struct alignas(sizeof(std::string)) employee{
	std::string name;
	short salary;
	std::size_t age;
	employee(std::string name, short salary, std::size_t age) : name{name}, salary{salary}, age{age}{}
};

namespace boost{
namespace multi{

namespace{
	struct priority_0{}; struct priority_1 : priority_0{};
	template<class Array, typename E = typename std::decay_t<Array>::element, typename R = decltype(std::real(E{}))>
	decltype(auto) real_(Array&& a, priority_0){return a;}
	template<class Array, typename E = typename std::decay_t<Array>::element, typename R = decltype(std::real(E{})), typename I = decltype(E{}.imag())>
	decltype(auto) real_(Array&& a, priority_1){
		struct C{R real; I imag;}; static_assert(sizeof(E) == sizeof(C));
		return member_array_cast<R>(reinterpret_array_cast<C>(a), &C::real);
	}
}

template<class Array> decltype(auto) real(Array&& a){return real_(a, priority_1{});}

template<class Array, typename E = typename std::decay_t<Array>::element, typename R = decltype(std::real(E{})), typename I = decltype(E{}.imag())>
decltype(auto) imag(Array&& a){
	struct C{R real; I imag;}; static_assert(sizeof(E) == sizeof(C));
	return member_array_cast<I>(reinterpret_array_cast<C>(a), &C::imag);
}

}}

int main(){
{
	multi::array<employee, 2> d2D = {
		{ {"Al"  , 1430, 35}, {"Bob"  , 3212, 34} }, 
		{ {"Carl", 1589, 32}, {"David", 2300, 38} }
	};

	using multi::member_array_cast;

	auto&& d2Dmember = member_array_cast<std::string>(d2D, &employee::name);
	assert( d2Dmember[1][1] == "David" );
	multi::array<std::string, 2> d2Dmember_copy = d2Dmember;
}
{
	using complex = std::complex<double>;
	multi::array<complex, 2> A = {
		{ {1.,2.}, {3.,4.} },
		{ {22.,33.}, {5.,9.} }
	};
	struct Complex{
		double real;
		double imag;
	};
	{
		auto&& Acast = multi::reinterpret_array_cast<Complex>(A);
		auto&& Areal = multi::member_array_cast<double>(Acast, &Complex::real);
		auto&& Aimag = multi::member_array_cast<double>(Acast, &Complex::imag);
		assert( Areal[1][0] == 22. and std::get<1>(strides(Areal)) == 2 );
		assert( Aimag[1][0] == 33. and std::get<1>(strides(Aimag)) == 2 );
	}
	{
		auto&& Areal = multi::member_array_cast<double>(multi::reinterpret_array_cast<Complex>(A), &Complex::real);
		auto&& Aimag = multi::member_array_cast<double>(multi::reinterpret_array_cast<Complex>(A), &Complex::imag);
		assert( Areal[1][0] == 22. and std::get<1>(strides(Areal)) == 2 );
		assert( Aimag[1][0] == 33. and std::get<1>(strides(Aimag)) == 2 );
		Areal[1][0] = 55.;
	}
	{
		auto&& Areal = real(A);
		auto&& Aimag = imag(A);
		assert( Areal[1][0] == 55. and std::get<1>(strides(Areal)) == 2 );
		assert( Aimag[1][0] == 33. and std::get<1>(strides(Aimag)) == 2 );
		Areal[1][0] = 888.;
	}
	{
		multi::array<double, 2> A = {
			{  1., 3.},
			{ 22., 5.}
		};
		auto&& Areal = real(A);
		assert( Areal[1][1] == 5. );
		Areal[1][1] = 55.;
	}
	{
		multi::array<double, 2> const A = {
			{  1., 3.},
			{ 22., 5.}
		};
		auto&& Areal = real(A);
		assert( Areal[1][1] == 5. );
	//	Areal[1][1] = 55.;
	}
}

}

