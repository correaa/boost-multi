#ifdef COMPILATION_INSTRUCTIONS
clang++ -O3 -std=c++17 -Wall $0 -o$0x && $0x && rm $0x; exit
#endif

#include "../array_ref.hpp"
#include "../array.hpp"

#include "../complex.hpp"

//#include<boost/iterator/transform_iterator.hpp>

#include<thrust/complex.h>

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

template<class Array> decltype(auto) Real(Array&& a){return real_(a, priority_1{});}

template<class Array, typename E = typename std::decay_t<Array>::element, typename R = decltype(std::real(E{})), typename I = decltype(E{}.imag())>
decltype(auto) Imag(Array&& a){
	struct C{R real; I imag;}; static_assert(sizeof(E) == sizeof(C));
	return member_array_cast<I>(reinterpret_array_cast<C>(a), &C::imag);
}

}}

using v3d = std::array<double, 3>;
struct particle{double mass; v3d __attribute__((aligned(2*sizeof(double)))) position;
	particle() = default;
	particle(double m, v3d v) : mass{m}, position{v}{}
	template<class P> particle(P&& p) : mass{p.mass}, position{p.position}{}
};
struct particle_ref{double& mass; v3d& position ;};
struct particles{multi::array<double,2> masses; multi::array<v3d,2> positions;
	auto operator()(int i, int j){return particle_ref{masses[i][j], positions[i][j]};}
};

int main(){

{
	multi::array<particle, 2> AoS({2, 2}); AoS[1][1] = particle{99., {1.,2.}};
	auto&& masses = multi::member_array_cast<double>(AoS, &particle::mass);
	assert( masses[1][1] == 99. );
	multi::array<double, 2> masses_copy = masses;
	particles SoA = {
		multi::member_array_cast<double>(AoS, &particle::mass), 
		multi::member_array_cast<v3d>(AoS, &particle::position)
	};
	particle p11 = SoA(1, 1); assert(p11.mass == 99. );
}
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
		struct complex{double real; double imag;};
		auto&& Areal = multi::member_array_cast<double>(A, &complex::real);
		auto&& Aimag = multi::member_array_cast<double>(A, &complex::imag);

		assert( Areal[1][0] == 22. );
		assert( Aimag[1][0] == 33. );
	}
	{
		auto&& Areal = multi::member_array_cast<double>(A, &multi::complex<double>::real);
		auto&& Aimag = multi::member_array_cast<double>(A, &multi::complex<double>::imag);

		assert( Areal[1][0] == 22. );
		assert( Aimag[1][0] == 33. );
	}
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
#if 0
	{
		auto&& Areal = real(A);
		auto&& Aimag = imag(A);
		auto Areal_copy = decay(real(A));
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
#endif
}

}

