#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4-*-
$CXXX $CXXFLAGS $0 -o $0.$X -lboost_unit_test_framework&&$0.$X&&rm $0.$X;exit
#endif
//  © Alfredo A. Correa 2018-2020

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi member cast"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

#include<complex>

namespace multi = boost::multi;

using v3d = std::array<double, 3>;

template<class T> void what(T&&) = delete;

BOOST_AUTO_TEST_CASE(member_array_cast_soa_aos){

	// some members might need explicit padding to work well with member_cast
	struct particle{
		double mass;
		v3d position alignas(2*sizeof(double));  // __attribute__((aligned(2*sizeof(double))))
	};

	class particles_SoA{
		multi::array<double,2> masses_; 
		multi::array<v3d,2> positions_;
	public:
		// cppcheck-suppress noExplicitConstructor ; particles_SoA can represent a particles' AoS
		particles_SoA(multi::array<particle, 2> const& AoS) : //NOLINT(
			masses_   (AoS.member_cast<double>(&particle::mass    )),
			positions_(AoS.member_cast<v3d   >(&particle::position))
		{}
		struct reference{
			double& mass;  // NOLINT(misc-non-private-member-variables-in-classes): exposed by design
			v3d& position; // NOLINT(misc-non-private-member-variables-in-classes): exposed by design
			operator particle() const{return {mass, position};} // NOLINT(google-explicit-constructor, hicpp-explicit-conversions): allow equal assignment
			auto operator+() const{return operator particle();}
		#if __cplusplus <= 201402L
		private:
			friend class particles_SoA;
			reference(reference const&) = default;
			reference(reference&&) = default;
		public:
			// NOLINTNEXTLINE(cert-oop54-cpp, fuchsia-trailing-return): simulate reference
			auto operator=(reference const& other) -> reference& { 
				std::tie(mass, position) = std::tie(other.mass, other.position);
				return *this;
			}
			// NOLINTNEXTLINE(fuchsia-trailing-return): simulate reference
			auto operator=(reference&& other) noexcept -> reference&{operator=(other); return *this;}
			~reference() noexcept = default;
		#endif
			auto operator==(reference const& other){return std::tie(mass, position) == std::tie(other.mass, other.position);}
			auto operator!=(reference const& other){return std::tie(mass, position) == std::tie(other.mass, other.position);}
		};
		auto operator()(int i, int j){
			return reference{masses_[i][j], positions_[i][j]};
		}
	};

	multi::array<particle, 2> AoS({2, 2}); 
	AoS[1][1] = particle{99., v3d{1., 2.}};

	auto&& masses = AoS.member_cast<double>(&particle::mass);
	BOOST_REQUIRE( size(masses) == 2 );
	BOOST_REQUIRE( masses[1][1] == 99. );

	multi::array<double, 2> masses_copy = masses;
	BOOST_REQUIRE( &masses_copy[1][1] != &masses[1][1] );
	
	particles_SoA SoA{AoS};

	BOOST_REQUIRE(SoA(1, 1).mass == 99. );
	
	particle p11 = SoA(1, 1); 
	BOOST_REQUIRE(p11.mass == 99. );
	
	auto autop11 = +SoA(1, 1);
	BOOST_REQUIRE(autop11.mass == 99. );

	SoA(1, 1).mass = 88;
	BOOST_REQUIRE(SoA(1, 1).mass == 88. );

	SoA(1, 1) = SoA(0, 0);
	BOOST_REQUIRE(SoA(1, 1).mass == SoA(0, 0).mass );
	BOOST_REQUIRE(SoA(1, 1) == SoA(0, 0) );

}

struct alignas(32) employee{
	std::string name;
	int16_t salary;
	std::size_t age;
//	private:
//	char padding_[9];// std::array<char, 9> padding_; // use alignment or padding to allow member_cast
};

BOOST_AUTO_TEST_CASE(member_array_cast_soa_aos_employee){

	multi::array<employee, 1> d1D = { {"Al"  , 1430, 35}, {"Bob"  , 3212, 34} }; 
	auto&& d1D_names = d1D.member_cast<std::string>(&employee::name);
	BOOST_REQUIRE( size(d1D_names) == size(d1D) );
	BOOST_REQUIRE(  d1D_names[1] ==  d1D[1].name );
	BOOST_REQUIRE( &d1D_names[1] == &d1D[1].name );

	multi::array<employee, 2> d2D = {
		{ {"Al"  , 1430, 35}, {"Bob"  , 3212, 34} }, 
		{ {"Carl", 1589, 32}, {"David", 2300, 38} }
	};
	BOOST_REQUIRE( d2D[0][0].name == "Al" );
	BOOST_REQUIRE( d2D[0][0].salary == 1430 );
	BOOST_REQUIRE( d2D[0][0].age == 35 );
	
	auto&& d2D_names = d2D.member_cast<std::string>(&employee::name);
	BOOST_REQUIRE( size(d2D_names) == size(d2D) ); 
	BOOST_REQUIRE( d2D_names[1][1] == "David" );

	multi::array<std::string, 2> d2D_names_copy{d2D_names};
//	BOOST_REQUIRE( d2D_names == d2D_names_copy );
//	BOOST_REQUIRE( base(d2D_names) != base(d2D_names_copy) );

}

//struct priority_0{}; struct priority_1 : priority_0{};
//template<class Array, typename E = typename std::decay_t<Array>::element, typename R = decltype(std::real(E{}))>
//decltype(auto) real_(Array&& a, priority_0){return a;}
//template<class Array, typename E = typename std::decay_t<Array>::element, typename R = decltype(std::real(E{})), typename I = decltype(E{}.imag())>
//decltype(auto) real_(Array&& a, priority_1){
//	struct C{R real; I imag;}; static_assert(sizeof(E) == sizeof(C), "!");
//	return std::forward<Array>(a).template reinterpret_array_cast<C>().template member_cast<R>(&C::real);
//}

//template<class Array> decltype(auto) Real(Array&& a){return real_(a, priority_1{});}

//template<class Array, typename E = typename std::decay_t<Array>::element, typename R = decltype(std::real(E{})), typename I = decltype(E{}.imag())>
//decltype(auto) Imag(Array&& a){
//	struct C{R real; I imag;}; static_assert(sizeof(E) == sizeof(C), "!");
//	return std::forward<Array>(a).template reinterpret_array_cast<C>().template member_cast<I>(&C::imag);
//}

//BOOST_AUTO_TEST_CASE(member_array_cast_complex){
//{
//	using complex = std::complex<double>;
//	multi::array<complex, 2> A = {
//		{ { 1.,  2.}, {  3.,  4.} },
//		{ {22., 33.}, {  5.,  9.} }
//	};
//	struct Complex{
//		double real;
//		double imag;
//	};
//	{
//		auto&& Acast = A.reinterpret_array_cast<Complex>();//multi::reinterpret_array_cast<Complex>(A);
//		auto&& Areal = Acast.member_cast<double>(&Complex::real);
//		auto&& Aimag = Acast.member_cast<double>(&Complex::imag);
//		BOOST_REQUIRE( Areal[1][0] == 22. and std::get<1>(strides(Areal)) == 2 );
//		BOOST_REQUIRE( Aimag[1][0] == 33. and std::get<1>(strides(Aimag)) == 2 );
//	}
//	{
//		auto&& Areal = A.reinterpret_array_cast<Complex>().member_cast<double>(&Complex::real);
//		auto&& Aimag = A.reinterpret_array_cast<Complex>().member_cast<double>(&Complex::imag);
//		BOOST_REQUIRE( Areal[1][0] == 22. and std::get<1>(strides(Areal)) == 2 );
//		BOOST_REQUIRE( Aimag[1][0] == 33. and std::get<1>(strides(Aimag)) == 2 );
//		Areal[1][0] = 55.;
//	}
//	{
//	//	auto&& Areal = Real(A);
//	//	auto&& Aimag = Imag(A);
//	//	auto Areal_copy = decay(Real(A));
//	//	BOOST_REQUIRE( Areal[1][0] == 55. and std::get<1>(strides(Areal)) == 2 );
//	//	BOOST_REQUIRE( Aimag[1][0] == 33. and std::get<1>(strides(Aimag)) == 2 );
//	//	Areal[1][0] = 888.;
//	}
//}
//{
//	multi::array<double, 2> A = {
//		{  1., 3.},
//		{ 22., 5.}
//	};
//	auto&& Areal = Real(A);
//	BOOST_REQUIRE( Areal[1][1] == 5. );
//	Areal[1][1] = 55.;
//}
//{
//	multi::array<double, 2> const A = {
//		{  1., 3.},
//		{ 22., 5.}
//	};
//	auto&& Areal = Real(A);
//	BOOST_REQUIRE( Areal[1][1] == 5. );
//}

//}

