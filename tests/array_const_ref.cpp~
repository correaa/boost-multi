#ifdef COMPILATION_INSTRUCTIONS
clang++ -O3 -std=c++17 -Wall `#-fmax-errors=2` `#-Wfatal-errors` -I${HOME}/prj $0 -o $0.x && time $0.x $@ && rm -f $0.x; exit
#endif

#include "../array_ref.hpp"

#include<algorithm>
#include<cassert>
#include<iostream>
#include<cmath>
#include<vector>
#include<list>
#include<numeric> //iota

using std::cout; using std::cerr;
namespace multi = boost::multi;
using multi::index;

int main(){

	double const d2D[4][5] {
		{ 0,  1,  2,  3,  4}, 
		{ 5,  6,  7,  8,  9}, 
		{10, 11, 12, 13, 14}, 
		{15, 16, 17, 18, 19}
	};
	static_assert( std::is_same<multi::array_cref<double, 2>, multi::const_array_ref<double, 2>>{} );

	multi::array_cref<double, 2> d2D_cref{&d2D[0][0], {4, 5}};

	assert( d2D_cref.cdata() == cdata(d2D_cref) );
	assert( d2D_cref.data() == data(d2D_cref) );
	assert( data(d2D_cref) == &d2D[0][0] );
	assert( d2D_cref.num_elements() == num_elements(d2D_cref) );
	assert( num_elements(d2D_cref) == 4*5 );
	assert( d2D_cref.size() == size(d2D_cref) );
	assert( d2D_cref.size() == 4 );
	assert( d2D_cref.size<0>() == size(d2D_cref) );
	assert( d2D_cref.size<0>() == 4);
	assert( d2D_cref.size<1>() == 5 );

	assert( d2D_cref.size<0>() == d2D_cref.size(0) );
	assert( d2D_cref.size<1>() == d2D_cref.size(1) );
	assert( d2D_cref.sizes()[0] == d2D_cref.size(0) );
	assert( d2D_cref.sizes()[1] == d2D_cref.size(1) );
	assert( sizes(d2D_cref) == d2D_cref.sizes() );

	assert( d2D_cref.stride() == d2D_cref.stride(0) );
	assert( d2D_cref.stride() == 5 );
	assert( d2D_cref.stride(1) == 1 );
	assert( strides(d2D_cref) == d2D_cref.strides() );
	assert( strides(d2D_cref)[1] == 1 );

	for(auto i = 0; i != d2D_cref.size<0>() ||!endl(cout); ++i)
		for(auto j = 0; j != d2D_cref.size<1>() ||!endl(cout); ++j)
			cout << d2D_cref[i][j] << ' ';

	for(auto i = 0; i != d2D_cref.size(0) ||!endl(cout); ++i)
		for(auto j = 0; j != d2D_cref.size(1) ||!endl(cout); ++j)
			cout << d2D_cref[i][j] << ' ';

	for(auto i = 0; i != d2D_cref.sizes()[0] ||!endl(cout); ++i)
		for(auto j = 0; j != d2D_cref.sizes()[1] ||!endl(cout); ++j)
			cout << d2D_cref[i][j] << ' ';

	assert( d2D_cref.extension(1) == d2D_cref.extension<1>() );
	assert( d2D_cref.extensions()[1] == d2D_cref.extension(1) );
	assert( extensions(d2D_cref)[1] == d2D_cref.extension(1) );

	for(auto i : d2D_cref.extension<0>()){
		for(auto j : d2D_cref.extension<1>()) cout << d2D_cref[i][j] << ' ';
		cout << '\n';
	}
	cout << '\n';

	for(auto i : d2D_cref.extension(0)){
		for(auto j : d2D_cref.extension(1)) cout << d2D_cref[i][j] << ' ';
		cout << '\n';
	}
	cout << '\n';

	for(auto i : extensions(d2D_cref)[0]){
		for(auto j : extensions(d2D_cref)[1]) cout << d2D_cref[i][j] << ' ';
		cout << '\n';
	}
	cout << '\n';

	multi::array_cref<double, 2> d2D_crefref{d2D_cref.data(), extensions(d2D_cref)};

	assert( d2D_cref.data() == data(d2D_cref) );
	assert( d2D_cref.data()[2] == d2D_cref[0][2] );
	assert( d2D_cref.data()[6] == d2D_cref[1][1] );

	assert( d2D_cref.begin() == begin(d2D_cref) );
	assert( d2D_cref.end() == end(d2D_cref) );
	assert( begin(d2D_cref) != end(d2D_cref) );
	assert( begin(d2D_cref) + size(d2D_cref) == end(d2D_cref) );
	assert( end(d2D_cref) - begin(d2D_cref) == size(d2D_cref) );
	using std::distance;
	assert( distance(begin(d2D_cref), end(d2D_cref)) == size(d2D_cref));

	assert( d2D_cref.sizes() == sizes(d2D_cref) );
	assert( d2D_cref.sizes()[1] == d2D_cref.size<1>() );
	assert( d2D_cref.size(1) == size(*begin(d2D_cref)) );
	assert( size(*begin(d2D_cref)) == 5 );
	assert( distance(begin(d2D_cref)->begin(), begin(d2D_cref)->end()) == begin(d2D_cref)->size() );
	assert( distance(begin(*begin(d2D_cref)), end(*begin(d2D_cref))) == size(*begin(d2D_cref)) );

	assert( size(d2D_cref[0]) == 5 );
//	assert( d2D_cref[0].num_elements() == 5 );

	using std::for_each;
    using namespace std::string_literals; //""s
	for_each(begin(d2D_cref), end(d2D_cref), [](auto&& row){
		for_each(begin(row), end(row), [](auto&& element){
			cout << ' ' << element;
		})("\n"s);
	})("\n"s);

	using std::is_sorted;
	assert( is_sorted(begin(d2D_cref), end(d2D_cref)) ); 

	for(auto it1 = begin(d2D_cref); it1 != end(d2D_cref) ||!endl(cout); ++it1)
		for(auto it2 = it1->begin()   ; it2 != it1->end()    ||!endl(cout); ++it2)
			cout << *it2 << ' ';

//	d2D_cref[3][1] = 3.; // cannot assign to const value

	double const d2D_prime[4][5] {
		{ 0,  1,  2,  3,  4}, 
		{ 5,  6,  7,  8,  9}, 
		{10, 11, 12, 13, 14}, 
		{15, 16, 17, 18, 19}
	};

	multi::array_cref<double, 2> d2D_prime_cref{&d2D_prime[0][0], {4, 5}};
//	multi::array_cref<double, 2> d2D_prime_cref{&d2D_prime[0][0], extensions(d2D_cref)};
	assert( d2D_cref == d2D_prime_cref ); // deep comparison
//	assert( d2D_cref == d2D_prime );
	assert( not(d2D_cref != d2D_prime_cref) );
	assert( not(d2D_cref < d2D_cref) );
	assert( not(d2D_cref > d2D_cref) );
	assert( d2D_cref <= d2D_cref );
	assert( d2D_cref >= d2D_cref );

	double const d2D_null[4][5] {
		{ 0,  0,  0,  0,  0}, 
		{ 0,  0,  0,  0,  0}, 
		{ 0,  0,  0,  0,  0}, 
		{ 0,  0,  0,  0,  0}
	};
	multi::array_cref<double, 2> d2D_null_cref{&d2D_null[0][0], {4, 5}};

	using std::min;
	assert( &min(d2D_null_cref, d2D_cref) == &d2D_null_cref );
	using std::max;
	assert( &max(d2D_null_cref, d2D_cref) == &d2D_cref );
	
	using std::find;
	auto f = find(begin(d2D_cref), end(d2D_cref), d2D_cref[2]);
	assert( f != end(d2D_cref) );
	assert( (*f)[3] == d2D_cref[2][3] );

	using std::find_if;
	auto fif1 = find_if(begin(d2D_cref), end(d2D_cref), [](auto&& e){return e[3] == 8.111;});
	assert( fif1 == end(d2D_cref) );

	using std::find_if;
	auto fif2 = find_if(begin(d2D_cref), end(d2D_cref), [](auto&& e){return e[3] == 8.;});
	assert( fif2 != end(d2D_cref) );
	assert( fif2->operator[](4) == 9. );

	using std::count;
	assert( count(begin(d2D_cref), end(d2D_cref), d2D_prime_cref[3]) == 1 );
	assert( count(begin(d2D_cref), end(d2D_cref), d2D_prime[3]     ) == 1 );

	using std::min_element;
	using std::max_element;

	assert( min_element(begin(d2D_cref), end(d2D_cref)) == begin(d2D_cref) );
	assert( max_element(begin(d2D_cref), end(d2D_cref)) == begin(d2D_cref) + size(d2D_cref) - 1 );

	using std::minmax_element;
	assert( minmax_element(begin(d2D_cref), end(d2D_cref)).first == min_element(begin(d2D_cref), end(d2D_cref)) );
	assert( minmax_element(begin(d2D_cref), end(d2D_cref)).first == min_element(begin(d2D_cref), end(d2D_cref)) );
	decltype(d2D_cref)::const_iterator it; // it{} == it{0} == it{nullptr} = it(0);
//	assert(not it); // there are not null iterators
	assert( std::addressof(it->operator[](0)) == nullptr);
	it = cbegin(d2D_cref);
	assert(it == cbegin(d2D_cref));
	it = decltype(it){};

	std::vector<double>::iterator vit;
	std::list<double>::iterator lit{nullptr};
	assert( std::addressof(*vit) == nullptr );

	std::ptrdiff_t NX = 2;
	std::ptrdiff_t NY = 2;
	std::ptrdiff_t NZ = 2;
	std::vector<double> v(NX*NY*NZ);
	iota(begin(v), end(v), 0.);

	multi::array_cref<double, 3> v3D_cref{v.data(), {NX, NY, NZ}};

	assert( v3D_cref.num_elements() == multi::size_type(v.size()) );
	for(auto i : v3D_cref.extension(0))
		for(auto j : v3D_cref.extension(1))
			for(auto k : v3D_cref.extension(2))
				cout << i << ' ' << j << ' ' << k << ' ' 
					<< v3D_cref[i][j][k] << '\n';

	cout << v3D_cref[9][9][9] << "\n\n";
	cout << *(v3D_cref.begin()->begin()->begin()) << "\n\n";

	assert(d2D_cref.begin() == d2D_cref.begin(0));
	assert(d2D_cref.begin() != d2D_cref.begin(1));
	for(auto it1 = d2D_cref.begin(1); it1 != d2D_cref.end(1)||!endl(cout); ++it1)
		for(auto it2 = it1->begin()   ; it2 != it1->end()   ||!endl(cout); ++it2)
			cout << *it2 << ' ';

	auto print = [](auto&& arr){
		for(auto it1 = arr.begin(); it1 != arr.end()||!endl(cout); ++it1)
			for(auto it2 = it1->begin()   ; it2 != it1->end()   ||!endl(cout); ++it2)
				cout << *it2 << ' ';
	};
	cout << "--\n";
	print(d2D_cref);
	cout << "--\n";
	print(d2D_cref.range({0, 2}));
	cout << "--\n";
	print(d2D_cref.rotated(1).range({0, 2}).rotated(1));
}

