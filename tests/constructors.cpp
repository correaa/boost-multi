#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -O3 $0 -o $0x &&$0x&&rm $0x;exit
#endif

#include "../array.hpp"
#include<vector>
#include<iostream>
#include<numeric>
#include<functional>

namespace multi = boost::multi;

int main(){
 {  multi::array<double, 1> A(10); assert(size(A)==10);
}{//multi::array<double, 1> A({10}); assert(size(A)==1); // warning in clang
}{//multi::array<double, 1> A({10}, double{}); assert(size(A)==10); // warning in clang
}{ multi::array<double, 1> A(10, {}); assert(size(A)==10);
}{ multi::array<double, 1> A(10, double{}); assert(size(A)==10);
}{//multi::array<double, 1> A({10}, double{}); assert(size(A)==10); // warning in clang
}{//multi::array<double, 1> A({10}, 0.); assert(size(A)==10); // warning in clang
}{//multi::array<double, 1> A({10}, {}); assert(size(A)==10); // error ambiguous 
}

{
	multi::array<double, 2> A(multi::index_extensions<2>{8, 8}, 8.);
	assert( size(A) == 8 );
	assert( std::get<0>(sizes(A)) == 8 );
	assert( std::get<1>(sizes(A)) == 8 );

}{
	multi::array<double, 2> A({8, 8}, 8.);
	assert( size(A) == 8 );
	assert( std::get<0>(sizes(A)) == 8 );
	assert( std::get<1>(sizes(A)) == 8 );

}
 {  multi::static_array<double, 1> A     ; assert( empty(A) );
}{  multi::static_array<double, 1> A{}   ; assert( empty(A) );
}{  multi::static_array<double, 1> A = {}; assert( empty(A) ); 
}{  multi::static_array<double, 2> A     ; assert( empty(A) );
}{  multi::static_array<double, 2> A{}   ; assert( empty(A) );
}{  multi::static_array<double, 2> A = {}; assert( empty(A) );
}{  multi::static_array<double, 3> A     ; assert( empty(A) );
}{  multi::static_array<double, 3> A{}   ; assert( empty(A) );
}{  multi::static_array<double, 3> A = {}; assert( empty(A) );
}{  multi::static_array<double, 3> A, B  ; assert( A == B );
}{
	multi::array<double, 1> A1 = {0.0, 1.0, };
	assert( size(A1) == 2 );
	assert( A1[1] == 1.0 );
}

{
	double a[4][5] {
		{ 0,  1,  2,  3,  4}, 
		{ 5,  6,  7,  8,  9}, 
		{10, 11, 12, 13, 14}, 
		{15, 16, 17, 18, 19}
	};
	double b[4][5];
	multi::array_ref<double, 2> A(&a[0][0], {4, 5});
	multi::array_ref
#if not __cpp_deduction_guides
		<double, 2, double const*>
#endif
		B((double const*)&b[0][0], {4, 5})
	;
	rotated(A) = rotated(B);
}

//{
//	std::vector<multi::static_array<double, 2>> v(9, {multi::index_extensions<2>{8, 8}});
//	#if __cpp_deduction_guides
//	std::vector w(9, multi::static_array<double, 2>({8, 8}));
//	#endif
//}

 {  multi::array<double, 1, std::allocator<double>> A{std::allocator<double>{}}; assert( empty(A) );
}{  multi::array<double, 2, std::allocator<double>> A{std::allocator<double>{}}; assert( empty(A) );
}{  multi::array<double, 3, std::allocator<double>> A{std::allocator<double>{}}; assert( empty(A) );
}{ multi::array<double, 1> A(3, {});  assert( size(A)==3 );
}{ multi::array<double, 1> A(3, 99.); assert( size(A)==3 and A[2]==99. );
}{ multi::array<double, 1> A({3});    assert( size(A)==1 and A[0]==3 );
}{// multi::array<double, 1> A({{3}});  assert( size(A)==1 and A[0]==3 );
}{ multi::array<double, 1> A(multi::iextensions<1>{3}); assert(size(A)==3);
}{ multi::array<double, 1> A(multi::array<double, 1>::extensions_type{3}); assert(size(A)==3);
}

#if 0


//#if not defined(__INTEL_COMPILER)
}{	multi::array<double, 1> A({3}); assert( size(A)==1 and A[0]==3. );  // uses init_list
}{	multi::array<double, 1> A({3}); assert( size(A)==1 and A[0]==3. );  // uses init_list
//#endif
//#if not defined(__INTEL_COMPILER)
}{  multi::array<double, 1> A({3.}); assert( size(A)==1 and A[0]==3. ); // uses init_list
}{  multi::array<double, 1> A = {3l}; assert( size(A)==1 and A[0]==3. ); // uses init_list
//#else
}{  multi::array<double, 1> A(3l, 0.); assert( size(A)==3 and A[0]==0. ); // gives warning in clang++ A({3l}, 0.);
//#endif
}{  multi::array<double, 1> A(multi::index_extensions<1>{{0, 3}}); assert( size(A)==3 and A[0]==0 );
#if (!defined(__INTEL_COMPILER)) && (defined(__GNUC) && __GNU_VERSION__ >= 600)
//}{  multi::array<double, 1> A({{0l, 3l}}); cout<<size(A)<<std::endl; assert( size(A)==3 and A[1]==0. ); //uses init_list
#endif
}{  // multi::array<double, 1, std::allocator<double>> A(multi::index_extensions<1>{2}, std::allocator<double>{}); assert( size(A)==2 );
}{  // multi::array<double, 1, std::allocator<double>> A(multi::index_extensions<1>{{0, 3}}, std::allocator<double>{}); assert( size(A)==3 );
}{  // multi::array<double, 1, std::allocator<double>> A(multi::iextensions<1>{2}, std::allocator<double>{}); assert( size(A)==2 );
}{  // multi::array<double, 1, std::allocator<double>> A(multi::iextensions<1>{{0, 3}}, std::allocator<double>{}); assert( size(A)==3 );
#if not defined(__INTEL_COMPILER) and (defined(__GNUC) and __GNU_VERSION >= 600)
}{  multi::array<double, 2> A({2, 3}); assert( num_elements(A)==6 );
#endif
}{  multi::array<double, 2> A(multi::iextensions<2>{2, 3}); assert( num_elements(A)==6 );
//}{  multi::array<double, 2> A({2, 3}); assert( num_elements(A)==6 and size(A)==2 and std::get<1>(sizes(A))==3 );
}{  multi::array<double, 2> A(multi::index_extensions<2>{{0,2}, {0,3}}); assert( num_elements(A)==6 );
#if not defined(__INTEL_COMPILER) and (defined(__GNUC__) and __GNU_VERSION__ >= 600)
}{  multi::array<double, 2, std::allocator<double>> A({2, 3}, std::allocator<double>{}); assert( num_elements(A)==6 );
#endif
}{  multi::array<double, 2, std::allocator<double>> A(multi::iextensions{2, 3}, std::allocator<double>{}); assert( num_elements(A)==6 );
#if not defined(__INTEL_COMPILER) and (defined(__GNUC__) and (__GNU_VERSION >= 600))
}{  multi::array<double, 3> A({2, 3, 4}); assert( num_elements(A)==24 and A[1][2][3]==0 );
#endif
}{  multi::array<double, 3> A(multi::iextensions<3>{2, 3, 4}); assert( num_elements(A)==24 and A[1][2][3]==0 );
#if not defined(__INTEL_COMPILER) and (defined(__GNUC__) and __GNU_VERSION__ >= 600 )
}{  multi::array<double, 3> A({{0, 2}, {0, 3}, {0, 4}}); assert( num_elements(A)==24 and A[1][2][3]==0 );
#endif
}{  multi::array<double, 3> A(multi::iextensions<3>{{0, 2}, {0, 3}, {0, 4}}); assert( num_elements(A)==24 and A[1][2][3]==0 );
#if (not defined(__INTEL_COMPILER)) and (defined(__GNUC__) and __GNU_VERSION__ >= 600)
}{  multi::array<double, 3, std::allocator<double>> A({2, 3, 4}, std::allocator<double>{}); assert( num_elements(A)==24 );
#endif
}

 {  multi::array<double, 1> A(multi::iextensions<1>{3}, 3.1); assert( size(A)==3 and A[1]==3.1 );
}{//multi::array<double, 1> A({3}, 3.1); assert( size(A)==3 and A[1]==3.1 ); // warning in clang
}{//multi::array<double, 1> A({3l}, 3.1); assert( size(A)==3 and A[1]==3.1 ); // warning in clang
}{  multi::array<double, 1> A( {{0,3}}, 3.1); assert( size(A)==3 and A[1]==3.1 );
}{  multi::array<double, 1> A(multi::iextension(3), 3.1); assert( size(A)==3 and A[1]==3.1 );
}{  multi::array<double, 1> A(3l, 3.1); assert( size(A)==3 and A[1]==3.1 );
}{  multi::array<double, 1> A(3, 3.1); assert( size(A)==3 and A[1]==3.1 );
}{  multi::array<double, 1> A({{0, 3}}, 3.1); assert( size(A)==3 and A[1]==3.1 );
#if (not defined(__INTEL_COMPILER)) and (defined(__GNUC__) and __GNU_VERSION__ >=600)
}{  multi::array<double, 2> A({2, 3}, 3.1); assert( num_elements(A)==6 and A[1][2]==3.1 );
#endif
}{  multi::array<double, 2> A(multi::iextensions<2>{2, 3}, 3.1); assert( num_elements(A)==6 and A[1][2]==3.1 );
#if (not defined(__INTEL_COMPILER)) and (defined(__GNUC__) and __GNU_VERSION__ >=600)
}{  multi::array<double, 2> A({{0,2}, {0,3}}, 3.1); assert( num_elements(A)==6 and A[1][2]==3.1 );
#endif
}{  multi::array<double, 2> A(multi::iextensions<2>{{0,2}, {0,3}}, 3.1); assert( num_elements(A)==6 and A[1][2]==3.1 );
#if (not defined(__INTEL_COMPILER)) and (defined(__GNUC__) and __GNU_VERSION__ >=600)
}{  multi::array<double, 3> A({2, 3, 4}, 3.1); assert( num_elements(A)==24 and A[1][2][3]==3.1 );
#endif
}{  multi::array<double, 3> A(multi::iextensions<3>{2, 3, 4}, 3.1); assert( num_elements(A)==24 and A[1][2][3]==3.1 );
}
#endif
}

