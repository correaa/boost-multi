#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXX $0 -o $0x -lboost_timer `pkg-config --libs tbb` &&$0x&&rm $0x;exit
#endif
// Copyright 2018-2024 Alfredo A. Correa

#include <boost/multi/array.hpp>
#include <boost/multi/detail/real.hpp>

#include<algorithm>  // transform
#include<execution>
#include<iostream>
#include<numeric>  // iota

#include<boost/timer/timer.hpp>

namespace multi = boost::multi;

template<class Matrix>
Matrix&& lu_fact(Matrix&& A){
	using multi::size;
	auto m = size(begin(A));// n = size(A[0]);//std::get<1>(sizes(A));
	using std::begin; using std::end; using multi::rotated;
	for(auto k = 0*m; k != std::min(m - 1, size(rotated(A))); ++k){
		auto const& Ak = A[k];
		auto const& Akk = Ak[k];
		std::for_each(std::execution::par, 
			begin(A) + k + 1, end(A), [&](auto&& Ai){
				std::transform(
					begin(Ai)+k+1, end(Ai), begin(Ak)+k+1, begin(Ai)+k+1,
					[z=(Ai[k]/=Akk)](auto a, auto b){return a - z*b;}
				);
			}
		);
	}
	return std::forward<Matrix>(A);
}

template<class Matrix>
Matrix&& lu_fact2(Matrix&& A){
	using multi::size;
	auto const [m, n] = A.sizes();

	for(decltype(m) k = 0; k != m - 1; ++k){
		for(auto i = k + 1; i != m; ++i){
			auto const z = A[i][k]/A[k][k];
			A[i][k] = z;
			std::transform(begin(A[i]) + k + 1, begin(A[i]) + std::max(n, k + 1), A[k].begin() + k + 1, begin(A[i]) + k + 1, [&](auto a, auto b){return a  - z*b;});
		}
	}
	return std::forward<Matrix>(A);
}

template<class Matrix>
Matrix&& lu_fact3(Matrix&& A){
	using multi::size;
	auto const [m, n] = A.sizes();
	for(auto k = 0*m; k != m - 1; ++k){
		auto&& Ak = A[k];
		std::for_each(std::execution::par, begin(A) + k + 1, end(A), [&](auto& Ai){
			auto const z = Ai[k]/Ak[k];
			Ai[k] = z;
			assert( k + 1 <= n );
			for(auto j = k + 1; j < n; ++j) Ai[j] -= z*Ak[j];
		});
	}
	return std::forward<Matrix>(A);
}

using std::cout;
int main(){
	{
		multi::array<boost::multi::float_type, 2> A = {
			{-3.0, 2.0, -4.0},
			{ 0.0, 1.0,  2.0},
			{ 2.0, 4.0,  5.0},
		};
		multi::array<boost::multi::float_type, 1> y = {12.0, 5.0, 2.0};
		boost::multi::float_type AA[3][3];  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) test legacy types
		using std::copy;
		copy( begin(A), end(A), begin(*multi::array_ptr(&AA)) );

		lu_fact(A);
		lu_fact(AA);
		assert( std::equal(begin(A), end(A), begin(*multi::array_ptr(&AA)), end(*multi::array_ptr(&AA))) );
	}
	{
		multi::array<boost::multi::float_type, 2> A({6000, 7000}); 
		//std::iota(begin(A), begin(A) + A.num_elements(), 0.1);
		
		std::transform(begin(A), begin(A) + A.num_elements(), begin(A), [](auto& x){return x/=2.0e6;});
		{
			boost::timer::auto_cpu_timer t;
			lu_fact(A({3000, 6000}, {0, 4000}));
			cout << A[456][123] << std::endl;
		}
	}
}
