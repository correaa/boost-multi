#ifdef COMPILATION_INSTRUCTIONS
c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic $0 -o $0.x -lboost_timer -ltbb && $0.x $@ && rm -f $0.x;exit
#endif

#include "../../multi/array.hpp"

#include<iostream>
#include<vector>
#include<numeric> // iota
#include<algorithm>
#include<execution>

namespace multi = boost::multi;
using std::cout;


template<class Matrix>
Matrix&& lu_fact(Matrix&& A){
	using multi::size;
	auto m = A.size(), n = std::get<1>(sizes(A));
	
	for(auto k = 0*m; k != m - 1; ++k){
		auto const& Ak = A[k];
		if(k > n) continue;
		std::for_each(std::execution::par, begin(A) + k + 1, end(A), [&](auto&& Ai){
//		for(auto i = k + 1; i != m; ++i){
		//	auto&& Ai = A[i];
		//	auto const z = Ai[k]/Ak[k];
		//	auto it = begin(Ai) + 
			Ai[k] /= Ak[k];//z;
		//	assert( k + 1 <= n );
		//	for(auto j = k + 1; j < n; ++j) Ai[j] -= z*Ak[j];
			std::transform(begin(Ai)+k+1, end(Ai), begin(Ak)+k+1, begin(Ai)+k+1, [z=Ai[k]](auto&& a, auto&& b){return a  - z*b;});
		});
	}
	return std::forward<Matrix>(A);
}

template<class Matrix>
Matrix&& lu_fact2(Matrix&& A){
	using multi::size;
	auto m = A.size(), n = std::get<1>(sizes(A));
	
	for(auto k = 0*m; k != m - 1; ++k){
		for(auto i = k + 1; i != m; ++i){
			auto const z = A[i][k]/A[k][k];
			A[i][k] = z;
			std::transform(begin(A[i]) + k + 1, begin(A[i]) + std::max(n, k + 1), A[k].begin() + k + 1, begin(A[i]) + k + 1, [&](auto&& a, auto&& b){return a  - z*b;});
		}
	}
	return std::forward<Matrix>(A);
}

template<class Matrix>
Matrix&& lu_fact3(Matrix&& A){
	using multi::size;
	auto m = A.size(), n = std::get<1>(sizes(A));
	
	for(auto k = 0*m; k != m - 1; ++k){
		auto&& Ak = A[k];
		std::for_each(std::execution::par, begin(A) + k + 1, end(A), [&](auto&& Ai){
			auto const z = Ai[k]/Ak[k];
			Ai[k] = z;
			assert( k + 1 <= n );
			for(auto j = k + 1; j < n; ++j) Ai[j] -= z*Ak[j];
		});
	}
	return std::forward<Matrix>(A);
}

#include <boost/timer/timer.hpp>
#include<iostream>

using std::cout;
int main(){
	{
		multi::array<double, 2> A = {{-3., 2., -4.},{0., 1., 2.},{2., 4., 5.}};
		multi::array<double, 1> y = {12.,5.,2.}; //(M); assert(y.size() == M); iota(y.begin(), y.end(), 3.1);
		lu_fact(A);
	}
	{
		multi::array<double, 2> A({6000, 7000}); std::iota(A.data(), A.data() + A.num_elements(), 0.1);
		std::transform(A.data(), A.data() + A.num_elements(), A.data(), [](auto x){return x/=2.e6;});
	//	std::vector<double> y(3000); std::iota(y.begin(), y.end(), 0.2);
		{
			boost::timer::auto_cpu_timer t;
			lu_fact(A({3000, 6000}, {0, 4000}));
			cout << A[456][123] << std::endl;
		}
	//	cout << y[45] << std::endl;
	}
}

