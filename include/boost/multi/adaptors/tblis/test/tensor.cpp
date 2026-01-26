// Copyright 2019-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/io.hpp>


#include <boost/core/lightweight_test.hpp>
// #include <boost/multi/adaptors/tblis.hpp>

#include <numeric>

namespace boost::multi::tblis {
	template<multi::dimensionality_t D>
	class tensor {
	 public:
		::tblis::tblis_tensor impl_;
		std::array<::tblis::len_type, D> Alens_;
		std::array<::tblis::stride_type, D> Astrides_;

	 public:
		tensor(tensor const&) = delete;
		tensor(tensor&&) = delete;

		template<class Array>
		tensor(Array&& AA)
		: Alens_{apply([](auto... el) { return std::array{static_cast<::tblis::len_type>(el)...}; }, AA.sizes())}
		, Astrides_{apply([](auto... el) { return std::array{static_cast<::tblis::stride_type>(el)...}; }, AA.strides())}
		{
			::tblis::tblis_init_tensor_d(
				&impl_, 4, Alens_.data(),
				AA.base(), Astrides_.data()
			);
		}

		auto operator&() const { return &impl_; }
		auto get() { return &impl_; }
	};

	// void mult(
	// 	tensor const& At, std::string_view A_indices,
	// 	tensor const& Bt, std::string_view B_indices,
	// 	tensor& Ct, std::string_view C_indices
	// ) {
	// 	::tblis::tblis_tensor_mult(NULL, NULL, At.get(), A_indices.data(), Bt.get(), B_indices.data(), const_cast<::tblis::tblis_tensor *>(Ct.get()), C_indices.data());
	// }
}

namespace multi = boost::multi;

int main() {
	auto as = multi::extension_t<>(8);
	auto bs = multi::extension_t<>(10);
	auto cs = multi::extension_t<>(2);
	auto ds = multi::extension_t<>(7);
	auto es = multi::extension_t<>(5);
	auto fs = multi::extension_t<>(9);

	multi::array<double, 4> AA({es, cs, fs, bs}, 0.0);
	std::iota(AA.elements().begin(), AA.elements().end(), 0.0);

	multi::array<double, 4> BB({as, fs, es, ds}, 0.0);
	std::iota(BB.elements().begin(), BB.elements().end(), 0.0);

	multi::array<double, 4> CC({as, bs, cs, ds}, 0.0);

	multi::tblis::tensor<4> At(AA);

	tblis::tblis_tensor B;
	tblis::tblis_init_tensor_d(
		&B, 4, (tblis::len_type[]){8, 9, 5, 7},  // {7, 5, 9, 8},
		BB.base(), (tblis::stride_type[]){315, 35, 7, 1}  // {1, 7, 35, 315}
	);

	// double data_C[7*2*10*8];
	tblis::tblis_tensor C;
	tblis::tblis_init_tensor_d(
		&C, 4, (tblis::len_type[]){8, 10, 2, 7},  // {7, 2, 10, 8},
		CC.base(), (tblis::stride_type[]){140, 14, 7, 1}  // {1, 7, 14, 140}
	);

	// this computes C[abcd] += A[cebf] B[afed]
	tblis::tblis_tensor_mult(NULL, NULL, &At, "ecfb", &B, "afed", &C, "abcd");

	auto const C_gold = [&]{
		multi::array<double, 4> _({as, bs, cs, ds}, 0.0);
		// this computers C_check[abcd] += A[cebf] B[afed]
		for(auto a : as) {
			for(auto b : bs) {
				for(auto c : cs) {
					for(auto d : ds) {
						_[a][b][c][d] = 0.0;
						for(auto e : es) {
							for(auto f : fs) {
								_[a][b][c][d] += AA[e][c][f][b]*BB[a][f][e][d];
							}
						}
						BOOST_TEST( _[a][b][c][d] == CC[a][b][c][d] );
					}
				}
			}
		}
		return _;
	}();

	BOOST_TEST( CC[1][2][1][4] == C_gold[1][2][1][4] );

	// for(auto i : C_gold.elements().extension()) {
	// 	std::cout << C_gold.elements()[i] << "==" << C.elements()[i] << "?";

	// 	BOOST_TEST( C_gold.elements()[i] == C.elements()[i] );
	// }
//	BOOST_TEST( std::equal(C_gold.elements().begin(), C_gold.elements().end(), C.elements().begin()) );


	// BOOST_AUTO_TEST_CASE(blis_matrix)/ multi::tblis::tensor At(A);
	// {
	// 	namespace tblis = multi::tblis;
	// 	using namespace multi::tblis;

	// 	auto const A = []{
	// 		multi::array<double, 2> _({5, 2}); std::iota(_.elements().begin(), _.elements().end(), 0.);
	// 		return _;
	// 	}();

	// 	auto const B = []{
	// 		multi::array<double, 2> _({2, 7}); std::iota(_.elements().begin(), _.elements().end(), 0.);
	// 		return _;
	// 	}();

	// 	// now the check
	// 	multi::array<double, 2> C_gold({5, 7}, 0.);

	// 	assert( extension(C_gold) == extension(A) );
	// 	assert( extension(C_gold[0]) == extension(B[0]) );
	// 	assert( extension(B) == extension(A[0]) );
	// 	for(auto a : extension(C_gold)){
	// 		for(auto b : extension(C_gold[0])){
	// 			for(auto c : extension(B)){
	// 				C_gold[a][b] += A[a][c]*B[c][b];
	// 			}
	// 		}
	// 	}

	// 	{
	// 		multi::array<double, 2> C({5, 7}, 0.);
	// 		// C[abcd] += A[cebf] B[afed]
	// 		tblis::mult(tblis::matrix(A), tblis::matrix(B), tblis::matrix(C));
	// 		BOOST_REQUIRE( C_gold == C );
	// 	}
	// 	{
	// 		multi::array<double, 2> C({5, 7}, 0.);
	// 		tblis::mult(tblis::tensor(A), "ac", tblis::tensor(B), "cb", tblis::tensor(C), "ab");
	// 		BOOST_REQUIRE( C_gold == C );
	// 	}
	// 	{
	// 		multi::array<double, 2> C({5, 7}, 0.);
	// 		tblis::mult(tblis::tensor(A)["ac"], tblis::tensor(B)["cb"], tblis::tensor(C)["ab"]);
	// 		BOOST_REQUIRE( C_gold == C );
	// 	}
	// 	{
	// 		multi::array<double, 2> C({5, 7}, 0.);
	// 		tblis::mult(tblis::tensor(A)["ac"], tblis::tensor(B)["cb"], tblis::tensor(C)["ab"]);
	// 		BOOST_REQUIRE( C_gold == C );
	// 	}
	// 	{
	// 		multi::array<double, 2> C({5, 7}, 0.);
	// 		using namespace tblis::indices;
	// 		tblis::mult(tblis::tensor(A)(a, c), tblis::tensor(B)(c, b), tblis::tensor(C)(a, b));
	// 		BOOST_REQUIRE( C_gold == C );
	// 	}
	// 	{
	// 		multi::array<double, 2> C({5, 7}, 0.);
	// 		using namespace tblis::indices;
	// 		tblis::mult(A(a, c), B(c, b), C(a, b));
	// 	//  BOOST_REQUIRE( C_gold == C );
	// 	}
	// }

	// BOOST_AUTO_TEST_CASE(tblis_tensor)
	// {
	// 	namespace multi = boost::multi;
	// 	namespace tblis = multi::tblis;

	// 	auto const A = []{
	// 		multi::array<double, 4> A({2, 5, 10, 9});
	// 		std::iota(A.data_elements(), A.data_elements() + A.num_elements(), 0.);
	// 		return A;
	// 	}();

	// 	auto const B = []{
	// 		multi::array<double, 4> B({8, 9, 5, 7});
	// 		std::iota(B.data_elements(), B.data_elements() + B.num_elements(), 0.);
	// 		return B;
	// 	}();

	// 	auto const C_gold = [&A, &B]{
	// 		multi::array<double, 4> _({8, 10, 2, 7}, 0.);
	// 		// this computers C_check[abcd] += A[cebf] B[afed]
	// 		for(auto a = 0; a != 8; ++a){
	// 			for(auto b = 0; b != 10; ++b){
	// 				for(auto c = 0; c != 2; ++c){
	// 					for(auto d = 0; d != 7; ++d){

	// 						for(auto e = 0; e != 5; ++e){
	// 							for(auto f = 0; f != 9; ++f){
	// 								_[a][b][c][d] += A[c][e][b][f]*B[a][f][e][d];
	// 							}
	// 						}

	// 					}
	// 				}
	// 			}
	// 		}
	// 		return _;
	// 	}();

	// 	{
	// 		multi::array<double, 4> C({8, 10, 2, 7}, 0.);
	// 		{
	// 			using namespace tblis::indices;
	// 			tblis::mult( A(c, e, b, f), B(a, f, e, d), C(a, b, c, d) );
	// 		}
	// 		BOOST_REQUIRE( C_gold == C );
	// 	}
	// 	#if defined(__clang__)
	// 	{
	// 		multi::array<double, 4> C({8, 10, 2, 7}, 0.);
	// 		{
	// 			using namespace tblis::indices::greek;
	// 			tblis::mult( A(γ, ε, β, ζ), B(α, ζ, ε, δ), C(α, β, γ, δ) );
	// 		}
	// 		BOOST_REQUIRE( C_gold == C );
	// 	}
	// 	#endif
	// 	{
	// 		multi::array<double, 4> C({8, 10, 2, 7}, 0.);
	// 		{
	// 			using namespace tblis::indices;
	// 			tblis::mult( tblis::tensor(A)(c, e, b, f), tblis::tensor(B)(a, f, e, d), tblis::tensor(C)(a, b, c, d) );
	// 		}
	// 		BOOST_REQUIRE( C_gold == C );
	// 	}

	// }
	return boost::report_errors();
}
#else
int main() {}
#endif
