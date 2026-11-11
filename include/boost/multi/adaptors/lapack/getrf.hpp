// Copyright 2020-2026 Alfredo A. Correa

#ifndef BOOST_MULTI_ADAPTORS_LAPACK_GETRF_HPP
#define BOOST_MULTI_ADAPTORS_LAPACK_GETRF_HPP

#include "boost/multi/adaptors/blas/filling.hpp"
#include "boost/multi/adaptors/lapack/core.hpp"

#include "boost/multi/detail/config/NODISCARD.hpp"
#include "boost/multi/detail/types.hpp"  // for multi::index

#include<cassert>

namespace boost::multi::lapack {

using index = int;

using blas::filling;

template<class Context, class A, class IPIV>
auto getrf(Context&& ctxt, A&& arr, IPIV&& ipiv){
	assert( ipiv.size() == std::min(arr.size(), (~arr).size()) );
	assert( arr.stride() == 1 );
	multi::index const i = std::forward<Context>(ctxt).getrf(static_cast<int>((~arr).size()), static_cast<int>(arr.size()), arr.base(), static_cast<int>((~arr).stride()), std::forward<IPIV>(ipiv).data() );
	// if(i == 0) { return arr(); }
	// else       { return arr({0, i - 1}, {0, i - 1}); }
	if(i == 0) { return std::forward<A>(arr)(); }
	return std::forward<A>(arr)({0, i - 1}, {0, i - 1});
}

template<class Context, class LU, class IPIV, class B>
auto getrs(Context&& ctxt, LU const& lu, IPIV const& ipiv, B&& barr) -> B&& {
	assert( lu.size() == (~lu).size() );
	assert( lu.stride() == 1 );
	assert( ipiv.size() >= lu.size() );
	assert( barr.stride() == 1 );
	std::forward<Context>(ctxt).getrs(
		'N', static_cast<int>(lu.size()),
		static_cast<int>((~barr).size()),
		lu.base(),
		static_cast<int>((~lu).stride()),
		ipiv.data(),
		barr.base(),
		static_cast<int>((~barr).stride())
	);
	return std::forward<B>(barr);
}

template<class Context, class LU, class IPIV, class V>
void getrs_one(Context&& ctxt, LU const& lu, IPIV const& ipiv, V&& barr){
	assert( lu.size() == (~lu).size() );
	assert( lu.stride() == 1 );
	assert( barr.stride() == 1 );
	std::forward<Context>(ctxt).getrs('N', static_cast<int>(lu.size()), 1, lu.base(), static_cast<int>((~lu).stride()), ipiv.data(), std::forward<V>(barr).base(), static_cast<int>(lu.size()));
}


template<class A, class IPIV>
auto getrf(A&& arr, IPIV&& ipiv){return getrf(::lapack::context{}, std::forward<A>(arr), std::forward<IPIV>(ipiv));}

template<class LU, class IPIV, class B>
auto getrs(LU const& lu, IPIV const& ipiv, B&& barr) -> decltype(auto) {return getrs(::lapack::context{}, lu, ipiv, std::forward<B>(barr));}

template<class LU, class IPIV, class B>
void getrs_one(LU const& lu, IPIV const& ipiv, B&& barr){return getrs_one(::lapack::context{}, lu, ipiv, std::forward<B>(barr));}

}  // namespace boost::multi::lapack

#endif


