#include <benchmark/benchmark.h>
#include "../array.hpp"

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include<fstream>
#include<sstream>

#include<random>

constexpr auto N = 64;

namespace multi = boost::multi;

template<class Ar>
void BM_oserialization(benchmark::State& st) {
	auto const A = [] {
	    std::random_device rd;
	    std::mt19937 mt(rd());
	    std::uniform_real_distribution<double> dist(-1.0, +1.0);

		multi::array<double, 4> A({N, N, N, N});
		std::generate(begin(elements(A)), end(elements(A)), [&]{return dist(mt);});
		return A;
	}();

    benchmark::ClobberMemory();
	for(auto _  : st) {
		std::ofstream fs{"file"};
		Ar xa{fs};
		xa<< BOOST_SERIALIZATION_NVP(A);
		fs.flush();
		benchmark::DoNotOptimize(A);
	    benchmark::ClobberMemory();
	}
	st.SetBytesProcessed(st.iterations()*A.num_elements()*sizeof(double));
	st.SetItemsProcessed(st.iterations()*A.num_elements());
}

template<class Ar>
void BM_iserialization(benchmark::State& st) {
	multi::array<double, 4> A({N, N, N, N});

    benchmark::ClobberMemory();
	for(auto _  : st) {
		std::ifstream fs{"file"};
		Ar xa{fs};
		xa>> BOOST_SERIALIZATION_NVP(A);
		benchmark::DoNotOptimize(A);
	    benchmark::ClobberMemory();
	}
	st.SetBytesProcessed(st.iterations()*A.num_elements()*sizeof(double));
	st.SetItemsProcessed(st.iterations()*A.num_elements());
}

BENCHMARK_TEMPLATE(BM_oserialization, boost::archive::xml_oarchive   );
BENCHMARK_TEMPLATE(BM_iserialization, boost::archive::xml_iarchive   );

BENCHMARK_TEMPLATE(BM_oserialization, boost::archive::text_oarchive  );
BENCHMARK_TEMPLATE(BM_iserialization, boost::archive::text_iarchive  );

BENCHMARK_TEMPLATE(BM_oserialization, boost::archive::binary_oarchive);
BENCHMARK_TEMPLATE(BM_iserialization, boost::archive::binary_iarchive);

void BM_gzip_oserialization(benchmark::State& st) {
	auto const A = []{
	    std::random_device rd;
	    std::mt19937 mt(rd());
	    std::uniform_real_distribution<double> dist(-1.0, +1.0);

		multi::array<double, 4> A({N, N, N, N});
		std::generate(begin(elements(A)), end(elements(A)), [&]{return dist(mt);});

		return A;
	}();

    benchmark::ClobberMemory();
	for(auto _  : st) {
		std::ofstream ofs{"file.gz"};
		boost::iostreams::filtering_ostream out;
		out.push(boost::iostreams::gzip_compressor{boost::iostreams::zlib::best_speed});
		out.push(ofs);
		boost::archive::xml_oarchive xa(out);
		xa<< BOOST_SERIALIZATION_NVP(A);
		benchmark::DoNotOptimize(A);
	    benchmark::ClobberMemory();
	}
	st.SetBytesProcessed(st.iterations()*A.num_elements()*sizeof(double));
	st.SetItemsProcessed(st.iterations()*A.num_elements());
}
BENCHMARK(BM_gzip_oserialization);

void BM_gzip_iserialization(benchmark::State& st) {
	multi::array<double, 4> A({N, N, N, N});

    benchmark::ClobberMemory();
	for(auto _  : st) {
		std::ifstream ifs{"file.gz"};
		boost::iostreams::filtering_istream in;
		in.push(boost::iostreams::gzip_decompressor{});
		in.push(ifs);
		boost::archive::xml_iarchive xa(in);
		xa>> BOOST_SERIALIZATION_NVP(A);
		benchmark::DoNotOptimize(A);
	    benchmark::ClobberMemory();
	}
	st.SetBytesProcessed(st.iterations()*A.num_elements()*sizeof(double));
	st.SetItemsProcessed(st.iterations()*A.num_elements());
}
BENCHMARK(BM_gzip_iserialization);

BENCHMARK_MAIN();
