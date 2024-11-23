// Copyright 2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 10.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/adaptors/mpi.hpp>
#include <boost/multi/array.hpp>

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include <boost/core/lightweight_test.hpp>

#include <iostream>  // for std::cout
#include <vector>

namespace boost::multi::mpi {

template<class F, F Fun>
class call_base {
//  F func_;

 public:
	// constexpr explicit call(F func) : func_{std::move(func)} {}

	template<class... Ts, decltype(Fun(std::declval<Ts&&>()...))* = nullptr>
	auto operator()(Ts&&... ts) const -> decltype(auto) {
		return Fun(std::forward<Ts>(ts)...);
	}

	template<
		class Message, class... Ts,
		class B                                                                                                                  = decltype(const_cast<void*>(std::declval<Message&>().buffer())),
		class C                                                                                                                  = decltype(std::declval<Message&>().count()),
		class D                                                                                                                  = decltype(std::declval<Message&>().datatype()),
		decltype(std::declval<F const&>()(std::declval<B&>(), std::declval<C&>(), std::declval<D&>(), std::declval<Ts&&>()...))* = nullptr>
	auto operator()(Message&& msg, Ts&&... ts) const -> decltype(auto) {
		auto buffer   = msg.buffer();
		auto count    = msg.count();
		auto datatype = msg.datatype();
		return this->operator()(buffer, count, datatype, std::forward<Ts>(ts)...);
	}

	template<
		class Message, class... Ts,
		class B                                                                                                                  = decltype(const_cast<void*>(std::declval<Message&>().buffer())),
		class C                                                                                                                  = decltype(std::declval<Message&>().count()),
		class D                                                                                                                  = decltype(std::declval<Message&>().datatype()),
		decltype(std::declval<F const&>()(std::declval<B&>(), std::declval<C&>(), std::declval<D&>(), std::declval<Ts&&>()..., MPI_STATUS_IGNORE))* = nullptr>
	auto operator()(Message&& msg, Ts&&... ts) const -> decltype(auto) {
		auto buffer   = msg.buffer();
		auto count    = msg.count();
		auto datatype = msg.datatype();
		return this->operator()(buffer, count, datatype, std::forward<Ts>(ts)..., MPI_STATUS_IGNORE);
	}

	template<
		class Value, class... Ts,
		class = decltype(std::declval<call_base const&>()(multi::mpi::message(std::declval<Value&&>()), std::declval<Ts&&>()...)
		)>
	auto operator()(Value&& val, Ts&&... ts) const -> decltype(auto) {
		return operator()(multi::mpi::message(std::forward<Value>(val)), std::forward<Ts>(ts)...);
	}

	template<class... Ts, decltype(std::declval<call_base const&>()(std::declval<Ts&&>()..., MPI_STATUS_IGNORE))* = nullptr>
	auto operator()(Ts&&... ts) const -> decltype(auto) {
		return operator()(std::forward<Ts>(ts)..., MPI_STATUS_IGNORE);
	}
};


template<class F, F Fun>
struct call : call_base<F, Fun> {
	using call_base<F, Fun>::operator();
};

template<>
class call<int(void*, int, ompi_datatype_t*, int, int, ompi_communicator_t*, ompi_status_public_t*), MPI_Recv>
: call_base<int(void*, int, ompi_datatype_t*, int, int, ompi_communicator_t*, ompi_status_public_t*), MPI_Recv> {
	using F = decltype(MPI_Recv);

 public:
	using call_base<decltype(MPI_Recv), MPI_Recv>::operator();

	template<class... Ts, class Comm, decltype(std::declval<decltype(MPI_Recv)>()(std::declval<Ts&&>()..., MPI_ANY_TAG, std::declval<Comm>()))* = nullptr>
	auto operator()(Ts&&... ts, Comm comm) const -> decltype(auto) {
		return call_base<decltype(MPI_Recv), MPI_Recv>::operator()(std::forward<Ts>(ts)..., MPI_ANY_TAG, comm);
	}
};

// template<>
// class call<decltype(MPI_Send), MPI_Send>
// : call_base<decltype(MPI_Send), MPI_Send> {
//  public:
//  using call_base<decltype(MPI_Send), MPI_Send>::operator();

//  template<class Range>  // , decltype(std::declval<decltype(MPI_Send)>()(std::declval<Ts&&>()..., MPI_ANY_TAG, std::declval<MPI_Comm>()))* = nullptr>
//  auto operator()(Range&& ts, int dest, MPI_Comm& comm) const -> decltype(auto) {
//    return call_base<decltype(MPI_Send), MPI_Send>::operator()(std::forward<Range>(ts), dest, MPI_ANY_TAG, comm);
//  }
// };

constexpr static auto send    = call<decltype(MPI_Send), MPI_Send>{};
constexpr static auto receive = call<decltype(MPI_Recv), MPI_Recv>{};

}  // end namespace boost::multi::mpi

namespace multi = boost::multi;

namespace {
void test_single_number(MPI_Comm comm) {
	int world_rank;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_rank(comm, &world_rank);
	int world_size;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_size(comm, &world_size);

	BOOST_TEST(world_size > 1);

	int number = 0;
	if(world_rank == 0) {
		number = -1;
		MPI_Send(&number, 1, MPI_INT, 1, 0, comm);
	} else if(world_rank == 1) {
		MPI_Recv(&number, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
		BOOST_TEST(number == -1);
	}
	{
		std::vector<int> vv(3, 99);  // NOLINT(fuchsia-default-arguments-calls)
		if(world_rank == 0) {
			vv = {1, 2, 3};
			MPI_Send(vv.data(), static_cast<int>(vv.size()), MPI_INT, 1, 0, comm);
		} else if(world_rank == 1) {
			MPI_Recv(vv.data(), static_cast<int>(vv.size()), MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
			BOOST_TEST( vv == std::vector<int>({1, 2, 3}) );  // NOLINT(fuchsia-default-arguments-calls)
		}
	}
}

void test_1d(MPI_Comm comm) {
	int world_rank;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_rank(comm, &world_rank);
	int world_size;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_size(comm, &world_size);
	{
		if(world_rank == 0) {
			multi::array<int, 1> const AA = multi::array<int, 1>({1, 2, 3, 4, 5, 6});
			auto const&&               BB = AA.strided(2);
			BOOST_TEST(( BB == multi::array<double, 1>({1, 3, 5}) ));

			auto const B_data = multi::mpi::data(BB.begin());

			MPI_Send(B_data.buffer(), static_cast<int>(BB.size()), B_data.datatype(), 1, 0, comm);
		} else if(world_rank == 1) {
			multi::array<int, 1> CC(3, 99);  // NOLINT(misc-const-correctness)

			auto const& C_msg = multi::mpi::message(CC.elements());

			MPI_Recv(C_msg.buffer(), C_msg.count(), C_msg.datatype(), 0, 0, comm, MPI_STATUS_IGNORE);
			BOOST_TEST(( CC == multi::array<double, 1>({1, 2, 3}) ));
		}
	}
	std::cout << world_rank << '\n';
	{
		if(world_rank == 0) {
			auto const  AA = multi::array<int, 1>({1, 2, 3, 4, 5, 6});
			auto const& BB = AA.strided(2);
			BOOST_TEST(( BB == multi::array<int, 1>({1, 3, 5}) ));

			auto const B_data = multi::mpi::data(BB.begin());

			MPI_Send(B_data.buffer(), static_cast<int>(BB.size()), B_data.datatype(), 1, 0, comm);
		} else if(world_rank == 1) {
			multi::array<int, 1> CC(3, 99);  // NOLINT(misc-const-correctness)

			auto const C_msg = multi::mpi::message(CC);

			MPI_Recv(C_msg.buffer(), C_msg.count(), C_msg.datatype(), 0, 0, comm, MPI_STATUS_IGNORE);
			BOOST_TEST(( CC == multi::array<double, 1>({1, 2, 3}) ));
		}
	}
	std::cout << world_rank << '\n';
	{
		if(world_rank == 0) {
			auto const  AA = multi::array<int, 1>({1, 2, 3, 4, 5, 6});
			auto const& BB = AA.strided(2);
			BOOST_TEST(( BB == multi::array<int, 1>({1, 3, 5}) ));

			MPI_Datatype B_type;  // NOLINT(cppcoreguidelines-init-variables)
			multi::mpi::create_subarray(BB.layout(), MPI_INT, &B_type);
			MPI_Type_commit(&B_type);
			MPI_Send(BB.base(), 1, B_type, 1, 0, comm);
			MPI_Type_free(&B_type);

		} else if(world_rank == 1) {
			multi::array<int, 1> CC(3, 99);  // NOLINT(misc-const-correctness)

			auto const C_msg = multi::mpi::message{CC.base(), CC.layout(), MPI_INT};

			MPI_Recv(C_msg.buffer(), C_msg.count(), C_msg.datatype(), 0, 0, comm, MPI_STATUS_IGNORE);
			BOOST_TEST(( CC == multi::array<double, 1>({1, 3, 5}) ));
		}
	}
}

void test_2d(MPI_Comm comm) {
	int world_rank;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_rank(comm, &world_rank);
	int world_size;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_size(comm, &world_size);

	{
		if(world_rank == 0) {
			auto const AA = multi::array<int, 2>({
				{1, 2, 3},
				{4, 5, 6}
			});

			auto const& BB = AA({0, 2}, {1, 3});
			BOOST_TEST(( BB == multi::array<double, 2>({{2, 3}, {5, 6}}) ));

			auto const B_msg = multi::mpi::message(BB.elements());

			MPI_Send(B_msg.buffer(), B_msg.count(), B_msg.datatype(), 1, 0, comm);
		} else if(world_rank == 1) {
			multi::array<int, 2> CC({2, 2}, 99);

			auto const C_sk = multi::mpi::skeleton(CC.elements().layout(), MPI_INT);

			MPI_Recv(CC.base(), C_sk.count(), C_sk.datatype(), 0, 0, comm, MPI_STATUS_IGNORE);
			std::cout << CC[0][0] << ' ' << CC[0][1] << '\n'
					  << CC[1][0] << ' ' << CC[1][1] << '\n';

			BOOST_TEST(( CC == multi::array<double, 2>({{2, 3}, {5, 6}}) ));
		}
	}
	{
		if(world_rank == 0) {
			auto const AA = multi::array<int, 2>({
				{1, 2, 3},
				{4, 5, 6}
			});

			auto const& BB = AA({0, 2}, {1, 3});
			BOOST_TEST(( BB == multi::array<double, 2>({{2, 3}, {5, 6}}) ));

			auto const B_msg = multi::mpi::message(BB.elements());

			MPI_Send(B_msg.buffer(), B_msg.count(), B_msg.datatype(), 1, 0, comm);
		} else if(world_rank == 1) {
			multi::array<int, 2> CC({2, 2}, 99);

			auto const C_sk = multi::mpi::skeleton(CC.elements().layout(), MPI_INT);

			MPI_Recv(CC.base(), C_sk.count(), C_sk.datatype(), 0, 0, comm, MPI_STATUS_IGNORE);
			std::cout << CC[0][0] << ' ' << CC[0][1] << '\n'
					  << CC[1][0] << ' ' << CC[1][1] << '\n';

			BOOST_TEST(( CC == multi::array<double, 2>({{2, 3}, {5, 6}}) ));
		}
	}
	{
		if(world_rank == 0) {
			auto const AA = multi::array<int, 2>{
				{1, 2, 3},
				{4, 5, 6}
			};
			auto const& BB = AA({0, 2}, {1, 3});
			BOOST_TEST(( BB == multi::array<int, 2>({{2, 3}, {5, 6}}) ));

			MPI_Datatype B_type;  // NOLINT(cppcoreguidelines-init-variables)
			multi::mpi::create_subarray(BB.layout(), MPI_INT, &B_type);
			MPI_Type_commit(&B_type);
			MPI_Send(BB.base(), 1, B_type, 1, 0, comm);
			MPI_Type_free(&B_type);
		} else if(world_rank == 1) {
			multi::array<int, 2> CC({2, 2}, 99);

			auto const C_sk = multi::mpi::skeleton(CC.layout(), MPI_INT);

			MPI_Recv(CC.base(), C_sk.count(), C_sk.datatype(), 0, 0, comm, MPI_STATUS_IGNORE);
			std::cout << CC[0][0] << ' ' << CC[0][1] << '\n'
					  << CC[1][0] << ' ' << CC[1][1] << '\n';

			BOOST_TEST(( CC == multi::array<double, 2>({{2, 3}, {5, 6}}) ));
		}
	}
	{
		if(world_rank == 0) {
			auto const AA = multi::array<int, 2>({
				{1, 2, 3},
				{4, 5, 6}
			});

			auto const& BB = AA({0, 2}, {1, 3});
			BOOST_TEST(( BB == multi::array<double, 2>({{2, 3}, {5, 6}}) ));

			auto const B_msg = multi::mpi::message(BB.elements());

			MPI_Send(B_msg.buffer(), B_msg.count(), B_msg.datatype(), 1, 0, comm);
		} else if(world_rank == 1) {
			multi::array<int, 2> CC({2, 2}, 99);

			auto const C_sk = multi::mpi::skeleton(CC.elements().layout(), MPI_INT);

			MPI_Recv(CC.base(), C_sk.count(), C_sk.datatype(), 0, 0, comm, MPI_STATUS_IGNORE);
			std::cout << CC[0][0] << ' ' << CC[0][1] << '\n'
					  << CC[1][0] << ' ' << CC[1][1] << '\n';

			BOOST_TEST(( CC == multi::array<double, 2>({{2, 3}, {5, 6}}) ));
		}
	}
}

void test_2d_int(MPI_Comm comm) {
	int world_rank;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_rank(comm, &world_rank);
	int world_size;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_size(comm, &world_size);

	{
		if(world_rank == 0) {
			auto const AA = multi::array<int, 2>({
				{1, 2, 3},
				{4, 5, 6}
			});

			auto const& BB = AA({0, 2}, {1, 3});
			BOOST_TEST(( BB == multi::array<int, 2>({{2, 3}, {5, 6}}) ));

			auto const B_msg = multi::mpi::message(BB.elements());

			MPI_Send(B_msg.buffer(), B_msg.count(), B_msg.datatype(), 1, 0, comm);
		} else if(world_rank == 1) {
			multi::array<int, 2> CC({2, 2}, 99);

			auto const C_sk = multi::mpi::skeleton<int>(CC.layout());

			MPI_Recv(CC.base(), C_sk.count(), C_sk.datatype(), 0, 0, comm, MPI_STATUS_IGNORE);
			std::cout << CC[0][0] << ' ' << CC[0][1] << '\n'
					  << CC[1][0] << ' ' << CC[1][1] << '\n';

			BOOST_TEST(( CC == multi::array<int, 2>({{2, 3}, {5, 6}}) ));
		}
	}
}

void test_2d_double(MPI_Comm comm) {
	int world_rank;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_rank(comm, &world_rank);
	int world_size;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_size(comm, &world_size);

	{
		if(world_rank == 0) {
			auto const AA = multi::array<double, 2>({
				{1, 2, 3},
				{4, 5, 6}
			});

			auto const& BB = AA({0, 2}, {1, 3});
			BOOST_TEST(( BB == multi::array<double, 2>({{2, 3}, {5, 6}}) ));

			auto const& B_msg = multi::mpi::message(BB.elements());

			MPI_Send(B_msg.buffer(), B_msg.count(), B_msg.datatype(), 1, 0, comm);
		} else if(world_rank == 1) {
			multi::array<double, 2> CC({2, 2}, 99.0);

			auto const C_sk = multi::mpi::skeleton<double>(CC.layout());

			MPI_Recv(CC.base(), C_sk.count(), C_sk.datatype(), 0, 0, comm, MPI_STATUS_IGNORE);
			std::cout << CC[0][0] << ' ' << CC[0][1] << '\n'
					  << CC[1][0] << ' ' << CC[1][1] << '\n';

			BOOST_TEST(( CC == multi::array<double, 2>({{2.0, 3.0}, {5.0, 6.0}}) ));
		}
	}
}

}  // namespace

auto main() -> int {  // NOLINT(bugprone-exception-escape)
	MPI_Init(nullptr, nullptr);

	int world_rank;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int world_size;  // NOLINT(cppcoreguidelines-init-variables)
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	std::cout << "size " << world_size << '\n';
	// int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  // NOLINT(cppcoreguidelines-init-variables)
	// int world_size; MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // NOLINT(cppcoreguidelines-init-variables)

	test_single_number(MPI_COMM_WORLD);
	test_1d(MPI_COMM_WORLD);

	{
		multi::array<int, 1> AA({3}, 99);
		if(world_rank == 0) {
			AA = multi::array<double, 1>({1, 2, 3});
			MPI_Send(AA.base(), static_cast<int>(AA.size()), MPI_INT, 1, 0, MPI_COMM_WORLD);
		} else if(world_rank == 1) {
			MPI_Recv(AA.base(), static_cast<int>(AA.size()), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			BOOST_TEST(( AA == boost::multi::array<double, 1>({1, 2, 3}) ));
		}
	}
	{
		if(world_rank == 0) {
			auto const AA = multi::array<int, 2>({
				{1, 2, 3},
				{4, 5, 6}
			});

			auto const& BB = AA({0, 2}, {1, 3});
			BOOST_TEST(( BB == multi::array<double, 2>({{2, 3}, {5, 6}}) ));

			auto const B_sk = multi::mpi::skeleton(BB.layout(), MPI_INT);

			MPI_Send(BB.base(), B_sk.count(), B_sk.datatype(), 1, 0, MPI_COMM_WORLD);
		} else if(world_rank == 1) {
			multi::array<int, 2> CC({2, 2}, 99);

			auto const C_sk = multi::mpi::skeleton(CC.elements().layout(), MPI_INT);

			MPI_Recv(CC.base(), C_sk.count(), C_sk.datatype(), 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			BOOST_TEST(( CC == multi::array<double, 2>({{2, 3}, {5, 6}}) ));
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	test_2d(MPI_COMM_WORLD);
	test_2d_int(MPI_COMM_WORLD);
	test_2d_double(MPI_COMM_WORLD);

	MPI_Finalize();

	return boost::report_errors();
}
