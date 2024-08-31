#include <boost/multi/array.hpp>

#include <mpi.h>

#include <boost/core/lightweight_test.hpp>

#include <cassert>   // for assert
#include <iostream>  // for std::cout
#include <vector>

namespace boost::multi::mpi {

class data {
	void*        buf_;
	MPI_Datatype datatype_;

 public:
	template<class It>
	explicit data(It first)                                            // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
	: buf_{const_cast<void*>(static_cast<void const*>(first.base()))}  // NOLINT(cppcoreguidelines-pro-type-const-cast)
	{
		MPI_Type_vector(
			1, 1,
			first.stride(),
			MPI_INT, &datatype_
		);

		MPI_Type_commit(&datatype_);  // type cannot be used until committed, in communication operations at least
	}

	data(data const&) = delete;
	data(data&&)      = delete;

	auto operator=(data const&) = delete;
	auto operator=(data&&)      = delete;

	~data() { MPI_Type_free(&datatype_); }

	auto buffer() const { return buf_; }
	auto type() const { return datatype_; }
};

template<class Size = int>
class skeleton {
	Size         count_;
	MPI_Datatype datatype_;

	skeleton() : datatype_{MPI_DATATYPE_NULL} {}
	skeleton(skeleton&& other) noexcept
	: count_{other.count_}, datatype_{std::exchange(other.datatype_, MPI_DATATYPE_NULL)} {}

	auto operator=(skeleton&& other) & -> skeleton& {
		count_    = other.count_;
		datatype_ = std::exchange(other.datatype_, MPI_DATATYPE_NULL);
		return *this;
	}

	template<class Layout>
	skeleton(Layout const& lyt, MPI_Datatype dt, Size subcount)  // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init,fuchsia-default-arguments-declarations)
	: count_{static_cast<Size>(lyt.size())} {
		assert(lyt.size() <= std::numeric_limits<Size>::max());

		MPI_Datatype sub_type;  // NOLINT(cppcoreguidelines-init-variables)
		skeleton     sk;
		if constexpr(Layout::dimensionality == 1) {
			sub_type = dt;
		} else {
			sk       = skeleton(lyt.sub(), dt, lyt.sub().size());
			sub_type = sk.type();
		}

		int dt_size;
		MPI_Type_size(dt, &dt_size);  // NOLINT(cppcoreguidelines-init-variables)

		{
			MPI_Datatype vector_datatype;  // NOLINT(cppcoreguidelines-init-variables)
			MPI_Type_create_hvector(
				subcount, 1,
				lyt.stride() * dt_size,
				sub_type, &vector_datatype
			);

			MPI_Type_create_resized(vector_datatype, 0, lyt.stride() * dt_size, &datatype_);
			MPI_Type_free(&vector_datatype);
		}
	}

 public:
	template<class Layout>
	skeleton(Layout const& lyt, MPI_Datatype dt)
	: skeleton{lyt, dt, 1} {
		MPI_Type_commit(&datatype_);
	}

	skeleton(skeleton const&) = delete;

	auto operator=(skeleton const&) = delete;

	~skeleton() {
		if(datatype_ != MPI_DATATYPE_NULL)
			MPI_Type_free(&datatype_);
	}

	auto count() const { return count_; }
	auto type() const { return datatype_; }
};

template<typename Size = int>
class message {
	void*        buf_;
	Size         count_;
	MPI_Datatype datatype_;

 public:
	message(void* buf, skeleton<Size> const& sk) : buf_{buf}, count_{sk.count()}, datatype_{sk.type()} {}
	message(data dat, Size count) : buf_{dat.buffer()}, count_{count}, datatype_{dat.type()} {}

	template<class Arr>
	explicit message(Arr const& arr)
	: message{
		  const_cast<void*>(static_cast<void const*>(arr.base())), skeleton<Size>{arr.layout(), MPI_INT}  // NOLINT(cppcoreguidelines-pro-type-const-cast)
    } {}
	// : message{data(arr.begin()), static_cast<Size>(arr.size())} { assert(arr.size() <= std::numeric_limits<Size>::max()); }

	message(message const&) = delete;
	message(message&&)      = delete;

	auto operator=(message const&) = delete;
	auto operator=(message&&)      = delete;

	~message() {if(datatype_ != MPI_DATATYPE_NULL) { MPI_Type_free(&datatype_);} }

	auto buffer() const { return buf_; }
	auto count() const { return count_; }
	auto type() const { return datatype_; }
};

}  // namespace boost::multi::mpi

namespace multi = boost::multi;

namespace {
void test_single_number(MPI_Comm comm) {
	int world_rank;
	MPI_Comm_rank(comm, &world_rank);  // NOLINT(cppcoreguidelines-init-variables)
	int world_size;
	MPI_Comm_size(comm, &world_size);  // NOLINT(cppcoreguidelines-init-variables)

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

			MPI_Send(B_data.buffer(), static_cast<int>(BB.size()), B_data.type(), 1, 0, comm);
		} else if(world_rank == 1) {
			multi::array<int, 1> CC(3, 99);  // NOLINT(misc-const-correctness)

			auto const C_msg = multi::mpi::message(CC);

			MPI_Recv(C_msg.buffer(), C_msg.count(), C_msg.type(), 0, 0, comm, MPI_STATUS_IGNORE);
			BOOST_TEST(( CC == multi::array<double, 1>({1, 2, 3}) ));
		}
	}
	{
		if(world_rank == 0) {
			auto const  AA = multi::array<int, 1>({1, 2, 3, 4, 5, 6});
			auto const& BB = AA.strided(2);
			BOOST_TEST(( BB == multi::array<int, 1>({1, 3, 5}) ));

			auto const B_data = multi::mpi::data(BB.begin());

			MPI_Send(B_data.buffer(), static_cast<int>(BB.size()), B_data.type(), 1, 0, comm);
		} else if(world_rank == 1) {
			multi::array<int, 1> CC(3, 99);  // NOLINT(misc-const-correctness)

			auto const C_msg = multi::mpi::message(CC);

			MPI_Recv(C_msg.buffer(), C_msg.count(), C_msg.type(), 0, 0, comm, MPI_STATUS_IGNORE);
			BOOST_TEST(( CC == multi::array<double, 1>({1, 2, 3}) ));
		}
	}
	{
		if(world_rank == 0) {
			auto const  AA = multi::array<int, 1>({1, 2, 3, 4, 5, 6});
			auto const& BB = AA.strided(2);
			BOOST_TEST(( BB == multi::array<double, 1>({1, 3, 5}) ));

			auto const B_sk = multi::mpi::skeleton(BB.layout(), MPI_INT);

			MPI_Send(BB.base(), B_sk.count(), B_sk.type(), 1, 0, comm);
		} else if(world_rank == 1) {
			multi::array<int, 1> CC(3, 99);  // NOLINT(misc-const-correctness)

			auto const C_msg = multi::mpi::message(CC);

			MPI_Recv(C_msg.buffer(), C_msg.count(), C_msg.type(), 0, 0, comm, MPI_STATUS_IGNORE);
			BOOST_TEST(( CC == multi::array<double, 1>({1, 3, 5}) ));
		}
	}
}

}  // namespace

auto main() -> int {
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
			auto const  AA = multi::array<int, 2>(
				{
					{1, 2, 3},
					{4, 5, 6}
				}
			);
			auto const& BB = AA({0, 2}, {1, 3});
			BOOST_TEST(( BB == multi::array<double, 2>({{2, 3}, {5, 6}}) ));

			auto const B_sk = multi::mpi::skeleton(BB.layout(), MPI_INT);

			MPI_Send(BB.base(), B_sk.count(), B_sk.type(), 1, 0, MPI_COMM_WORLD);
		} else if(world_rank == 1) {
			multi::array<int, 2> CC({2, 2}, 99);

			auto const C_sk = multi::mpi::skeleton(CC.layout(), MPI_INT);

			MPI_Recv(CC.base(), C_sk.count(), C_sk.type(), 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			std::cout << CC[0][0] << ' ' << CC[0][1] << '\n'
					  << CC[1][0] << ' ' << CC[1][1] << '\n';

			BOOST_TEST(( CC == multi::array<double, 2>({{2, 3}, {5, 6}}) ));
		}
	}
	// {
	//  if(world_rank == 0) {
	//      auto const  AA = multi::array<int, 2>(
	//          {
	//              {1, 2, 3},
	//              {4, 5, 6}
	//          }
	//      );
	//      auto const& BB = AA({0, 2}, {1, 3});
	//      BOOST_TEST(( BB == multi::array<double, 2>({{2, 3}, {5, 6}}) ));

	//      auto const B_msg = multi::mpi::message(BB);

	//      MPI_Send(B_msg.buffer(), B_msg.count(), B_msg.type(), 1, 0, MPI_COMM_WORLD);
	//  } else if(world_rank == 1) {
	//      multi::array<int, 2> CC({2, 2}, 99);

	//      auto const C_sk = multi::mpi::skeleton(CC.layout(), MPI_INT);

	//      MPI_Recv(CC.base(), C_sk.count(), C_sk.type(), 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	//      std::cout << CC[0][0] << ' ' << CC[0][1] << '\n'
	//                << CC[1][0] << ' ' << CC[1][1] << '\n';

	//      BOOST_TEST(( CC == multi::array<double, 2>({{2, 3}, {5, 6}}) ));
	//  }
	// }


	MPI_Finalize();

	return boost::report_errors();
}
