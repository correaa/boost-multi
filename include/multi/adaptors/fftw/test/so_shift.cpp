#include <multi/array.hpp>

#include <multi/adaptors/fftw.hpp>  // includes fftw3.hpp

#include <algorithm>  // for std::rotate
#include <complex>
#include <iostream>
#include <numeric>  // for std::iota

namespace multi = boost::multi;

int main() {
    using complex = std::complex<double>;

    // input array
    auto const x = std::invoke([]{  // NOLINT(readability-identifier-length)
        multi::array<complex, 1> ret(8);
        // fill the first array with some numbers
        std::iota(ret.begin(), ret.end(), 1.0);
        return ret;
    });

    // output array
    multi::array<complex, 1> y(x.size());  // NOLINT(readability-identifier-length)

    // compute the FFT of x and store results in y
    y = multi::fftw::dft_forward(x);  // or auto y = +multi::fftw::dft_forward(x);

    // display the results
    std::cout << "FFT =" << std::endl;
    std::copy(y.begin(), y.end(), std::ostream_iterator<complex>(std::cout, "\n"));

    // "shifted" results
    std::rotate(y.begin(), y.begin() + y.size()/2 + y.size()%2, y.end());

    std::cout << "FFT shifted =" << std::endl;
    std::copy(y.begin(), y.end(), std::ostream_iterator<complex>(std::cout, "\n"));
}
