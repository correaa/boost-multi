<!--
(pandoc `#--from gfm` --to html --standalone --metadata title=" " $0 > $0.html) && firefox --new-window $0.html; sleep 5; rm $0.html; exit
-->

**[Boost.] Multi**

> **Disclosure: This is not an official or accepted Boost library and is unrelated to the std::mdspan proposal. It is in the process of being proposed for inclusion in [Boost](https://www.boost.org/) and it doesn't depend on Boost libraries.**

_© Alfredo A. Correa, 2018-2025_

_Multi_ is a modern C++ library that provides manipulation and access of data in multidimensional arrays for both CPU and GPU memory.

```cpp
#include <cassert>          // for assert
#include <multi/array.hpp>  // from https://gitlab.com/correaa/boost-multi or https://gitlab.com/correaa/boost-multi

namespace multi = boost::multi;

int main() {
    multi::array<int, 2> A = {{1, 2, 3}, {4, 5, 6}};  // 2D array of integers

    assert(A.size() == 2);                    // the array has 2 rows
    assert(A.size() == A.end() - A.begin());  // interators to rows

    assert(A[1][1] == 5);  // element access through indexing

    assert(A.elements().size() == 2 * 3);  // array has 6 elements
    assert(A.elements()[4] == 5);          // elements gives "flat" sequences
}
```
[(online)](https://godbolt.org/z/6cnqzK4ah)

## Learn about Multi

* [Online documentation](https://correaa.gitlab.io/boost-multi/multi.html)

## Try Multi

Before installing the library, you can try it [online](https://godbolt.org/z/dvacqK8jE) through the Godbolt's Compiler Explorer.

## Install Multi

_Multi_ has no external dependencies and can be used immediately after downloading.
```bash
git clone https://gitlab.com/correaa/boost-multi.git
```

_Multi_ doesn't require installation since a single header is enough to use the entire core library;
```c++
#include <multi/array.hpp>

int main() { ... }
```

The library can be also installed with CMake.
The header (and CMake) files will be installed in the chosen prefix location (by default, `/usr/local/include/multi` and `/usr/local/share/multi`).
```bash
cd boost-multi
mkdir -p build && cd build
cmake . -B ./build  # --install-prefix=$HOME/.local
cmake --install ./build  # or sudo ...
```

_Testing_ the library requires Boost.Core (headers), installed for example, via `sudo apt install cmake git g++ libboost-test-dev make` or `sudo dnf install boost-devel cmake gcc-c++ git`.
A CMake build system is provided to compile and run basic tests.
```bash
ctest -C ./build
```

Once installed, other CMake projects (targets) can depend on Multi by adding a simple `add_subdirectory(my_multi_path)` or by `find_package`:
```cmake
find_package(multi)  # see https://gitlab.com/correaa/boost-multi
```

Alternatively, the library can be fetched on demand:
```cmake
include(FetchContent)
FetchContent_Declare(multi GIT_REPOSITORY https://gitlab.com/correaa/boost-multi.git)
FetchContent_MakeAvailable(multi)
...
target_link_libraries(my_target PUBLIC multi)
```

## Support

* File a Gitlab [issue](https://gitlab.com/correaa/boost-multi/-/issues/new?type=ISSUE) or Github [issue](https://github.com/correaa/boost-multi/issues/new/choose).
* Join the [**#boost-multi**](https://cpplang.slack.com/archives/C071VGKUA5P) discussion group at [cpplang.slack.com](https://cpplang.slack.com/)
