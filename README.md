<!--
(pandoc `#--from gfm` --to html --standalone --metadata title=" " $0 > $0.html) && firefox --new-window $0.html; sleep 5; rm $0.html; exit
-->

**[Boost.] Multi**

> **Disclosure: This is not an official or accepted Boost library and is unrelated to the std::mdspan proposal. It is in the process of being proposed for inclusion in [Boost](https://www.boost.org/) and it doesn't depend on Boost libraries.**

_© Alfredo A. Correa, 2018-2025_

_Multi_ is a modern C++ library that provides manipulation and access of data in multidimensional arrays for both CPU and GPU memory.

# [Introduction](doc/multi/intro.adoc)

**Contents:**

[[_TOC_]]

# [Installation and tests](doc/multi/install.adoc)

# [Primer (basic usage)](doc/multi/primer.adoc)

# [Tutorial (advanced usage)](doc/multi/tutorial.adoc)

# [Reference](doc/multi/reference.adoc)

# [Interoperability](doc/multi/interop.adoc)

# [Technical point](doc/multi/technical.adoc)

# [Appendix](doc/multi/appendix.adoc)
