// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
// © Alfredo A. Correa 2021

#ifndef MULTI_ADAPTORS_FFTW
#define MULTI_ADAPTORS_FFTW

#include<fftw3.h> // external fftw3 library

namespace boost{
namespace multi{
namespace fftw{

void cleanup(){fftw_cleanup();}

struct environment{
	~environment(){cleanup();}
};

}
}
}
#endif
