#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4-*-
$CXXX $CXXFLAGS -D'MULTI_MARK_SCOPE(MsG)=BOOST_LOG_TRIVIAL(trace)<<MsG' -DBOOST_LOG_DYN_LINK $0 -o $0x -lboost_log -lboost_thread -lboost_system -lboost_log_setup -lpthread &&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2020

#ifndef MULTI_CONFIG_MARK_HPP
#define MULTI_CONFIG_MARK_HPP

#include<cassert>

#ifndef MULTI_MARK_SCOPE
#define MULTI_MARK_SCOPE(MsG) ((void)0)
#endif

#ifndef MULTI_MARK_FUNCTION
#define MULTI_MARK_FUNCTION MULTI_MARK_SCOPE(__func__)
#endif

#if not __INCLUDE_LEVEL__

#include <iostream>
#include <boost/log/trivial.hpp>

void multi_fun(){
	MULTI_MARK_SCOPE("function multi_fun"); //	BOOST_LOG_TRIVIAL(trace)<< "[2020-09-23 16:42:55.035034] [0x00007f6060af3a00] [trace]   function multi_fun";
}

void multi_gun(){
	MULTI_MARK_FUNCTION;
}

int main(){
	multi_fun();
	multi_gun();
}

#endif
#endif

