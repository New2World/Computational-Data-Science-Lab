_MAIN_=start.cpp
PARALLEL_LAGS=-lboost_thread -lboost_system -lpthread
CXXFLAGS=-std=c++11 -O3

start.out: $(_MAIN_)
	$(CXX) -o $@ $^ $(CXXFLAGS)

all: start.out

clean:
	rm -f start.out
