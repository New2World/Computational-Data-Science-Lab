_MAIN_=StartTime.cpp
CXXFLAGS=-lboost_thread -lboost_system -lpthread -std=c++14 -O2

start.out: $(_MAIN_)
	$(CXX) -o $@ $^ $(CXXFLAGS)

all: start.out

clean:
	rm -f start.out
