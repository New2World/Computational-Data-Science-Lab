_MAIN_=StartTime.cpp
CXXFLAGS=-lboost_thread -lboost_system -lpthread -std=c++14 -O2

startTime: $(_MAIN_)
	$(CXX) -o $@ $^ $(CXXFLAGS)

all: startTime

clean:
	rm -f startTime
