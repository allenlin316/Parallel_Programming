CXX = g++
CXXFLAGS = -std=c++11 -O3 -pthread -fopenmp

all: hw1

hw1: hw1.cpp
	$(CXX) $(CXXFLAGS) -o hw1 hw1.cpp -lpng

clean:
	rm -f hw1

.PHONY: all clean

