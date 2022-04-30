# The build target 
TARGET = testcases_chebyshev_algorithms testcases_root_solvers
INC_DIR = ..\inc
SRC_DIR = ..\src
DEPS = $(INC_DIR)\chebyshev_algorithms.h

# the compiler: gcc for C program, define as g++ for C++
CC = g++
 
# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings
CFLAGS  = -g -Wall -I$(INC_DIR)

SRCS = $(SRC_DIR)\chebyshev_algorithms.cpp 

CXXFLAGS = -O3 -fopenmp

all: $(TARGET)

%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) $(CXXFLAGS) -c $< -o $@