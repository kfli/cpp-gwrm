# The build target 
TARGET = main
INC_DIR = ..\..\inc
SRC_DIR = ..\..\src
DEPS = $(INC_DIR)\root_solvers.h
ARMA_DIR = \armadillo\armadillo-10.7-RC1\include
LAPACK_DIR = \CMAKE_BUILD_LAPACK\include

# the compiler: gcc for C program, define as g++ for C++
CC = g++
 
# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings
CFLAGS  = -g -Wall -I$(INC_DIR)

CXXFLAGS = -O3 -fopenmp -I$(ARMA_DIR) -I$(LAPACK_DIR)

SRCS = $(SRC_DIR)\root_solvers.cpp

all: $(TARGET)

%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) $(CXXFLAGS)-c $< -o $@