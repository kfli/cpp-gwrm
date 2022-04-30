# The build target 
TARGET = main
INC_DIR = ..\..\inc
SRC_DIR = ..\..\src
DEPS = $(INC_DIR)\root_solvers.h
EIG_DIR = \Eigen\eigen-3.4.0\

# the compiler: gcc for C program, define as g++ for C++
CC = g++
NVCC = nvcc
 
# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings
CFLAGS  = -g -Wall -I$(INC_DIR)

CXXFLAGS = -O3 -fopenmp -I$(EIG_DIR) -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"



SRCS = $(SRC_DIR)\root_solvers.cpp 

all: $(TARGET)

%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(CFLAGS) $(CXXFLAGS)-c $< -o $@