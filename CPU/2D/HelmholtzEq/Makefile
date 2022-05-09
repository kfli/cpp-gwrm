# The build target
TARGET = main_multidomain main
INC_DIR = ../../inc
SRC_DIR = ../../src
DEPS = $(INC_DIR)/root_solvers.h
EIG_DIR = /usr/local/include/eigen-3.4.0/

# the compiler: gcc for C program, define as g++ for C++
CC = g++

# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings
CFLAGS  = -g -Wall -I$(INC_DIR)

CXXFLAGS = -O3 -fopenmp -I$(EIG_DIR)
SRCS = $(SRC_DIR)/root_solvers.cpp

all: $(TARGET)

%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) $(CXXFLAGS)-c $< -o $@
