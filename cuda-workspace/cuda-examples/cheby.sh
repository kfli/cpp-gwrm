#!/bin/bash
nvcc -O3 -Xcompiler -fopenmp -arch=sm_50 chebyprodTest.cu -o chebyprodTest
./chebyprodTest 100
./chebyprodTest 200
./chebyprodTest 300
./chebyprodTest 400
./chebyprodTest 500
./chebyprodTest 600
./chebyprodTest 700
./chebyprodTest 800
./chebyprodTest 900
./chebyprodTest 1000


