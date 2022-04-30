#!/bin/bash
nvcc -O3 -arch=sm_50 -Xcompiler -fopenmp -o chebyprod2D chebyprod2D.cu
./chebyprod2D 10 10
./chebyprod2D 20 20
./chebyprod2D 40 40
./chebyprod2D 60 60
./chebyprod2D 80 80
./chebyprod2D 100 100
./chebyprod2D 120 120
./chebyprod2D 140 140
./chebyprod2D 160 160
./chebyprod2D 180 180
./chebyprod2D 200 200
