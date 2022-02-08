#include <stdio.h>
#include <iostream>
#include <cuda.h>

__global__ void chebyprod(int n, float *a, float *b){
   int i = blockIdx.x *blockDim.x + threadIdx.x;
   float sum;
   if (i < n)
      sum = 0.f;
      for (int j = 0; j <= i; j++){ sum += a[j]; }
      b[i] = sum;  
}

int main(void){
  int N = 10;
  float *a, *b, *d_a, *d_b;
  a = (float*)malloc(N*sizeof(float));
  b = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_a, N*sizeof(float)); 
  cudaMalloc(&d_b, N*sizeof(float));


  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
  }

  cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
  
  int blockSize, gridSize, mDim;
  // Number of threads in each thread block
  blockSize = 1024;

  // Maximum dimensionality of grid of thread blocks
  mDim = 3;
  
  // Number of thread blocks in grid
  gridSize = (int)ceil((float)N/blockSize);
  if (gridSize > mDim){
     gridSize = mDim;
  }

  std::cout << "blockSize: " << blockSize << "\ngridSize: " << gridSize << "\n";
  
  // Perform chebyprod on N elements
  chebyprod<<< gridSize, blockSize >>>(N, d_a, d_b);
  
  cudaMemcpy(b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost);
  
  std::cout << "Vector c: [ ";
  for (int k = 0; k < N; ++k)
    std::cout << b[k] << " ";
  std::cout <<"]\n";

  cudaFree(d_a);
  cudaFree(d_b);
  free(a);
  free(b);
}
