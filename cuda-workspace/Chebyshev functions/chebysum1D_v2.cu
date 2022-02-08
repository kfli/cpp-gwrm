#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <time.h>
//typedef double mt;

/*
__global__ void chebyprod(int n, float *a, float *b, float *c){
   int i = blockIdx.x *blockDim.x + threadIdx.x;
   float sum;
   if (i < n) {
      sum = 0.f;
      for (int j = 0; j<=i; j++){
         sum += a[j]*b[j-i];
      }
      for (int j = 1; j < n-i; j++){
         sum += a[j]*b[j+i]+a[j+i]*b[j];
      }
      c[i] = 0.5f*sum;
   }  
}
*/

template <int blockSize>
__global__ void child_sum(int n, int k, float *a, float *b) {
   extern volatile __shared__ float sdata[];
   int     tid        = threadIdx.x,
           gridSize   = blockSize * gridDim.x,
           p          = blockIdx.x * blockSize + tid;
   float c;
   sdata[tid] = 0;
   while (p < n - k)
       {sdata[tid] += a[p]*b[p+k] + a[p+k]*b[p];
        p += gridSize;}
   __syncthreads();
   if (blockSize >= 512)
       { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
   if (blockSize >= 256)
       { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
   if (blockSize >= 128)
       { if (tid <  64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }
   if (tid < 32)
       { if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
         if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
         if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
         if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
         if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
         if (blockSize >= 2)  sdata[tid] += sdata[tid + 1]; }
   if (tid == 0) {c[blockIdx.x] = 0.5f*sdata[0];}
   //if (!threadIdx.x) c = 0.5f*sdata[0];
   return c;
}

__global__ void parent_chebyprod(int n, float *a, float *b, float *c) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   float c_k;
   if (i < n) {
      c_k = child_sum <256> <<<(n + 256 - 1) / 256, 256, 256*sizeof(float)>>>
                (n, blockIdx.x * blockDim.x + threadIdx.x, a, b);
      float sum = 0.f;
      for (int k = 0; k <= i; k++) {sum += a[k]*b[i-k];}
      //cudaDeviceSynchronize();
      //__syncthreads();
      c[i] = 0.5f*(sum + c_k);
   }
}

int main(void){
  clock_t tStart = clock();
  int N = 100;
  float *a, *b, *c, *d, *d_a, *d_b, *d_c, *d_d;
  a = (float*)malloc(N*sizeof(float));
  b = (float*)malloc(N*sizeof(float));
  c = (float*)malloc(N*sizeof(float));
  d = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_a, N*sizeof(float)); 
  cudaMalloc(&d_b, N*sizeof(float));
  cudaMalloc(&d_c, N*sizeof(float));
  cudaMalloc(&d_d, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    a[i] = 0.1f;
    b[i] = 0.2f;
    d[i] = 0.0f;
  }
   
  for (int i = 0; i < N; i++) {
     float sum = 0.f;
     for (int k = 0; k < N-i; k++) {sum += a[k]*b[k+i]+a[k+i]*b[k];}
     for (int k = 0; k <= i; k++) {sum += a[k]*b[i-k];}
     d[i] = 0.5f*sum;
  }

  std::cout << "Vector d: [ ";
  for (int k = 0; k < 10; ++k)
    std::cout << d[k] << " ";
  std::cout <<"]\n";

  for (int i = 0; i < N; i++) {
     d[i] = 0.0f;
  }

  

  cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_d, d, N*sizeof(float), cudaMemcpyHostToDevice);
  
  int blockSize, gridSize;
  // Number of threads in each thread block
  blockSize = 256;
  
  // Number of thread blocks in grid
  gridSize = (N + blockSize - 1) / blockSize;

  std::cout << "blockSize: " << blockSize << "\ngridSize: " << gridSize << "\n";
  
  // Perform chebyprod on N elements
  parent_chebyprod<<< gridSize, blockSize >>>(N, d_a, d_b, d_c);
  printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
  
  cudaMemcpy(c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
  
  std::cout << "Vector c: [ ";
  for (int k = 0; k < 10; ++k)
    std::cout << c[k] << " ";
  std::cout <<"]\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(a);
  free(b);
  free(c);
}
