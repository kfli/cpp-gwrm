#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <time.h>

template <unsigned int blockSize>
__global__ void parreduc(float *array_in, float *reduct, size_t array_len)
    {
    extern volatile __shared__ float sdata[];
    size_t  tid        = threadIdx.x,
            gridSize   = blockSize * gridDim.x,
            i          = blockIdx.x * blockSize + tid;
    sdata[tid] = 0;
    while (i < array_len)
        { sdata[tid] += array_in[i];
        i += gridSize; }
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
    if (tid == 0) reduct[blockIdx.x] = sdata[0];
    }

int main(void){
  clock_t tStart;
  size_t N = 10000000;
  float *a, *b, *d_a, *d_b;
  a = (float*)malloc(N*sizeof(float));
  b = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_a, N*sizeof(float));
  cudaMalloc(&d_b, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    a[i] = 0.5f;
    b[i] = 0.0f;
  }
  tStart = clock();
  cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
  printf("cudaMemcpy: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);  
  
  const unsigned int blockSize=128, gridSize=256;
  std::cout << "blockSize: " << blockSize << "\ngridSize: " << gridSize << "\n";

  tStart = clock();
  float cpu_sum = 0.0f;
  for (size_t i = 0; i < N; i++) {cpu_sum += a[i];}
  printf("CPU time: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);  

  tStart = clock();
  // Perform sum on N elements
  parreduc<blockSize> <<< gridSize, blockSize, blockSize*sizeof(float)>>>(d_a, d_b, N);
  parreduc<blockSize> <<<        1, blockSize, blockSize*sizeof(float)>>>(d_b, d_b, N);
  cudaDeviceSynchronize();
  printf("GPU time: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

  cudaMemcpy(b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Sum of a: " << b[0] << "\n";

  cudaFree(d_a);
  cudaFree(d_b);
  free(a);
  free(b);
}
