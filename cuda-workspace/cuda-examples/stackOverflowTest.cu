#include <stdio.h>
#include <iostream>
#include <cuda.h>
typedef float mt;
#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL
const bool sync = true;
const bool nosync = false;
unsigned long long dtime_usec(unsigned long long start, bool use_sync = nosync){
  if (use_sync == sync) cudaDeviceSynchronize();
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}
__global__ void chebyprod(int n, const mt * __restrict__ a, const mt * __restrict__ b, mt * __restrict__ c){
   int i = blockIdx.x *blockDim.x + threadIdx.x;
   mt sum;
   if (i < n) {
      sum = 0.f;
      for (int j = 0; j<=i; j++){
         sum += a[j]*b[i-j];
      }
      for (int j = 1; j < n-i; j++){
         sum += a[j]*b[j+i]+a[j+i]*b[j];
      }
      c[i] = 0.5f*sum;
   }
}
// assume one threadblock per c_k coefficient
// assume a power-of-2 threadblock size
const int tpb_p2 = 8;
const int nTPB = 1<<tpb_p2;
const unsigned row_mask = ~((0xFFFFFFFFU>>tpb_p2)<<tpb_p2);

__global__ void chebyprod_imp(int n, const mt * __restrict__ a, const mt * __restrict__ b, mt * __restrict__ c){
#ifndef NO_WS
  __shared__ mt sd[32];
  if (threadIdx.x < 32) sd[threadIdx.x] = 0;
  __syncthreads();
#else
  __shared__ mt sd[nTPB];
#endif
  int k = blockIdx.x;
  mt sum = 0.0f;
  int row_width = (((k)>(n-k))?(k):(n-k))+1;
  int strides = (row_width>>tpb_p2)+ ((row_width&row_mask)?1:0);
  int j = threadIdx.x;
  mt tmp_a;
  for (int s=0; s < strides; s++){ // block-stride loop
    if (j < n) tmp_a = a[j];
    if (j <= k) sum += tmp_a*b[k-j];
    if ((j > 0) && (j < (n-k))) sum += tmp_a*b[j+k] + a[j+k]*b[j];
    j += nTPB;
    }
#ifndef NO_WS
  // 1st warp-shuffle reduction
  int lane = threadIdx.x & (warpSize-1);
  int warpID = threadIdx.x >> 5; // assumes warpSize == 32
  unsigned mask = 0xFFFFFFFFU;
  for (int offset = warpSize>>1; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(mask, sum, offset);
  if (lane == 0) sd[warpID] = sum;
  __syncthreads(); // put warp results in shared mem
  // hereafter, just warp 0
  if (warpID == 0){
  // reload val from shared mem if warp existed
    sum = sd[lane];
  // final warp-shuffle reduction
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(mask, sum, offset);
  }
#else
  sd[threadIdx.x] = sum;
  for (int s = nTPB>>1; s > 0; s>>=1){ // sweep reduction
    __syncthreads();
    if (threadIdx.x < s) sd[threadIdx.x] += sd[threadIdx.x+s];}
  if (!threadIdx.x) sum = sd[0];
#endif
  if (!threadIdx.x) c[k] = sum*0.5f;
}

int main(int argc, char *argv[]){
  int N = 10000;
  if (argc>1) N = atoi(argv[1]);
  std::cout << "N = " << N << std::endl;
  mt *a, *b, *c, *ic, *d_a, *d_b, *d_c;
  a  = (mt*)malloc(N*sizeof(mt));
  b  = (mt*)malloc(N*sizeof(mt));
  c  = (mt*)malloc(N*sizeof(mt));
  ic = (mt*)malloc(N*sizeof(mt));

  cudaMalloc(&d_a, N*sizeof(mt));
  cudaMalloc(&d_b, N*sizeof(mt));
  cudaMalloc(&d_c, N*sizeof(mt));

  for (int i = 0; i < N; i++) {
    a[i] = 0.1f;
    b[i] = 0.2f;
  }

  cudaMemcpy(d_a, a, N*sizeof(mt), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N*sizeof(mt), cudaMemcpyHostToDevice);
  int blockSize, gridSize;
  // Number of threads in each thread block
  blockSize = 1024;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)N/blockSize);

  std::cout << "blockSize: " << blockSize << "\ngridSize: " << gridSize << "\n";

  // Perform chebyprod on N elements
  unsigned long long  dt = dtime_usec(0);
  chebyprod<<< gridSize, blockSize >>>(N, d_a, d_b, d_c);
  dt = dtime_usec(dt,sync);

  cudaMemcpy(c, d_c, N*sizeof(mt), cudaMemcpyDeviceToHost);
  printf("Time taken: %fs\n", dt/(float)USECPSEC);
  std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
  std::cout << "Vector c: [ ";
  for (int k = 0; k < 10; ++k)
    std::cout << c[k] << " ";
  std::cout <<"]\n";
  dt = dtime_usec(0);
  chebyprod_imp<<< N, nTPB >>>(N, d_a, d_b, d_c);
  dt = dtime_usec(dt,sync);
  cudaMemcpy(ic, d_c, N*sizeof(mt), cudaMemcpyDeviceToHost);
  printf("Time taken: %fs\n", dt/(float)USECPSEC);
  std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
  std::cout << "Vector c: [ ";
  for (int k = 0; k < 10; ++k)
    std::cout << ic[k] << " ";
  std::cout <<"]\n";
  mt max_error = 0;
  for (int k = 0; k < N; k++)
    max_error = fmax(max_error, fabs(c[k] - ic[k]));
  std::cout << "Max error = " << max_error << std::endl;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(a);
  free(b);
  free(c);
  free(ic);
}
