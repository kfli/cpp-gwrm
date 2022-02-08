#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <omp.h>
typedef double df;
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
	
__global__ void chebyprod(int n, df * __restrict__ a, df * __restrict__ b, df * __restrict__ c){
   int i = blockIdx.x *blockDim.x + threadIdx.x;
   df sum;
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
const int tpb_p2 = 5;
const int nTPB = 1<<tpb_p2;
const unsigned row_mask = ~((0xFFFFFFFFU>>tpb_p2)<<tpb_p2);
__global__ void chebyprod_imp(int n, const df * __restrict__ a, const df * __restrict__ b, df * __restrict__ c){
  int k = blockIdx.x;
  if (k < n) {
    df sum = 0.0f;
    int row_width = (((k)>(n-k))?(k):(n-k))+1;
    int strides = (row_width>>tpb_p2)+ ((row_width&row_mask)?1:0);
    int j = threadIdx.x;
    df tmp_a;
    for (int s=0; s < strides; s++){ // block-stride loop
      if (j < n) tmp_a = a[j];
      if (j <= k) sum += tmp_a*b[k-j];
      if ((j > 0) && (j < (n-k))) sum += tmp_a*b[j+k] + a[j+k]*b[j];
      j += 32;
    }
    for (int offset = warpSize>>1; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);}
    if (!threadIdx.x) c[k] = sum*0.5f;
  }
}


int main(int argc, char** argv){
  int N = atoi(argv[1]);
  printf("This N = %i ,", N);
  df *a, *b, *c, *ic, *d_a, *d_b, *d_c, *cc;
  a   = (df*)malloc(N*sizeof(df));
  b   = (df*)malloc(N*sizeof(df));
  c   = (df*)malloc(N*sizeof(df));
  cc  = (df*)malloc(N*sizeof(df));
  ic  = (df*)malloc(N*sizeof(df));

  cudaMalloc(&d_a, N*sizeof(df));
  cudaMalloc(&d_b, N*sizeof(df));
  cudaMalloc(&d_c, N*sizeof(df));

  for (int i = 0; i < N; i++) {
    a[i] = 0.01f;
    b[i] = 0.02f;
  }

  unsigned long long  dt = dtime_usec(0);
  df sum;
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
     sum = 0.f;
     for (int j = 0; j<=i; j++){
        sum += a[j]*b[i-j];
     }
     for (int j = 1; j < N-i; j++){
        sum += a[j]*b[j+i]+a[j+i]*b[j];
     }
     cc[i] = 0.5f*sum;
  }
  dt = dtime_usec(dt,sync);
  printf("Time taken serial CPU: %fs\n", dt/(float)USECPSEC);
  df dtc = dt/(float)USECPSEC;
  
  std::cout << "Vector c: [ ";
  for (int k = 0; k < 10; ++k)
    std::cout << cc[k]<< " ";
  std::cout <<"]\n";
  
  cudaMemcpy(d_a, a, N*sizeof(df), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N*sizeof(df), cudaMemcpyHostToDevice);

  int blockSize, gridSize;
  // Number of threads in each thread block
  blockSize = 256;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)N/blockSize);

  //std::cout << "blockSize: " << blockSize << "\ngridSize: " << gridSize << "\n";

  // Perform chebyprod on N elements
  dt = dtime_usec(0);
  chebyprod<<< gridSize, blockSize >>>(N, d_a, d_b, d_c);
  dt = dtime_usec(dt,sync);

  cudaMemcpy(c, d_c, N*sizeof(df), cudaMemcpyDeviceToHost);
  
  //printf("Time taken monolithic kernel: %fs\n", dt/(float)USECPSEC);
  /*
  std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
  std::cout << "Vector c: [ ";
  for (int k = 0; k < 10; ++k
    std::cout << c[k] << " ";
  std::cout <<"]\n";
  */
  printf("Speedup1: %f :", dtc/(dt/(float)USECPSEC));

  dt = dtime_usec(0);
  chebyprod_imp<<< N, nTPB >>>(N, d_a, d_b, d_c);
  dt = dtime_usec(dt,sync);

  cudaMemcpy(ic, d_c, N*sizeof(df), cudaMemcpyDeviceToHost);
  
  printf("Time taken stride kernel: %fs\n", dt/(float)USECPSEC);
  
  std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
  std::cout << "Vector c: [ ";
  for (int k = 0; k < 10; ++k)
    std::cout << ic[k] << " ";
  std::cout <<"]\n";
  
  printf("Speedup2: %f\n", dtc/(dt/(float)USECPSEC));
  /*
  mt max_error = 0;
  for (int k = 0; k < N; k++)
    max_error = fmax(max_error, fabs(c[k] - ic[k]));
  std::cout << "Max error = " << max_error << std::endl;
  */
  
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(a);
  free(b);
  free(c);
  free(ic);
}
