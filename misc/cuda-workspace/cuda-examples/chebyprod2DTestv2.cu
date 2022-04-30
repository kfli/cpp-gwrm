#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <time.h>
#include <time.h>
#include <sys/time.h>
typedef double df;
#define USECPSEC 1000000ULL
#define BS 1<<5
#define N 100
#define M 100

const bool sync = true;
const bool nosync = false;
unsigned long long dtime_usec(unsigned long long start, bool use_sync = nosync){
  if (use_sync == sync) cudaDeviceSynchronize();
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

float cpu_sum(int n, int m, df *a, df *b, df *c) {
   df q, r;
   #pragma omp parallel for collapse(2)
   for (int x = 0; x < n; x++) {
      for (int y = 0; y < m; y++) {
         q = 0.0f;
         for (int i = 0; i <= x; i++) {
            r = 0.0f;
            for (int j = 0; j <= y; j++) {
               r += a[i * n + j] * b[(x - i) * n + y - j];
            }
            for (int j = 1; j < m - y; j++) {
               r += a[i * n + j] * b[(x - i) * n + y + j] 
                    + a[i * n + y + j] * b[(x - i) * n + j];
            }
            q += r;
         }
         for (int i = 1; i < n-x; i++) {
            r = 0.0f;
            for (int j = 0; j <= y; j++) {
               r += a[i * n + j] * b[(x + i) * n + y - j]
                    + a[(x + i) * n + j] * b[ i * n + y - j];
            }
            for (int j = 1; j < m - y; j++) {
               r += a[i * n + j] * b[(x + i) * n + y + j] 
                    + a[(x + i) * n + y + j] * b[(x + i) * n + j]
                 
                    +a[(x + i) * n + j] * b[i * n + y + j] 
                    + a[(x + i) * n + y + j] * b[i * n + j];
            }
            q += r;
         }
      c[x * N + y] = 0.25f*q;
      }
   }
   return 0;
}

const int P2  = 5;
const int TPB = 1<<P2;
const unsigned row_mask = ~((0xFFFFFFFFU>>P2)<<P2);
__global__ void chebyprod_imp(int n, int m, df *a, df *b, df *c){
   __shared__ df sdata[TPB];
   int x = blockIdx.x;
   int row_width_x = (((x)>(n-x))?(x):(n-x))+1;
   int row_width_y = (((y)>(m-y))?(y):(m-y))+1;
   int strides_x = (row_width_x>>P2) + ((row_width_x&row_mask)?1:0);
   int strides_y = (row_width_y>>P2) + ((row_width_y&row_mask)?1:0);
   int i = threadIdx.x;
   df tmp_a, r;
   df sum = 0.0f;
   for (int s=0; s < strides_x; s++){ // block-stride x loop
      if (i < n && j < m) {tmp_a = a[i * n + j];}
      if (i <= x) {
         r = 0.0f;
            if (j <= y) {r += tmp_a * b[(x - i) * n + y - j];}
            if ((j > 0) && (j < (m-y))) {r += tmp_a * b[(x - i) * n + y + j] 
                                              + a[i * n + y + j] * b[(x - i) * n + j];}
            sum += r;
         }
      }
      if ((i > 0) && (i < (n-x))) {	
         r = 0.0f;
            if (j <= y) {r += tmp_a * b[(x + i) * n + y - j]
                              + a[(x + i) * n + j] * b[ i * n + y - j];}
            if ((j > 0) && (j < (m-y))) {r += tmp_a * b[(x + i) * n + y + j] 
                                              + a[(x + i) * n + y + j] * b[(x + i) * n + j]
                                              + a[(x + i) * n + j] * b[i * n + y + j] 
                                              + a[(x + i) * n + y + j] * b[i * n + j];}
            sum += r;         }
      }
      i += TPB;
   }
   sdata[threadIdx.x * n] = sum;
   for (int s = TPB>>1; s > 0; s>>=1) { // sweep reduction in x
      __syncthreads();
      if (threadIdx.x < s) {
         sdata[threadIdx.x] += sdata[threadIdx.x  + s];
      }
   }
   if (!threadIdx.x) c[x * n ] = 0.25f*sdata[0];
}

int main(void){
  int size = N*M*sizeof(df);
  df *a, *b, *c, *cc, *d_a, *d_b, *d_c;
  a  = (df*)malloc(size);
  b  = (df*)malloc(size);
  c  = (df*)malloc(size);
  cc = (df*)malloc(size);

  cudaMalloc(&d_a, size); 
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);
  #pragma omp parallel for collapse (2)
  for (int i = 0; i < N; i++) {
     for (int j = 0; j < M; j++) {
        a[i * M + j] = 0.1f;
        b[i * M + j] = 0.2f;
     }
  }

  unsigned long long  dt = dtime_usec(0);
  // Perform chebyprod on N elements
  cpu_sum(N, M, a, b, cc);
  dt = dtime_usec(dt,sync);
  printf("Time taken 2D CPU: %fs\n", dt/(float)USECPSEC);
  df dtc = dt/(float)USECPSEC;

  std::cout << "Vector cc: [ ";
  for (int k = 0; k < 10; ++k)
    std::cout << cc[k] << " ";
  std::cout <<"]\n";

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  const int GS = (N*M + BS -1) / BS;
  
  dt = dtime_usec(0);
  // Perform chebyprod on N elements
  chebyprod_imp<<< GS, BS >>>(N, M, d_a, d_b, d_c);
  dt = dtime_usec(dt,sync);
  printf("Time taken 2D stride kernel: %fs\n", dt/(float)USECPSEC);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  
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
  free(cc);
}
