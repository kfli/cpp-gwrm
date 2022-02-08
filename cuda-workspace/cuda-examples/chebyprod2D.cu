#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>
typedef double df;
#define USECPSEC 1000000ULL
#define BSX 1<<5
#define BSY 1<<5
//#define N 100
//#define M 100

const bool sync = true;
const bool nosync = false;
unsigned long long dtime_usec(unsigned long long start, bool use_sync = nosync){
  if (use_sync == sync) cudaDeviceSynchronize();
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

int divUp(int a, int b) {return (a + b - 1) / b;}

void cpu_sum(int n, int m, df *a, df *b, df *c) {
   df q, r;
   #pragma omp parallel for collapse(2)
   for (int x = 0; x < n; x++) {
     for (int y = 0; y < m; y++) {
        q = 0.0f;
        for (int i = 0; i <= x; i++) {
           r = 0.0f;
           for (int j = 0; j <= y; j++) {
              r += a[j * n + i] * b[(y - j) * n + x - i];
           }
           for (int j = 1; j < m - y; j++) {
              r += a[j * n + i] * b[(y+j) * n + x - i] + a[(y+j)* n + i] * b[j*n+(x - i)];
           }
           q += r;
        }
        for (int i = 1; i < n-x; i++) {
           r = 0.0f;
           for (int j = 0; j <= y; j++) {
              r += a[j * n + i] * b[(y-j) * n + x+i] + a[j*n + (x + i)] * b[(y - j)*n + i];
           }
           for (int j = 1; j < m - y; j++) {
              r += a[j * n + i] * b[(y+j) * n + x+i]
                      +  a[(y+j) * n + x + i] * b[j*n+(x + i)]
                      +  a[j*n + (x + i)] * b[(y+j)*n + i]
                      +  a[(y+j)*n + x + i] * b[j*n+i];
           }
           q += r;
        }
        c[y * m + x] = 0.25f*q;
     }   
   }
}

// choose one warp per output point
const int P2  = 5;  // assumes warp size is 32
const unsigned row_mask = ~((0xFFFFFFFFU>>P2)<<P2);
__global__ void chebyprod_imp(int n, int m, const df * __restrict__ a, const df * __restrict__ b, df * __restrict__ c){
   int x = blockIdx.x;
   int y = threadIdx.y+blockDim.y*blockIdx.y;
   int width_x = (((x)>(n-x))?(x):(n-x))+1;
   int height_y = (((y)>(m-y))?(y):(m-y))+1;
   int strides_x = (width_x>>P2) + ((width_x&row_mask)?1:0);
   int strides_y = height_y;
   int i = threadIdx.x;
   df tmp_a;
   df sum = 0.0f;
   if ((x < n) && (y < m)){
   for (int s=0; s < strides_x; s++) { // warp-stride x loop
     for (int j=0; j < strides_y; j++) { // y loop
       if (i < n && j < m) {tmp_a = a[j * n + i];}
       if (i <= x) {
         if (j <= y) {sum += tmp_a * b[(y 
- j) * n + x - i];}
         if ((j > 0) && (j < (m-y))) {sum += tmp_a * b[(y+j) * n + x - i] 
                                          + a[(y+j)* n + i] * b[j*n+(x - i)];}
       }
       if ((i > 0) && (i < (n-x))) {

         if (j <= y) {sum += tmp_a * b[(y-j) * n + x+i] + a[j*n + (x + i)] * b[(y - j)*n + i];}
         if ((j > 0) && (j < (m-y)))
           {sum += tmp_a * b[(y+j) * n + x+i]
                +  a[(y+j) * n + x + i] * b[j*n+(x + i)]
                +  a[j*n + (x + i)] * b[(y+j)*n + i]
                +  a[(y+j)*n + x + i] * b[j*n+i];}
       }
     }
     i += 32;
   }
   // warp-shuffle reduction
   for (int offset = warpSize>>1; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
   if (!threadIdx.x) c[y * m + x] = 0.25f*sum;}
}

__global__ void chebyprod(int n, int m, df *a, df *b, df *c){
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   df q, r;
   if (x < n && y < m) {
      q = 0.0f;
      for (int i = 0; i <= x; i++) {
         r = 0.0f;
         for (int j = 0; j <= y; j++) {
            r += a[j * n + i] * b[(y - j) * n + x - i];
         }
         for (int j = 1; j < m - y; j++) {
            r += a[j * n + i] * b[(y+j) * n + x - i] + a[(y+j)* n + i] * b[j*n+(x - i)];
         }
         q += r;
      }
      for (int i = 1; i < n-x; i++) {
         r = 0.0f;
         for (int j = 0; j <= y; j++) {
            r += a[j * n + i] * b[(y-j) * n + x+i] + a[j*n + (x + i)] * b[(y - j)*n + i];
         }
         for (int j = 1; j < m - y; j++) {
             r += a[j * n + i] * b[(y+j) * n + x+i]
                  +  a[(y+j) * n + x + i] * b[j*n+(x + i)]
                  +  a[j*n + (x + i)] * b[(y+j)*n + i]
                  +  a[(y+j)*n + x + i] * b[j*n+i];
         }
         q += r;
      }
      c[y * m + x] = 0.25f*q;
   }
}

int main(int argc, char** argv){
  int N = atoi(argv[1]);
  int M = atoi(argv[2]);
  printf("N = %i and M = %i: ", N, M);
  int size = N*M*sizeof(df);
  df *a, *b, *c, *cc, *ci, *d_a, *d_b, *d_c, *d_ci;
  a  = (df*)malloc(size);
  b  = (df*)malloc(size);
  c  = (df*)malloc(size);
  cc = (df*)malloc(size);
  ci = (df*)malloc(size);

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);
  cudaMalloc(&d_ci, size);
  #pragma omp parallel for collapse (2)
  for (int j = 0; j < M; j++) {
	for (int i = 0; i < N; i++) {
		a[j * N + i] = 0.1f;
		b[j * N + i] = 0.2f;
	}
  }

  unsigned long long  dt = dtime_usec(0);
  // Perform chebyprod on N elements
  cpu_sum(N, M, a, b, cc);
  dt = dtime_usec(dt,sync);
  df dtc = dt/(float)USECPSEC;
  
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  dim3 dimBlock(BSX, BSY);
  dim3 dimGrid(divUp(N, BSX), divUp(M, BSY));

  dt = dtime_usec(0);
  // Perform chebyprod on N elements
  chebyprod<<< dimGrid, dimBlock >>>(N, M, d_a, d_b, d_c);
  dt = dtime_usec(dt,sync);
  printf("Speedup1: %fs :", dtc/(dt/(float)USECPSEC));

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  dt = dtime_usec(0);
  // Perform chebyprod on N elements
  dim3 dimGrid2(N, (M+dimBlock.y-1)/dimBlock.y);
  chebyprod_imp<<< dimGrid2, dimBlock >>>(N, M, d_a, d_b, d_ci);
  dt = dtime_usec(dt,sync);
  printf("Speedup2: %fs\n", dtc/(dt/(float)USECPSEC));

  cudaMemcpy(ci, d_ci, size, cudaMemcpyDeviceToHost);
  
  printf("Time taken 2D CPU: %fs\n", dtc);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_ci);
  free(a);
  free(b);
  free(c);
  free(cc);
  free(ci);
}
