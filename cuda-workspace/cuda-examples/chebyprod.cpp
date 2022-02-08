#include <stdio.h>
#include <iostream>
#include <time.h>

void chebyprod(int n, float *a, float *b, float *c){
   float sum;
   for (int i = 0; i < n; i++) {
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

int main(void){
  clock_t tStart = clock();
    
  int N = 10000;
  float *a, *b, *c;
  a = (float*)malloc(N*sizeof(float));
  b = (float*)malloc(N*sizeof(float));
  c = (float*)malloc(N*sizeof(float));

  for (int i = 0; i < N; i++) {
    a[i] = 0.1f;
    b[i] = 0.2f;
  }
 
  // Perform chebyprod on N elements
  chebyprod(N, a, b, c);
  printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
  
  std::cout << "Vector c: [ ";
  for (int k = 0; k < 10; ++k)
    std::cout << c[k] << " ";
  std::cout <<"]\n";

  free(a);
  free(b);
  free(c);
}
