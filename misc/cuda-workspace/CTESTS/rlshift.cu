#include <stdio.h>
#include <iostream>
#include <cuda.h>

int main() {
   unsigned int i = 10;
   unsigned int k = 1<<i;
   printf("%i\n", k);
   const unsigned row_mask = ~((0xFFFFFFFFU>>i)<<i);
   printf("%i\n", row_mask);
   return 0;
}
