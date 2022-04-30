#include <stdio.h>
#include <iostream>
#include <sys/time.h>
typedef double df;

int main(int argc, char** argv){
  int x = 1; int y = 2;
  int n = 5; int m = 5;
  
  int width_x = (((x)>(n-x))?(x):(n-x))+1;
  int height_y = x > y ? n : m;

  printf("width_x: %i\n", width_x);
  printf("width_x: %i\n", height_y);

  return 0;
  
}
