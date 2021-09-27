#include <iostream>
#include <omp.h>
#include <vector>
#include "..\src\chebyshev_algorithms.cpp"

using namespace std;
typedef vector<vector<double>> Matrix;
typedef vector<vector<vector<double>>> Array3D;

/* ------------------------------------------- */
/*  functions that use flattened Malloc arrays */
/* ------------------------------------------- */
void chebyshev_x_derivative_1D(int l, double* a, double* b, double BMAx);

void chebyshev_y_derivative_2D(int l, int m, double* a, double* b, double BMAy);

void chebyshev_z_derivative_3D(int l, int m, int n, double* a, double* b, double BMAz);

void chebyshev_product_1D(int n, double* a, double* b, double* c);

void chebyshev_product_2D(int l, int m, double* a, double* b, double* c);

void chebyshev_product_3D(int l, int m, int n, double* a, double* b, double* c);

/* -------------------------------------------------------- */
/*  functions that use arrays created with the vector class */
/* -------------------------------------------------------- */
void chebyshev_coefficients_1D(int M, double (*f)(double),  vector<double> &c, double BMA, double BPA);
	
void chebyshev_x_derivative_1D_array(int l, vector<double> &a, vector<double> &b, double BMAx);

void chebyshev_x_derivative_2D_array(int l, int m, Matrix &a, Matrix &b, double BMAx);

void chebyshev_y_derivative_2D_array(int l, int m, Matrix &a, Matrix &b, double BMAy); 

void chebyshev_z_derivative_3D_array(int l, int m, int n, Array3D &a, Array3D &b, double BMAz);

void chebyshev_product_1D_array(int l, vector<double> &a, vector<double> &b, vector<double> &c);

void chebyshev_product_2D_array(int l, int m, Matrix &a, Matrix &b, Matrix &c);

void chebyshev_product_3D_array(int l, int m, int n, Array3D &a, Array3D &b, Array3D &c);