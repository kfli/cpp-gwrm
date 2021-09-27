#include <iostream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <ctime>
#include "..\..\inc\root_solvers.h"
#include "..\..\inc\chebyshev_algorithms.h"
using namespace std;

// Define global variables
int K = 8, L = 8;
double Lx = 0, Rx = 1.0;
double Lt = 0, Rt = 1.0;
double BMAx = 0.5 * (Rx - Lx), BPAx = 0.5 * (Rx + Lx);
double BMAt = 0.5 * (Rt - Lt), BPAt = 0.5 * (Rt + Lt);

// Initial condition
vector<double> init(K+1);

double f(double x) {
	return x * (1.0 - x);
}

// GWRM function
vector<double> gwrm(const vector<double> x) {
    int nelem = x.size();
	double sum;
    vector<double> fvec(nelem,0);
	Matrix a(K+1, vector<double>(L+1,0));
    Matrix at(K+1, vector<double>(L+1,0));
	Matrix ax(K+1, vector<double>(L+1,0));
	Matrix axx(K+1, vector<double>(L+1,0));
	Matrix a_ax(K+1, vector<double>(L+1,0));
	
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			a[i][j] = x[i + (K + 1) * j];
		}
    }
	
	chebyshev_x_derivative_2D_array(K, L, a, ax, BMAx);
	chebyshev_x_derivative_2D_array(K, L, ax, axx, BMAx);
	chebyshev_y_derivative_2D_array(K, L, a, at, BMAt);
	chebyshev_product_2D_array(K, L, a, ax, a_ax);
	
	// f(x) = du/dt + u.du/dx - nu.du2/dx2 = 0
	double nu = 0.03;
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			fvec[i + (K + 1) * j] = at[i][j] + a_ax[i][j] - nu * axx[i][j];
		}
    }
	
	// initial condition: L mode
	for (int i = 0; i < K-1; i++) {
		sum = 0.5 * a[i][0];
		for (int j = 1; j < L+1; j++) { sum += pow(-1,j) * a[i][j]; }
		fvec[i + (K + 1) * L] = sum - init[i];
    }
	
	// boundary conditions: K and K-1 mode
	for (int j = 0; j < L+1; j++) {
		sum = 0.5 * a[0][j];
		for (int i = 1; i < K+1; i++) { sum += a[i][j]; }
		fvec[K + (K + 1) * j] = sum - 0.0;
		
		sum = 0.5 * a[0][j];
		for (int i = 1; i < K+1; i++) { sum += pow(-1,i) * a[i][j]; }
		fvec[(K - 1) + (K + 1) * j] = sum - 0.0;
    }
	
	

    return fvec;
}

int main()
{
    vector<double> x0((K + 1) * (L + 1),0);
    vector<double> x1((K + 1) * (L + 1));
	
	chebyshev_coefficients_1D(K+1, f, init, BMAx, BPAx);
	for (double elem : init) {
        cout << elem << " ";
    }
	cout << endl;
	for (int i = 0; i < K+1; i++) {
		x0[i + (K + 1) * 0] = init[i];
    }
    clock_t c_start = clock();
    //x1 = quasi_newton(x0, gwrm);
	x1 = newton(x0, gwrm);
    clock_t c_end = clock();

    long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "CPU time used: " << time_elapsed_ms << " ms\n";
    for (double elem : x1) {
        cout << elem << " ";
    }
    cout << endl;
    return 0;
} 