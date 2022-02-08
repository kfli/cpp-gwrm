#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <ctime>
#include "..\..\inc\root_solvers.h"
#include "..\..\inc\chebyshev_algorithms.h"
using namespace std;

// Define global variables
int K = 16, L = 16;
double Lx = 0, Rx = 1.0;
double Lt = 0, Rt = 10.0;
double BMAx = 0.5 * (Rx - Lx), BPAx = 0.5 * (Rx + Lx);
double BMAt = 0.5 * (Rt - Lt), BPAt = 0.5 * (Rt + Lt);

// Initial condition
vector<double> init(K+1);

double f(double x) {
	return x * (1.0 - x);
}

vector<double> chebyshev_polynomials(double x, int n) {
	vector<double> T(n);
	T[0] = 1;
	T[1] = x;
	for (int i = 1; i < n-1; i++) {
		T[i+1] = 2 * x * T[i] - T[i-1];
	}
	return T;
}

double eval_chebyshev_series(const vector<double> x, const double xp, const double tp) {
	int nelem = x.size();
	vector<double> fp(nelem,1);
	vector<double> Tx(K);
	vector<double> Tt(L);
	fp[0] = fp[0]/2;
	Tx = chebyshev_polynomials( (xp - BMAx)/BPAx, K );
	Tt = chebyshev_polynomials( (tp - BMAt)/BPAt, L );
	double sum = 0;
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			sum += fp[i] * fp[j] * x[i + (K + 1) * j] * Tx[i] * Tt[j];
		}
	}
	return sum;
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
	double nu = 0.01;
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
	x1 = newton(x0, gwrm);
    clock_t c_end = clock();

    long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "CPU time used: " << time_elapsed_ms << " ms\n";
	
	// create equidistant mesh
	int x_points = 50;
	int t_points = 50;
	int tot_points = x_points * t_points;
	double data_array[tot_points][3];
	vector<double> x_grid(x_points);
	vector<double> t_grid(t_points);
	
	for (int i = 0; i < x_points; i++ ) {
		x_grid[i] = Lx + (Rx - Lx) * i / (x_points - 1);
	}
	for (int j = 0; j < t_points; j++ ) {
		t_grid[j] = Lt + (Rt - Lt) * j / (t_points - 1);
	}
	
	// evaluate GWRM solution 
	for (int i = 0; i < x_points; i++) {
		for (int j = 0; j < t_points; j++) {
			data_array[i + x_points * j][0] = x_grid[i];
			data_array[i + x_points * j][1] = t_grid[j];
			data_array[i + x_points * j][2] = eval_chebyshev_series(x1, x_grid[i], t_grid[j]);
		}
	}

	// write data to file
	ofstream myfile ("array_data.txt");
	if (myfile.is_open()) {
		for(int i = 0; i < tot_points; i++) {
			for(int j = 0; j < 2; j++) {
				myfile << data_array[i][j] << ", ";
			}
			myfile << data_array[i][2] << "\n";
		}
		myfile.close();
	}
	else cout << "Unable to open file";

    return 0;
} 