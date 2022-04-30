#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <ctime>
#include "../../inc/root_solvers.h"
#include "../../inc/chebyshev_algorithms.h"
using namespace std;
const double PI = 3.141592653589793238463;

// Define global variables
int K = 22, L = 22, M = 5;
int N = (K + 1) * (L + 1) * (M + 1);
double Lx = 0, Rx = 2.0 * PI;
double Ly = 0, Ry = 2.0 * PI;
double Lt = 0, Rt = 1.0;

double BMAx = 0.5 * (Rx - Lx), BPAx = 0.5 * (Rx + Lx);
double BMAy = 0.5 * (Ry - Ly), BPAy = 0.5 * (Ry + Ly);
double BMAt = 0.5 * (Rt - Lt), BPAt = 0.5 * (Rt + Lt);

// Initial condition
vector<double> init_a((K + 1) * (L + 1));
vector<double> init_b((K + 1) * (L + 1));

double u0(double x, double y) {
	return ( cos(x) + sin(y) );
}
double v0(double x, double y) {
	return ( cos(x) - sin(y) );
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

double eval_chebyshev_series(const vector<double> x, const double xp, const double yp, const double tp) {
	int nelem = x.size();
	vector<double> fp(nelem,1);
	vector<double> Tx(K);
	vector<double> Ty(L);
	vector<double> Tt(M);
	fp[0] = fp[0]/2;
	Tx = chebyshev_polynomials( (xp - BMAx)/BPAx, K );
	Ty = chebyshev_polynomials( (yp - BMAy)/BPAy, L );
	Tt = chebyshev_polynomials( (tp - BMAt)/BPAt, M );
	double sum = 0;
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			for (int k = 0; k < M+1; k++) {
				sum += fp[i] * fp[j] * fp[k] * x[i + (K + 1) * ( j + (L + 1) * k )] * Tx[i] * Ty[j] * Tt[k];
			}
		}
	}
	return sum;
}

void boundary_conditions(const Array3D& a, const Array3D& ax, const Array3D& ay, const Array3D& b, const Array3D& bx, const Array3D& by, Eigen::VectorXd& fvec) {
	double sum_right, sum_left;
	// boundary conditions: K and K-1 mode
	for (int j = 0; j < L+1; j++) {
		for (int k = 0; k < M+1; k++) {
			sum_right = 0.5 * a[0][j][k];
			for (int i = 1; i < K+1; i++) { sum_right += a[i][j][k]; }
			sum_left = 0.5 * a[0][j][k];
			for (int i = 1; i < K+1; i++) { sum_left += pow(-1,i) * a[i][j][k]; }
			fvec(0 * N + K + (K + 1) * ( j + (L + 1) * k )) = sum_right - sum_left;
		
			sum_right = 0.5 * ax[0][j][k];
			for (int i = 1; i < K+1; i++) { sum_right += ax[i][j][k]; }
			sum_left = 0.5 * ax[0][j][k];
			for (int i = 1; i < K+1; i++) { sum_left += pow(-1,i) * ax[i][j][k]; }
			fvec(0 * N + (K - 1) + (K + 1) * ( j + (L + 1) * k )) = sum_right - sum_left;
			
			sum_right = 0.5 * b[0][j][k];
			for (int i = 1; i < K+1; i++) { sum_right += b[i][j][k]; }
			sum_left = 0.5 * b[0][j][k];
			for (int i = 1; i < K+1; i++) { sum_left += pow(-1,i) * b[i][j][k]; }
			fvec(1 * N + K + (K + 1) * ( j + (L + 1) * k )) = sum_right - sum_left;
		
			sum_right = 0.5 * bx[0][j][k];
			for (int i = 1; i < K+1; i++) { sum_right += bx[i][j][k]; }
			sum_left = 0.5 * bx[0][j][k];
			for (int i = 1; i < K+1; i++) { sum_left += pow(-1,i) * bx[i][j][k]; }
			fvec(1 * N + (K - 1) + (K + 1) * ( j + (L + 1) * k )) = sum_right - sum_left;
		}
    }
	
	// boundary conditions: L and L-1 mode
	for (int i = 0; i < K+1; i++) {
		for (int k = 0; k < M+1; k++) {
			sum_right = 0.5 * a[i][0][k];
			for (int j = 1; j < L+1; j++) { sum_right += a[i][j][k]; }
			sum_left = 0.5 * a[i][0][k];
			for (int j = 1; j < L+1; j++) { sum_left += pow(-1,j) * a[i][j][k]; }
			fvec(0 * N + i + (K + 1) * ( L + (L + 1) * k )) = sum_right - sum_left;
		
			sum_right = 0.5 * ay[i][0][k];
			for (int j = 1; j < L+1; j++) { sum_right += ay[i][j][k]; }
			sum_left = 0.5 * ay[i][0][k];
			for (int j = 1; j < L+1; j++) { sum_left += pow(-1,j) * ay[i][j][k]; }
			fvec(0 * N + i + (K + 1) * ( (L - 1) + (L + 1) * k )) = sum_right - sum_left;
			
			sum_right = 0.5 * b[i][0][k];
			for (int j = 1; j < L+1; j++) { sum_right += b[i][j][k]; }
			sum_left = 0.5 * b[i][0][k];
			for (int j = 1; j < L+1; j++) { sum_left += pow(-1,j) * b[i][j][k]; }
			fvec(1 * N + i + (K + 1) * ( L + (L + 1) * k )) = sum_right - sum_left;
		
			sum_right = 0.5 * by[i][0][k];
			for (int j = 1; j < L+1; j++) { sum_right += by[i][j][k]; }
			sum_left = 0.5 * by[i][0][k];
			for (int j = 1; j < L+1; j++) { sum_left += pow(-1,j) * by[i][j][k]; }
			fvec(1 * N + i + (K + 1) * ( (L - 1) + (L + 1) * k )) = sum_right - sum_left;
		}
    }
}

// GWRM function
Eigen::VectorXd gwrm_linear(const Eigen::VectorXd x) {
    int nelem = x.size();
	bool is_integration = false;
	double sum;
    Eigen::VectorXd fvec = Eigen::VectorXd::Zero(nelem);
	Array3D a(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	Array3D b(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			for (int k = 0; k < M+1; k++) {
				a[i][j][k] = x(0 * N + i + (K + 1) * ( j + (L + 1) * k ));
				b[i][j][k] = x(1 * N + i + (K + 1) * ( j + (L + 1) * k ));
			}
		}
    }
	
	// derivatives
	Array3D ax(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, a, ax, BMAx);
	Array3D axx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_x_derivative_3D_array(K, L, M, ax, axx, BMAx);
	Array3D ay(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, a, ay, BMAy);
	Array3D ayy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_y_derivative_3D_array(K, L, M, ay, ayy, BMAy);
	Array3D at(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  
	
	Array3D bx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, b, bx, BMAx);
	Array3D bxx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_x_derivative_3D_array(K, L, M, bx, bxx, BMAx);
	Array3D by(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, b, by, BMAy);
	Array3D byy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_y_derivative_3D_array(K, L, M, by, byy, BMAy);
	Array3D bt(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); 
	
	// du/dt - nu.( du2/dx2 +  du2/dy2 ) = 0
	// dv/dt - nu.( dv2/dx2 +  dv2/dy2 ) = 0
	double nu = 0.05;
	if (!is_integration) {
		chebyshev_z_derivative_3D_array(K, L, M, a, at, BMAt);
		chebyshev_z_derivative_3D_array(K, L, M, b, bt, BMAt);
		
		for (int i = 0; i < K-1; i++) {
			for (int j = 0; j < L-1; j++) {
				for (int k = 0; k < M; k++) {
						fvec(0 * N + i + (K + 1) * ( j + (L + 1) * k )) = at[i][j][k] - nu * (axx[i][j][k] + ayy[i][j][k]);
						fvec(1 * N + i + (K + 1) * ( j + (L + 1) * k )) = bt[i][j][k] - nu * (bxx[i][j][k] + byy[i][j][k]);
				}
			}
		}
		
		// initial condition: M mode
		for (int i = 0; i < K+1; i++) {
			for (int j = 0; j < L+1; j++) {
				sum = 0.5 * a[i][j][0];
				for (int k = 1; k < M+1; k++) { sum += pow(-1,k) * a[i][j][k]; }
				fvec(0 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_a[i + (K + 1) *  j];
			
				sum = 0.5 * b[i][j][0];
				for (int k = 1; k < M+1; k++) { sum += pow(-1,k) * b[i][j][k]; }
				fvec(1 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_b[i + (K + 1) *  j];
			}
		}
	} else {
		Array3D ai(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
		Array3D bi(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
		for (int i = 0; i < K-1; i++) {
			for (int j = 0; j < L-1; j++) {
				for (int k = 0; k < M + 1; k++) {
						at[i][j][k] = nu * (axx[i][j][k] + ayy[i][j][k]);
						bt[i][j][k] = nu * (bxx[i][j][k] + byy[i][j][k]);
				}
			}
		}
		chebyshev_z_integration_3D_array(K, L, M, at, ai, BMAt);
		chebyshev_z_integration_3D_array(K, L, M, bt, bi, BMAt);
		for (int i = 0; i < K-1; i++) {
			for (int j = 0; j < L-1; j++) {
				for (int k = 0; k < M + 1; k++) {
						fvec(0 * N + i + (K + 1) * ( j + (L + 1) * k )) = ai[i][j][k] - a[i][j][k];
						fvec(1 * N + i + (K + 1) * ( j + (L + 1) * k )) = bi[i][j][k] - b[i][j][k];
				}
			}
		}
		
		// initial condition: 0th mode
		for (int i = 0; i < K+1; i++) {
			for (int j = 0; j < L+1; j++) {
				fvec(0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = fvec(0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) + 2.0 * init_a[i + (K + 1) *  j];
				fvec(1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = fvec(1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) + 2.0 * init_b[i + (K + 1) *  j];
			}
		}
    }
	
	boundary_conditions(a, ax, ay, b, bx, by, fvec);
	
    return fvec;
}

// GWRM function
Eigen::VectorXd gwrm(const Eigen::VectorXd x) {
    int nelem = x.size();
	double sum;
	bool is_integration = false;
    Eigen::VectorXd fvec = Eigen::VectorXd::Zero(nelem);
	Array3D a(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	Array3D b(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			for (int k = 0; k < M+1; k++) {
				a[i][j][k] = x(0 * N + i + (K + 1) * ( j + (L + 1) * k ));
				b[i][j][k] = x(1 * N + i + (K + 1) * ( j + (L + 1) * k ));
			}
		}
    }
	
	// derivatives
	Array3D ax(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, a, ax, BMAx);
	Array3D axx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_x_derivative_3D_array(K, L, M, ax, axx, BMAx);
	Array3D ay(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, a, ay, BMAy);
	Array3D ayy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_y_derivative_3D_array(K, L, M, ay, ayy, BMAy);
	Array3D at(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  
	
	Array3D bx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, b, bx, BMAx);
	Array3D bxx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_x_derivative_3D_array(K, L, M, bx, bxx, BMAx);
	Array3D by(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, b, by, BMAy);
	Array3D byy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_y_derivative_3D_array(K, L, M, by, byy, BMAy);
	Array3D bt(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); 
 
	// products
	Array3D a_ax(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, a, ax, a_ax);
	Array3D b_ay(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, b, ay, b_ay);
	Array3D a_bx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, a, bx, a_bx);
	Array3D b_by(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, b, by, b_by);

	// du/dt + u.du/dx + v.du/dy - nu.( du2/dx2 +  du2/dy2 ) = 0
	// dv/dt + u.dv/dx + v.dv/dy - nu.( dv2/dx2 +  dv2/dy2 ) = 0
	double nu = 0.05;
	if (!is_integration) {
		chebyshev_z_derivative_3D_array(K, L, M, a, at, BMAt);
		chebyshev_z_derivative_3D_array(K, L, M, b, bt, BMAt);
		
		for (int i = 0; i < K-1; i++) {
			for (int j = 0; j < L-1; j++) {
				for (int k = 0; k < M; k++) {
						fvec(0 * N + i + (K + 1) * ( j + (L + 1) * k )) = at[i][j][k] + a_ax[i][j][k] + b_ay[i][j][k] - nu * (axx[i][j][k] + ayy[i][j][k]);
						fvec(1 * N + i + (K + 1) * ( j + (L + 1) * k )) = bt[i][j][k] + a_bx[i][j][k] + b_by[i][j][k] - nu * (bxx[i][j][k] + byy[i][j][k]);
				}
			}
		}
		
		// initial condition: M mode
		for (int i = 0; i < K+1; i++) {
			for (int j = 0; j < L+1; j++) {
				sum = 0.5 * a[i][j][0];
				for (int k = 1; k < M+1; k++) { sum += pow(-1,k) * a[i][j][k]; }
				fvec(0 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_a[i + (K + 1) *  j];
			
				sum = 0.5 * b[i][j][0];
				for (int k = 1; k < M+1; k++) { sum += pow(-1,k) * b[i][j][k]; }
				fvec(1 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_b[i + (K + 1) *  j];
			}
		}
	} else {
		Array3D ai(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
		Array3D bi(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
		for (int i = 0; i < K-1; i++) {
			for (int j = 0; j < L-1; j++) {
				for (int k = 0; k < M + 1; k++) {
						at[i][j][k] = -a_ax[i][j][k] - b_ay[i][j][k] + nu * (axx[i][j][k] + ayy[i][j][k]);
						bt[i][j][k] = -a_bx[i][j][k] - b_by[i][j][k] + nu * (bxx[i][j][k] + byy[i][j][k]);
				}
			}
		}
		chebyshev_z_integration_3D_array(K, L, M, at, ai, BMAt);
		chebyshev_z_integration_3D_array(K, L, M, bt, bi, BMAt);
		for (int i = 0; i < K-1; i++) {
			for (int j = 0; j < L-1; j++) {
				for (int k = 0; k < M + 1; k++) {
						fvec(0 * N + i + (K + 1) * ( j + (L + 1) * k )) = ai[i][j][k] - a[i][j][k];
						fvec(1 * N + i + (K + 1) * ( j + (L + 1) * k )) = bi[i][j][k] - b[i][j][k];
				}
			}
		}
		
		// initial condition: 0th mode
		for (int i = 0; i < K+1; i++) {
			for (int j = 0; j < L+1; j++) {
				fvec(0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = fvec(0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) + 2.0 * init_a[i + (K + 1) *  j];
				fvec(1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = fvec(1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) + 2.0 * init_b[i + (K + 1) *  j];
			}
		}
    }
	
	
	boundary_conditions(a, ax, ay, b, bx, by, fvec);
	
    return fvec;
}
	
int main()
{
	cout << "*** STEP 1: GWRM STARTED *** \n";
	int num_eq = 2;
	int nelem = num_eq * (K + 1) * (L + 1) * (M + 1);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nelem);
    Eigen::VectorXd x1(nelem);
	vector<double> a((K + 1) * (L + 1) * (M + 1));
	vector<double> b((K + 1) * (L + 1) * (M + 1));
	
	chebyshev_coefficients_2D(K+1, L+1, u0, init_a, BMAx, BPAx, BMAy, BPAy);
	chebyshev_coefficients_2D(K+1, L+1, v0, init_b, BMAx, BPAx, BMAy, BPAy);
	
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			x0(0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * init_a[i + (K + 1) *  j];
			x0(1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * init_b[i + (K + 1) *  j];
		}
    }
	cout << "*** STEP 1.5: START CLOCK *** \n";
    clock_t c_start = clock();

	cout << "*** STEP 2: SOLVER STARTED *** \n";
	//x1 = newton(x0, gwrm);

	
	Eigen::VectorXd dh(nelem);
	Eigen::VectorXd f0(nelem);
	Eigen::VectorXd f1(nelem);
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nelem,nelem);
	f0 = gwrm_linear(x0);
	double h = 0.01;
	for (int j = 0; j < nelem; j++) {
		dh = Eigen::VectorXd::Zero(nelem);
		dh(j) = h;
		x1 = x0 + dh;
		f1 = gwrm_linear(x1);
		for (int i = 0; i < nelem; i++) {
			H(i,j) = (f1(i) - f0(i)) / h;
		}
	}
	//Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nelem,nelem);
	//for (int j = 0; j < nelem; j++) {
	//	H(j,j) = 1.0;
	//}
	cout << "*** STEP 2: CALC INV ***";
	H = H.inverse();
	
	x1 = quasi_newton(x0, gwrm, H);
	
	
	//x1 = AMFA(x0, gwrm);
	//x1 = anderson_acceleration(x0, gwrm);
	//x1 = anderson_picard_acceleration(x0, gwrm);
    clock_t c_end = clock();
    long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "CPU time used: " << time_elapsed_ms << " ms\n";
	
	vector<double> sigma(K+1,1);
	vector<double> cutoff(K+1,1);
	int p = 4;
	int Nc = 6;
	//double eps =  2.220446 * pow(10, -16);
	//double alpha = -log(eps);
	double alpha = 17;
	for (int i = K-p; i < K+1; i++) {
		sigma[i] = exp(-alpha * pow(i / K, 2 * p));
		cutoff[i] = 0;
	}
	
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			for (int k = 0; k < M+1; k++) {
				a[i + (K + 1) * ( j + (L + 1) * k )] = x1(0 * N + i + (K + 1) * ( j + (L + 1) * k ));
				b[i + (K + 1) * ( j + (L + 1) * k )] = cutoff[i] * cutoff[j] * x1(1 * N + i + (K + 1) * ( j + (L + 1) * k ));
			}
		}
    }
	
	// create equidistant mesh
	int x_points = 50;
	int y_points = 50;
	int tot_points = x_points * y_points;
	double data_array[tot_points][3];
	vector<double> x_grid(x_points);
	vector<double> y_grid(y_points);
	
	for (int i = 0; i < x_points; i++ ) {
		x_grid[i] = Lx + (Rx - Lx) * i / (x_points - 1);
	}
	for (int j = 0; j < y_points; j++ ) {
		y_grid[j] = Ly + (Ry - Ly) * j / (y_points - 1);
	}
	
	// evaluate GWRM solution 
	for (int i = 0; i < x_points; i++) {
		for (int j = 0; j < y_points; j++) {
			data_array[i + x_points * j][0] = x_grid[i];
			data_array[i + x_points * j][1] = y_grid[j];
			data_array[i + x_points * j][2] = eval_chebyshev_series(b, x_grid[i], y_grid[j], Rt);
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
