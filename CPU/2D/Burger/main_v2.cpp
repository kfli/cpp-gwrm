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
int K = 16, L = 16, M = 3;
int N = (K + 1) * (L + 1) * (M + 1);
int Ne = 2;
double Lx = 0, Rx = 2.0 * PI;
double Ly = 0, Ry = 2.0 * PI;
double tend = 1.0;

double BMAt, BPAt;
double Lt, Rt;

double BMAx = 0.5 * (Rx - Lx), BPAx = 0.5 * (Rx + Lx);
double BMAy = 0.5 * (Ry - Ly), BPAy = 0.5 * (Ry + Ly);

// Initial condition
vector<double> init_a((K + 1) * (L + 1));
vector<double> init_b((K + 1) * (L + 1));

double u0(double x, double y) {
	return ( sin(x/2.0) * sin(y) );
}
double v0(double x, double y) {
	return ( -sin(x/2.0) * sin(y) );
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

tuple<double, double> calculate_dirichlet_bc(const vector<double> a) {
	// boundary conditions: K and K-1 mode
	double sum;
	double bc1, bc2;
	double alpha = 0.0;	// u(0,t) = 0
	double beta = 0.0;	// u(1,t) = 0
	if ( K % 2 == 0 ) { // K is even
		sum = 0.0;
		for (int i = 1; i < K-2; i++) { 
			if ( i % 2 != 0 ) { // odd
				sum += a[i];
			}
		}
		bc1 = (beta - alpha)/2.0 - sum;

		sum = 0.5 * a[0];
		for (int i = 1; i < K-1; i++) { 
			if ( i % 2 == 0 ) { // even
				sum +=a[i];
			}
		}
		bc2 = (alpha + beta)/2.0 - sum;

	} else { // K is odd
		sum = 0.5 * a[0];
		for (int i = 1; i < K-2; i++) { 
			if ( i % 2 == 0 ) { //even
				sum += a[i];
			}
		}
		bc1 = (alpha + beta)/2.0 - sum;
			
		sum = 0.0;
		for (int i = 1; i < K-1; i++) { 
			if ( i % 2 != 0 ) { // odd
				sum +=a[i];
			}
		}
		bc2 = (beta - alpha)/2.0 - sum;
	}
	return make_tuple(bc1,bc2);
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

void boundary_conditions(Eigen::VectorXd& g, const Eigen::VectorXd& x) {
	double b1, b2;
	vector<double> Ktmp((K + 1));
	vector<double> Ltmp((L + 1));
	// boundary conditions: K and K-1 mode
	for (int ne = 0; ne < Ne; ne++) {
		for (int j = 0; j < L+1; j++) {
			for (int k = 0; k < M+1; k++) {
				for (int i = 0; i < K+1; i++) { Ktmp[i] = x(ne * N + i + (K + 1) * ( j + (L + 1) * k )); }
				tie(b1, b2) = calculate_dirichlet_bc(Ktmp);
				g(ne * N + (K - 1) + (K + 1) * ( j + (L + 1) * k )) =  b1;
				g(ne * N + K + (K + 1) * ( j + (L + 1) * k )) =  b2;
			}
    	}
	
		// boundary conditions: L and L-1 mode
		for (int i = 0; i < K+1; i++) {
			for (int k = 0; k < M+1; k++) {
				for (int j = 0; j < L+1; j++) { Ltmp[j] = x(ne * N + i + (K + 1) * ( j + (L + 1) * k )); }
				tie(b1, b2) = calculate_dirichlet_bc(Ltmp);
				g(ne * N + i + (K + 1) * ( (L - 1) + (L + 1) * k )) =  b1;
				g(ne * N + i + (K + 1) * ( L + (L + 1) * k )) =  b2;
			}
		}
	}
}

// GWRM function
Eigen::VectorXd gwrm(const Eigen::VectorXd x) {
    int nelem = x.size();
	double sum;
	bool is_integration = false;
    Eigen::VectorXd g = Eigen::VectorXd::Zero(nelem);
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
	Array3D a_ax(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K, L, M, a, ax, a_ax);
	Array3D b_ay(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K, L, M, b, ay, b_ay);
	Array3D a_bx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K, L, M, a, bx, a_bx);
	Array3D b_by(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K, L, M, b, by, b_by);

	// du/dt + u.du/dx + v.du/dy - nu.( du2/dx2 +  du2/dy2 ) = 0
	// dv/dt + u.dv/dx + v.dv/dy - nu.( dv2/dx2 +  dv2/dy2 ) = 0
	double nu = 0.03;

	Array3D ai(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	Array3D bi(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			for (int k = 0; k < M + 1; k++) {
				at[i][j][k] = -a_ax[i][j][k] - b_ay[i][j][k] + nu * (axx[i][j][k] + ayy[i][j][k]);
				bt[i][j][k] = -a_bx[i][j][k] - b_by[i][j][k] + nu * (bxx[i][j][k] + byy[i][j][k]);
			}
		}
	}
	chebyshev_z_integration_3D_array(K, L, M, at, ai, BMAt);
	chebyshev_z_integration_3D_array(K, L, M, bt, bi, BMAt);
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			for (int k = 0; k < M + 1; k++) {
				g(0 * N + i + (K + 1) * ( j + (L + 1) * k )) = ai[i][j][k];
				g(1 * N + i + (K + 1) * ( j + (L + 1) * k )) = bi[i][j][k];
			}
		}
	}
	
	// initial condition: 0th mode
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			g(0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = ai[i][j][0] + 2.0 * init_a[i + (K + 1) *  j];
			g(1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = bi[i][j][0] + 2.0 * init_b[i + (K + 1) *  j];
		}
	}
	
	boundary_conditions(g, x);
	
    return g;
}
	
int main()
{
	cout << "*** STEP 1: GWRM STARTED *** \n";
	
	int nelem = Ne * (K + 1) * (L + 1) * (M + 1);
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
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nelem,nelem);
	for (int j = 0; j < nelem; j++) {
		H(j,j) = -1.0;
	}

	int steps = 1000;
	double h = tend / steps;
	Lt = 0.0, Rt = Lt + h;
	BMAt = 0.5 * (Rt - Lt), BPAt = 0.5 * (Rt + Lt);
	clock_t c_start = clock();
	double sum;
	for ( int n = 0; n < steps; n++ ) {
		
		cout << "*** STEP 2: SOLVER STARTED *** \n";
		cout << "*** STEP 2.1: COMPUTE INITIAL JACOBIAN *** \n";
		Eigen::VectorXd dh(nelem);
		Eigen::VectorXd g0(nelem);
		Eigen::VectorXd g1(nelem);
		Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nelem,nelem);
		g0 = gwrm(x0);
		double h = pow(10,-6);
		for (int j = 0; j < nelem; j++) {
			dh = Eigen::VectorXd::Zero(nelem);
			dh(j) = h;
			x1 = x0 + dh;
			g1 = gwrm(x1);
			for (int i = 0; i < nelem; i++) {
				H(i, j) = ((g1(i) - x1(i)) - (g0(i) - x0(i))) / h;
			}
		}
		cout << "*** STEP 2.2: COMPUTE INVERSE *** \n";
		Eigen::MatrixXd I(nelem,nelem);
		I.setIdentity();
		H =  H.colPivHouseholderQr().solve(I);
		x1 = quasi_newton(x0, gwrm, H);
		//x1 = anderson_acceleration(x0, gwrm);

		Lt = Rt;
		Rt += h;
		BMAt = 0.5 * (Rt - Lt);
		BPAt = 0.5 * (Rt + Lt);

		for (int i = 0; i < K+1; i++) {
			for (int j = 0; j < L+1; j++) {
				sum = 0.5 * x1(0 * N + i + (K + 1) * ( j + (L + 1) * 0 ));
				for (int k = 0; k < M+1; k++) {
					sum += x1(0 * N + i + (K + 1) * ( j + (L + 1) * k )) ;
				}
				init_a[i + (K + 1) *  j] = sum;
				x0(0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * sum;

				sum = 0.5 * x1(1 * N + i + (K + 1) * ( j + (L + 1) * 0 ));
				for (int k = 0; k < M+1; k++) {
					sum += x1(1 * N + i + (K + 1) * ( j + (L + 1) * k ));
				}
				init_b[i + (K + 1) *  j] = sum;
				x0(1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * sum;
			}
		}
	}
	clock_t c_end = clock();
		
    long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "CPU time used: " << time_elapsed_ms << " ms\n";
	
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			for (int k = 0; k < M+1; k++) {
				a[i + (K + 1) * ( j + (L + 1) * k )] = x1(0 * N + i + (K + 1) * ( j + (L + 1) * k ));
				b[i + (K + 1) * ( j + (L + 1) * k )] = x1(1 * N + i + (K + 1) * ( j + (L + 1) * k ));
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
			data_array[i + x_points * j][2] = eval_chebyshev_series(a, x_grid[i], y_grid[j], Rt);
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
