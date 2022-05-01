#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <ctime>
#include "..\..\inc\root_solvers.h"
#include "..\..\inc\chebyshev_algorithms.h"
using namespace std;
const double PI = 3.141592653589793238463;

// Define global variables
int K = 5, L = 5, M = 4;
int N = (K + 1) * (L + 1) * (M + 1);
int Ne = 2;
int Nt = N * Ne;
int Nx = 3, Ny = 3;
double Lx = 0, Rx = 2.0 * PI;
double Ly = 0, Ry = 2.0 * PI;
double Lt = 0, Rt = 2.0;

vector<double> x_grid(Nx+1);
vector<double> y_grid(Ny+1);
vector<double> BMAx(Nx); vector<double> BPAx(Nx);
vector<double> BMAy(Ny); vector<double> BPAy(Ny);
double BMAt = 0.5 * (Rt - Lt), BPAt = 0.5 * (Rt + Lt);

// Initial condition
vector<double> init_a((K + 1) * (L + 1));
vector<double> init_b((K + 1) * (L + 1));
vector<double> init_as(Nx * Ny * (K + 1) * (L + 1));
vector<double> init_bs(Nx * Ny * (K + 1) * (L + 1));

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
	Tx = chebyshev_polynomials( (xp - BPAx)/BMAx, K );
	Ty = chebyshev_polynomials( (yp - BPAy)/BMAy, L );
	Tt = chebyshev_polynomials( (tp - BPAt)/BMAt, M );
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

// GWRM function
Eigen::VectorXd gwrm_linear(const Eigen::VectorXd x) {
    	int nelem = x.size();
	double sum;
	double al_sol_right, al_sol_left, ac_sol_left, ac_sol_right, ar_sol_left, ar_sol_right;
	double al_der_right, al_der_left, ac_der_left, ac_der_right, ar_der_left, ar_der_right;
	double bl_sol_right, bl_sol_left, bc_sol_left, bc_sol_right, br_sol_left, br_sol_right;
	double bl_der_right, bl_der_left, bc_der_left, bc_der_right, br_der_left, br_der_right;
	vector<double> al_tmp((K + 1));
	vector<double> ac_tmp((K + 1));
	vector<double> ar_tmp((K + 1));
	vector<double> bl_tmp((K + 1));
	vector<double> bc_tmp((K + 1));
	vector<double> br_tmp((K + 1));
	bool is_integration = false;
    	Eigen::VectorXd fvec = Eigen::VectorXd::Zero(nelem);
	Array3D a = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	Array3D b = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	
	for (sx = 0; sx < Nx; sx++) {
		for (sy = 0; sy < Ny; sy++) {
			for (int i = 0; i < K+1; i++) {
				for (int j = 0; j < L+1; j++) {
					for (int k = 0; k < M+1; k++) {
						a[0 * N + i + (K + 1) * ( j + (L + 1) * k )] = x((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
						b[1 * N + i + (K + 1) * ( j + (L + 1) * k )] = x((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
					}	
				}
			}
			
			// derivatives
			Array3D ax = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, a, ax, BMAx[sx]);
			Array3D axx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_x_derivative_3D_array(K, L, M, ax, axx, BMAx[sx]);
			Array3D ay = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, a, ay, BMAy[sy]);
			Array3D ayy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_y_derivative_3D_array(K, L, M, ay, ayy, BMAy[sy]);
			Array3D at(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  
			
			Array3Dbx = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, b, bx, BMAx[sx]);
			Array3D bxx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_x_derivative_3D_array(K, L, M, bx, bxx, BMAx[sx]);
			Array3D by = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, b, by, BMAy[sy]);
			Array3D byy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_y_derivative_3D_array(K, L, M, by, byy, BMAy[sy]);
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
								fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k )) = at[i][j][k] - nu * (axx[i][j][k] + ayy[i][j][k]);
								fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k )) = bt[i][j][k] - nu * (bxx[i][j][k] + byy[i][j][k]);
						}
					}
				}
				
				// initial condition: M mode
				for (int i = 0; i < K+1; i++) {
					for (int j = 0; j < L+1; j++) {
						sum = 0.5 * a[i][j][0];
						for (int k = 1; k < M+1; k++) { sum += pow(-1,k) * a[i][j][k]; }
						fvec(0 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_as[(sx + Nx * sy ) * (K + 1) * (L + 1) + i + K * j];
					
						sum = 0.5 * b[i][j][0];
						for (int k = 1; k < M+1; k++) { sum += pow(-1,k) * b[i][j][k]; }
						fvec(1 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_bs[(sx + Nx * sy ) * (K + 1) * (L + 1) + i + K * j];
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
								fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k )) = ai[i][j][k] - a[i][j][k];
								fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k )) = bi[i][j][k] - b[i][j][k];
						}
					}
				}
				
				// initial condition: 0th mode
				// initial condition: 0th mode
				for (int i = 0; i < K+1; i++) {
					for (int j = 0; j < L+1; j++) {
						fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) 
							+ 2.0 * init_as[(sx + Nx * sy ) * (K + 1) * (L + 1) + i + K * j];
						fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) 
							+ 2.0 * init_bs[(sx + Nx * sy ) * (K + 1) * (L + 1) + i + K * j];
					}
				}
			}
		}
	}
	
	for (sx = 0; sx < Nx; sx++) {
		for (sy = 0; sy < Ny; sy++) {
			// boundary conditions: K and K-1 mode
			for (int j = 0; j < L+1; j++) {
				for (int k = 0; k < M+1; k++) {
					for (int i = 0; i < K+1; i++) {
						ac_tmp[i] = x((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
						bc_tmp[i] = x((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						if (sx == 0) {
							al_tmp[i] = x((Nx - 1 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[i] = x((Nx - 1 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[i] = x((sx + 1 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[i] = x((sx + 1 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						} else if ( sx == Nx - 1) {
							al_tmp[i] = x((sx - 1 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[i] = x((sx - 1 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[i] = x((sx + 1 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[i] = x((sx + 1 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						} else {
							al_tmp[i] = x((sx - 1 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[i] = x((sx - 1 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[i] = x((0 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[i] = x((0 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						}
					}
					double [al_sol_right, al_der_right] = echebser1(1.0, al_tmp);
					double [ac_sol_left, ac_der_left] = echebser1(-1.0, ac_tmp);
					double [ac_sol_right, ac_der_right] = echebser1(1.0, ac_tmp);
					double [ar_sol_left, ar_der_left] = echebser1(-1.0, ar_tmp);

					fvec((sx + Nx * sy ) * num_sub + 0 * N + K + (K + 1) * ( j + (L + 1) * 0 )) = ac_sol_left - al_sol_right;
					fvec((sx + Nx * sy ) * num_sub + 0 * N + (K - 1) + (K + 1) * ( j + (L + 1) * 0 )) = ac_der_right - ar_der_left;
						
					double [bl_sol_right, bl_der_right] = echebser1(1.0, bl_tmp);
					double [bc_sol_left, bc_der_left] = echebser1(-1.0, bc_tmp);
					double [bc_sol_right, bc_der_right] = echebser1(1.0, bc_tmp);
					double [br_sol_left, br_der_left] = echebser1(-1.0, br_tmp);

					fvec((sx + Nx * sy ) * num_sub + 1 * N + K + (K + 1) * ( j + (L + 1) * 0 )) = bc_sol_left - bl_sol_right;
					fvec((sx + Nx * sy ) * num_sub + 1 * N + (K - 1) + (K + 1) * ( j + (L + 1) * 0 )) = bc_der_right - br_der_left;
				}
			}
			
			// boundary conditions: L and L-1 mode
			for (int i = 0; i < K+1; i++) {
				for (int k = 0; k < M+1; k++) {
					for (int j = 0; j < L+1; j++) {
						ac_tmp[j] = x((sx + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
						bc_tmp[j] = x((sx + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						if (sy == 0) {
							al_tmp[j] = x((sx + Nx * (Ny - 1)) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[j] = x((sx + Nx * (Ny - 1)) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[j] = x((sx + Nx * (sy + 1)) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[j] = x((sx + Nx * (sy + 1)) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						} else if ( sy == Ny - 1) {
							al_tmp[j] = x((sx + Nx * (sy - 1)) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[j] = x((sx + Nx * (sy - 1)) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[j] = x((sx + Nx * (sy + 1)) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[j] = x((sx + Nx * (sy + 1)) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						} else {
							al_tmp[j] = x((sx + Nx * (sy - 1) ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[j] = x((sx + Nx * (sy - 1) ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[j] = x((sx + Nx * 0 ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[j] = x((sx + Nx * 0 ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						}
					}
					double [al_sol_right, al_der_right] = echebser1(1.0, al_tmp);
					double [ac_sol_left, ac_der_left] = echebser1(-1.0, ac_tmp);
					double [ac_sol_right, ac_der_right] = echebser1(1.0, ac_tmp);
					double [ar_sol_left, ar_der_left] = echebser1(-1.0, ar_tmp);

					fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( L + (L + 1) * 0 )) = ac_sol_left - al_sol_right;
					fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( (L - 1) + (L + 1) * 0 )) = ac_der_right - ar_der_left;
						
					double [bl_sol_right, bl_der_right] = echebser1(1.0, bl_tmp);
					double [bc_sol_left, bc_der_left] = echebser1(-1.0, bc_tmp);
					double [bc_sol_right, bc_der_right] = echebser1(1.0, bc_tmp);
					double [br_sol_left, br_der_left] = echebser1(-1.0, br_tmp);

					fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( L + (L + 1) * 0 )) = bc_sol_left - bl_sol_right;
					fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( (L - 1) + (L + 1) * 0 )) = bc_der_right - br_der_left;
				}
				}
			}
		}
	}
	
    return fvec;
}

// GWRM function
Eigen::VectorXd gwrm(const Eigen::VectorXd x) {
    int nelem = x.size();
	double sum;
	double al_sol_right, al_sol_left, ac_sol_left, ac_sol_right, ar_sol_left, ar_sol_right;
	double al_der_right, al_der_left, ac_der_left, ac_der_right, ar_der_left, ar_der_right;
	double bl_sol_right, bl_sol_left, bc_sol_left, bc_sol_right, br_sol_left, br_sol_right;
	double bl_der_right, bl_der_left, bc_der_left, bc_der_right, br_der_left, br_der_right;
	vector<double> al_tmp((K + 1));
	vector<double> ac_tmp((K + 1));
	vector<double> ar_tmp((K + 1));
	vector<double> bl_tmp((K + 1));
	vector<double> bc_tmp((K + 1));
	vector<double> br_tmp((K + 1));
	bool is_integration = false;
    Eigen::VectorXd fvec = Eigen::VectorXd::Zero(nelem);
	Array3D a = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	Array3D b = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	
	for (sx = 0; sx < Nx; sx++) {
		for (sy = 0; sy < Ny; sy++) {
			for (int i = 0; i < K+1; i++) {
				for (int j = 0; j < L+1; j++) {
					for (int k = 0; k < M+1; k++) {
						a[0 * N + i + (K + 1) * ( j + (L + 1) * k )] = x((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
						b[1 * N + i + (K + 1) * ( j + (L + 1) * k )] = x((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
					}	
				}
			}
			
			// derivatives
			Array3D ax = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, a, ax, BMAx[sx]);
			Array3D axx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_x_derivative_3D_array(K, L, M, ax, axx, BMAx[sx]);
			Array3D ay = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, a, ay, BMAy[sy]);
			Array3D ayy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_y_derivative_3D_array(K, L, M, ay, ayy, BMAy[sy]);
			Array3D at(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  
			
			Array3Dbx = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, b, bx, BMAx[sx]);
			Array3D bxx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_x_derivative_3D_array(K, L, M, bx, bxx, BMAx[sx]);
			Array3D by = Array3D(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, b, by, BMAy[sy]);
			Array3D byy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_y_derivative_3D_array(K, L, M, by, byy, BMAy[sy]);
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
								fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k )) = at[i][j][k] + a_ax[i][j][k] + b_ay[i][j][k] - nu * (axx[i][j][k] + ayy[i][j][k]);
								fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k )) = bt[i][j][k] + a_bx[i][j][k] + b_by[i][j][k] - nu * (bxx[i][j][k] + byy[i][j][k]);
						}
					}
				}
				
				// initial condition: M mode
				for (int i = 0; i < K+1; i++) {
					for (int j = 0; j < L+1; j++) {
						sum = 0.5 * a[i][j][0];
						for (int k = 1; k < M+1; k++) { sum += pow(-1,k) * a[i][j][k]; }
						fvec(0 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_as[(sx + Nx * sy ) * (K + 1) * (L + 1) + i + K * j];
					
						sum = 0.5 * b[i][j][0];
						for (int k = 1; k < M+1; k++) { sum += pow(-1,k) * b[i][j][k]; }
						fvec(1 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_bs[(sx + Nx * sy ) * (K + 1) * (L + 1) + i + K * j];
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
								fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k )) = ai[i][j][k] - a[i][j][k];
								fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k )) = bi[i][j][k] - b[i][j][k];
						}
					}
				}
				
				// initial condition: 0th mode
				// initial condition: 0th mode
				for (int i = 0; i < K+1; i++) {
					for (int j = 0; j < L+1; j++) {
						fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) 
							+ 2.0 * init_a[(sx + Nx * sy ) * (K + 1) * (L + 1) + i + K * j];
						fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) 
							+ 2.0 * init_b[(sx + Nx * sy ) * (K + 1) * (L + 1) + i + K * j];
					}
				}
			}
		}
	}
	
	for (sx = 0; sx < Nx; sx++) {
		for (sy = 0; sy < Ny; sy++) {
			// boundary conditions: K and K-1 mode
			for (int j = 0; j < L+1; j++) {
				for (int k = 0; k < M+1; k++) {
					for (int i = 0; i < K+1; i++) {
						ac_tmp[i] = x((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
						bc_tmp[i] = x((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						if (sx == 0) {
							al_tmp[i] = x((Nx - 1 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[i] = x((Nx - 1 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[i] = x((sx + 1 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[i] = x((sx + 1 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						} else if ( sx == Nx - 1) {
							al_tmp[i] = x((sx - 1 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[i] = x((sx - 1 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[i] = x((sx + 1 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[i] = x((sx + 1 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						} else {
							al_tmp[i] = x((sx - 1 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[i] = x((sx - 1 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[i] = x((0 + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[i] = x((0 + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						}
					}
					double [al_sol_right, al_der_right] = echebser1(1.0, al_tmp);
					double [ac_sol_left, ac_der_left] = echebser1(-1.0, ac_tmp);
					double [ac_sol_right, ac_der_right] = echebser1(1.0, ac_tmp);
					double [ar_sol_left, ar_der_left] = echebser1(-1.0, ar_tmp);

					fvec((sx + Nx * sy ) * num_sub + 0 * N + K + (K + 1) * ( j + (L + 1) * 0 )) = ac_sol_left - al_sol_right;
					fvec((sx + Nx * sy ) * num_sub + 0 * N + (K - 1) + (K + 1) * ( j + (L + 1) * 0 )) = ac_der_right - ar_der_left;
						
					double [bl_sol_right, bl_der_right] = echebser1(1.0, bl_tmp);
					double [bc_sol_left, bc_der_left] = echebser1(-1.0, bc_tmp);
					double [bc_sol_right, bc_der_right] = echebser1(1.0, bc_tmp);
					double [br_sol_left, br_der_left] = echebser1(-1.0, br_tmp);

					fvec((sx + Nx * sy ) * num_sub + 1 * N + K + (K + 1) * ( j + (L + 1) * 0 )) = bc_sol_left - bl_sol_right;
					fvec((sx + Nx * sy ) * num_sub + 1 * N + (K - 1) + (K + 1) * ( j + (L + 1) * 0 )) = bc_der_right - br_der_left;
				}
			}
			
			// boundary conditions: L and L-1 mode
			for (int i = 0; i < K+1; i++) {
				for (int k = 0; k < M+1; k++) {
					for (int j = 0; j < L+1; j++) {
						ac_tmp[j] = x((sx + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
						bc_tmp[j] = x((sx + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						if (sy == 0) {
							al_tmp[j] = x((sx + Nx * (Ny - 1)) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[j] = x((sx + Nx * (Ny - 1)) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[j] = x((sx + Nx * (sy + 1)) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[j] = x((sx + Nx * (sy + 1)) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						} else if ( sy == Ny - 1) {
							al_tmp[j] = x((sx + Nx * (sy - 1)) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[j] = x((sx + Nx * (sy - 1)) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[j] = x((sx + Nx * (sy + 1)) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[j] = x((sx + Nx * (sy + 1)) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						} else {
							al_tmp[j] = x((sx + Nx * (sy - 1) ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							bl_tmp[j] = x((sx + Nx * (sy - 1) ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
							ar_tmp[j] = x((sx + Nx * 0 ) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * k ));
							br_tmp[j] = x((sx + Nx * 0 ) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * k ));
						}
					}
					double [al_sol_right, al_der_right] = echebser1(1.0, al_tmp);
					double [ac_sol_left, ac_der_left] = echebser1(-1.0, ac_tmp);
					double [ac_sol_right, ac_der_right] = echebser1(1.0, ac_tmp);
					double [ar_sol_left, ar_der_left] = echebser1(-1.0, ar_tmp);

					fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( L + (L + 1) * 0 )) = ac_sol_left - al_sol_right;
					fvec((sx + Nx * sy ) * num_sub + 0 * N + i + (K + 1) * ( (L - 1) + (L + 1) * 0 )) = ac_der_right - ar_der_left;
						
					double [bl_sol_right, bl_der_right] = echebser1(1.0, bl_tmp);
					double [bc_sol_left, bc_der_left] = echebser1(-1.0, bc_tmp);
					double [bc_sol_right, bc_der_right] = echebser1(1.0, bc_tmp);
					double [br_sol_left, br_der_left] = echebser1(-1.0, br_tmp);

					fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( L + (L + 1) * 0 )) = bc_sol_left - bl_sol_right;
					fvec((sx + Nx * sy ) * num_sub + 1 * N + i + (K + 1) * ( (L - 1) + (L + 1) * 0 )) = bc_der_right - br_der_left;
				}
				}
			}
		}
	}
	
    return fvec;
}
	
int main()
{
	for (int i = 0; i < Nx+1; i++ ) { x_grid[i] = Lx + (Rx - Lx) * i / Nx; }
	for (int i = 0; i < Ny+1; i++ ) { y_grid[i] = Ly + (Ry - Ly) * i / Ny; }
	for (int i = 0; i < Nx; i++) { BMAx[i] = 0.5 * (x_grid[i+1] - x_grid[i]); }
	for (int i = 0; i < Ny; i++) { BMAy[i] = 0.5 * (y_grid[i+1] - y_grid[i]); }

	int num_eq = 2;
	int num_sub = num_eq * (K + 1) * (L + 1) * (M + 1);
	int nelem = Nx * Ny * num_eq * (K + 1) * (L + 1) * (M + 1);
	
    	Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nelem);
    	Eigen::VectorXd x1(nelem);
	vector<double> a((K + 1) * (L + 1) * (M + 1));
	vector<double> b((K + 1) * (L + 1) * (M + 1));
	
	for (int sx = 1; i < Nx+1; sx++) {
		for (int sy = 1; j < Ny+1; sy++) {
			chebyshev_coefficients_2D(K+1, L+1, sx, sy, Nx, Ny, N, u0, init_as, BMAx[i], BPAx[i], BMAy[j], BPAy[j]);
			chebyshev_coefficients_2D(K+1, L+1, sx, sy, Nx, Ny, N, v0, init_bs, BMAx[i], BPAx[i], BMAy[j], BPAy[j]);
		}
	}
	
	for (int sx = 0; sx < Nx; sx++) {
		for (int sy = 0; sy < Ny; sy++) {
			for (int i = 0; i < K+1; i++) {
				for (int j = 0; j < L+1; j++) {
					x0((sx + Nx * sy) * num_sub + 0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * init_as[(sx + Nx * sy) * num_sub + i + K * j];
					x0((sx + Nx * sy) * num_sub + 1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * init_bs[(sx + Nx * sy) * num_sub + i + K * j];
				}
			}
		}
    	}
	
    	clock_t c_start = clock();
	//x1 = newton(x0, gwrm);
	/*
	Eigen::VectorXd dh(nelem);
	Eigen::VectorXd f0(nelem);
	Eigen::VectorXd f1(nelem);
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nelem,nelem);
	f0 = gwrm_linear(x0);
	double h = 0.001;
	for (int j = 0; j < nelem; j++) {
		dh = Eigen::VectorXd::Zero(nelem);
		dh(j) = h;
		x1 = x0 + dh;
		f1 = gwrm(x1);
		for (int i = 0; i < nelem; i++) {
			H(i,j) = (f1(i) - f0(i)) / h;
		}
	}
	H = H.inverse();
	
	
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nelem,nelem);
	for (int j = 0; j < nelem; j++) {
		H(j,j) = 1.0;
	}
	*/
	x1 = quasi_newton(x0, gwrm, H);
	
	
	//x1 = AMFA(x0, gwrm);
	x1 = anderson_acceleration(x0, gwrm);
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
	vector<double> x_plot(x_points);
	vector<double> y_plot(y_points);
	
	for (int i = 0; i < x_points; i++ ) {
		x_plot[i] = Lx + (Rx - Lx) * i / (x_points - 1);
	}
	for (int j = 0; j < y_points; j++ ) {
		y_plot[j] = Ly + (Ry - Ly) * j / (y_points - 1);
	}
	
	// evaluate GWRM solution 
	for (int i = 0; i < x_points; i++) {
		for (int j = 0; j < y_points; j++) {
			data_array[i + x_points * j][0] = x_plot[i];
			data_array[i + x_points * j][1] = y_plot[j];
			data_array[i + x_points * j][2] = eval_chebyshev_series(b, x_plot[i], y_plot[j], Rt);
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
