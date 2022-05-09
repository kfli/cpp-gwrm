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
int K = 5, L = 5;
int N = (K + 1) * (L + 1);
int Ne = 1;
int Nm = N * Ne;
int Nx = 5, Ny = 5;
double Lx = 0.0, Rx = 10.0;
double Ly = 0.0, Ry = 10.0;

vector<double> x_grid(Nx+1);
vector<double> y_grid(Ny+1);
vector<double> BMAx(Nx); vector<double> BPAx(Nx);
vector<double> BMAy(Ny); vector<double> BPAy(Ny);

double alpha0(double x, double y) {
	double x1 = 3.5, x2 = 6.5;
	double y1 = 5.0, y2 = 5.0;
	double omega1 = exp(-10.0 * ( pow((x - x1),2) + pow((y - y1),2)) );
	double omega2 = exp(-10.0 * ( pow((x - x2),2) + pow((y - y2),2)) );
	double s = omega1 + omega2;
	return s;
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

double eval_chebyshev_series(const vector<double> x, int ne, const double xp, const double yp, int sx, int sy) {
	int Nt = x.size();
	vector<double> fp(Nt,1);
	vector<double> Tx(K);
	vector<double> Ty(L);
	fp[0] = fp[0]/2;
	Tx = chebyshev_polynomials( (xp - BPAx[sx])/BMAx[sx], K );
	Ty = chebyshev_polynomials( (yp - BPAy[sy])/BMAy[sy], L );
	double sum = 0;
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			sum += fp[i] * fp[j] * x[( sx + Nx * sy ) * Nm + ne * N + i + (K + 1) * j] * Tx[i] * Ty[j];
		}
	}
	return sum;
}

// GWRM function
Eigen::VectorXd gwrm(const Eigen::VectorXd x) {
    int Nt = x.size();
	double sum;
	double al_sol_right, al_sol_left, ac_sol_left, ac_sol_right, ar_sol_left, ar_sol_right;
	double al_der_right, al_der_left, ac_der_left, ac_der_right, ar_der_left, ar_der_right;
	vector<double> al_tmp(K+1-2);
	vector<double> ac_tmp(K+1-2);
	vector<double> ar_tmp(K+1-2);
    Eigen::VectorXd fvec = Eigen::VectorXd::Zero(Nt);
	Matrix a(K+1, vector<double>(L+1,0));
	vector<double> alpha((K + 1) * (L + 1));
	
	for (int sx = 0; sx < Nx; sx++) {
		for (int sy = 0; sy < Ny; sy++) {
			for (int i = 0; i < K+1; i++) {
				for (int j = 0; j < L+1; j++) {
					a[i][j] = x(( sx + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
				}
			}
			
			// derivatives
			Matrix ax(K+1, vector<double>(L+1,0)); chebyshev_x_derivative_2D_array(K, L, a, ax, BMAx[sx]);
			Matrix axx(K+1, vector<double>(L+1,0)); chebyshev_x_derivative_2D_array(K, L, ax, axx, BMAx[sx]);

			Matrix ay(K+1, vector<double>(L+1,0)); chebyshev_y_derivative_2D_array(K, L, a, ay, BMAy[sy]);
			Matrix ayy(K+1, vector<double>(L+1,0)); chebyshev_y_derivative_2D_array(K, L, ay, ayy, BMAy[sy]);

			chebyshev_coefficients_2D(K+1, L+1, alpha0, alpha, BMAx[sx], BPAx[sx], BMAy[sy], BPAy[sy]);

			// ( du2/dx2 +  du2/dy2 ) = -s**2 * u + alpha(x,y)
			int s = 1.0;
			for (int i = 0; i < K+1; i++) {
				for (int j = 0; j < L+1; j++) {
					fvec(( sx + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j) = axx[i][j] + ayy[i][j] + pow(s,2)*a[i][j] - alpha[i + (K + 1) * j];
				}
			}
		}
	}
	
	for (int sx = 0; sx < Nx; sx++) {
		for (int sy = 0; sy < Ny; sy++) {
			// boundary conditions: K and K-1 mode
			for (int j = 0; j < L+1; j++) {
				for (int i = 0; i < K+1-2; i++) {
					ac_tmp[i] = x(( sx + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
					if ( Nx > 1 ) {
						if (sx == 0) {
							al_tmp[i] = x(( Nx - 1 + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
							ar_tmp[i] = x(( sx + 1 + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
						} else if ( sx == Nx-1 ) {
							al_tmp[i] = x(( sx - 1 + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
							ar_tmp[i] = x(( 0 + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
						} else {
							al_tmp[i] = x(( sx - 1 + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
							ar_tmp[i] = x(( sx + 1 + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
						}
					} else {
						al_tmp[i] = x(( sx + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
						ar_tmp[i] = x(( sx + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
					}
					
				}
				tie(al_sol_right, al_der_right) = echebser1(1.0, al_tmp);
				tie(ac_sol_left, ac_der_left) = echebser1(-1.0, ac_tmp);
				tie(ac_sol_right, ac_der_right) = echebser1(1.0, ac_tmp);
				tie(ar_sol_left, ar_der_left) = echebser1(-1.0, ar_tmp);

				fvec(( sx + Nx * sy ) * Nm + 0 * N + K + (K + 1) * j) = ac_sol_left - al_sol_right;
				fvec(( sx + Nx * sy ) * Nm + 0 * N + (K - 1) + (K + 1) * j) = ac_der_right - ar_der_left;
			}
			
			// boundary conditions: L and L-1 mode
			for (int i = 0; i < K+1; i++) {
				for (int j = 0; j < L+1-2; j++) {
					ac_tmp[j] = x(( sx + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
					if ( Ny > 1 ) {
						if (sy == 0) {
							al_tmp[j] = x(( sx + Nx * (Ny - 1) ) * Nm + 0 * N + i + (K + 1) * j);
							ar_tmp[j] = x(( sx + Nx * (sy + 1) ) * Nm + 0 * N + i + (K + 1) * j);
						} else if ( sy == Ny-1 ) {
							al_tmp[j] = x(( sx + Nx * (sy - 1) ) * Nm + 0 * N + i + (K + 1) * j);
							ar_tmp[j] = x(( sx + Nx * 0 ) * Nm + 0 * N + i + (K + 1) * j);
						} else {
							al_tmp[j] = x(( sx + Nx * (sy - 1) ) * Nm + 0 * N + i + (K + 1) * j);
							ar_tmp[j] = x(( sx + Nx * (sy + 1) ) * Nm + 0 * N + i + (K + 1) * j);
						}
					} else {
						al_tmp[j] = x(( sx + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
						ar_tmp[j] = x(( sx + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j);
					}
					
				}
				tie(al_sol_right, al_der_right) = echebser1(1.0, al_tmp);
				tie(ac_sol_left, ac_der_left) = echebser1(-1.0, ac_tmp);
				tie(ac_sol_right, ac_der_right) = echebser1(1.0, ac_tmp);
				tie(ar_sol_left, ar_der_left) = echebser1(-1.0, ar_tmp);

				fvec(( sx + Nx * sy ) * Nm + 0 * N + i + (K + 1) * L) = 1.0 * ( ac_sol_left - al_sol_right );
				fvec(( sx + Nx * sy ) * Nm + 0 * N + i + (K + 1) * (L - 1)) = 1.0 * ( ac_der_right - ar_der_left );
			}
		}
	}
	
    return fvec;
}
	
int main()
{
	cout << "*** STEP 1: GWRM STARTED *** \n";
	for (int i = 0; i < Nx+1; i++ ) { x_grid[i] = Lx + (Rx - Lx) * i / Nx; }
	for (int i = 0; i < Ny+1; i++ ) { y_grid[i] = Ly + (Ry - Ly) * i / Ny; }

	for (int i = 0; i < Nx; i++) { BMAx[i] = 0.5 * (x_grid[i+1] - x_grid[i]); }
	for (int i = 0; i < Nx; i++) { BPAx[i] = 0.5 * (x_grid[i+1] + x_grid[i]); }

	for (int i = 0; i < Ny; i++) { BMAy[i] = 0.5 * (y_grid[i+1] - y_grid[i]); }
	for (int i = 0; i < Ny; i++) { BPAy[i] = 0.5 * (y_grid[i+1] + y_grid[i]); }

	int Nt = Nx * Ny * Ne * (K + 1) * (L + 1);
	
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(Nt);
    Eigen::VectorXd x1(Nt);
	vector<double> a(Nt);

	cout << "*** STEP 2: START SOLVER *** \n";
    clock_t c_start = clock();
	//x1 = newton(x0, gwrm);
	
	cout << "*** STEP 2.1: COMPUTE INITIAL JACOBIAN *** \n";
	Eigen::VectorXd dh(Nt);
	Eigen::VectorXd f0(Nt);
	Eigen::VectorXd f1(Nt);
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(Nt,Nt);
	f0 = gwrm(x0);
	double h = pow(10,-6);
	for (int j = 0; j < Nt; j++) {
		dh = Eigen::VectorXd::Zero(Nt);
		dh(j) = h;
		x1 = x0 + dh;
		f1 = gwrm(x1);
		for (int i = 0; i < Nt; i++) {
			H(i, j) = (f1(i) - f0(i)) / h;
		}
	}
	cout << "*** STEP 2.2: COMPUTE INVERSE *** \n";
	Eigen::MatrixXd I(Nt,Nt);
	I.setIdentity();
	Eigen::MatrixXd  H_inv =  H.colPivHouseholderQr().solve(I);

	cout << "*** STEP 2.3: BEGIN: QUASI NEWTON *** \n";
	x1 = quasi_newton(x0, gwrm, H_inv);
	
	
	//cout << "*** STEP 3.0: BEGIN: ANDERSON ACCELERATION *** \n";
	//x1 = anderson_acceleration(x0, gwrm);

    clock_t c_end = clock();
    long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "CPU time used: " << time_elapsed_ms << " ms\n";
	
	
	for (int sx = 0; sx < Nx; sx++) {
		for (int sy = 0; sy < Ny; sy++) {
			for (int i = 0; i < K+1; i++) {
				for (int j = 0; j < L+1; j++) {
					a[(sx + Nx * sy) * Nm + 0 * N + i + (K + 1) * j] = x1((sx + Nx * sy) * Nm + 0 * N + i + (K + 1) * j);
				}
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
	for (int sx = 0; sx < Nx; sx++) {
		for (int sy = 0; sy < Ny; sy++) {
			for (int i = 0; i < x_points; i++) {
				for (int j = 0; j < y_points; j++) {
					if ( x_plot[i] >= x_grid[sx] && x_plot[i] <= x_grid[sx+1] &&
						y_plot[j] >= y_grid[sy] && y_plot[j] <= y_grid[sy+1] ) {
							data_array[i + x_points * j][0] = x_plot[i];
							data_array[i + x_points * j][1] = y_plot[j];
							data_array[i + x_points * j][2] = eval_chebyshev_series(a, 0, x_plot[i], y_plot[j], sx, sy);
					}	
				}
			}
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
