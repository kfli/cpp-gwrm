#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <ctime>
#include "../inc/root_solvers.h"
#include "../inc/chebyshev_algorithms.h"

using namespace std;
const double PI = 3.141592653589793238463;

double u0(double x, double y) {
		return ( cos(x) + sin(y) );
	}
double v0(double x, double y) {
	return ( cos(x) - sin(y) );
}


// Define global variables
int K = 8, L = 8;
int N = (K + 1) * (L + 1);
int Ne = 2;
int Nm = N * Ne;
int Nx = 3, Ny = 3;
int Nt = Nx * Ny * Ne * N;
double Lx = 0, Rx = 2.0 * PI;
double Ly = 0, Ry = 2.0 * PI;

int main()
{
	vector<double> x_grid(Nx+1);
	vector<double> y_grid(Ny+1);
	vector<double> BMAx(Nx); vector<double> BPAx(Nx);
	vector<double> BMAy(Ny); vector<double> BPAy(Ny);

	for (int i = 0; i < Nx+1; i++ ) { x_grid[i] = Lx + (Rx - Lx) * i / Nx; }
	for (int j = 0; j < Ny+1; j++ ) { y_grid[j] = Ly + (Ry - Ly) * j / Ny; }

	for (int i = 0; i < Nx; i++) { BMAx[i] = 0.5 * (x_grid[i+1] - x_grid[i]); }
	for (int i = 0; i < Nx; i++) { BPAx[i] = 0.5 * (x_grid[i+1] + x_grid[i]); }

	for (int j = 0; j < Ny; j++) { BMAy[j] = 0.5 * (y_grid[j+1] - y_grid[j]); }
	for (int j = 0; j < Ny; j++) { BPAy[j] = 0.5 * (y_grid[j+1] + y_grid[j]); }


	// Initial condition
	vector<double> init_as(Nx * Ny * (K + 1) * (L + 1));
	vector<double> init_bs(Nx * Ny * (K + 1) * (L + 1));

	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			chebyshev_coefficients_2D(K+1, L+1, i, j, Nx, Ny, u0, init_as, BMAx[i], BPAx[i], BMAy[j], BPAy[j]);
			chebyshev_coefficients_2D(K+1, L+1, i, j, Nx, Ny, v0, init_bs, BMAx[i], BPAx[i], BMAy[j], BPAy[j]);
		}
	}

	vector<double> x(Nt);

	for (int sx = 0; sx < Nx; sx++) {
		for (int sy = 0; sy < Ny; sy++) {
			for (int i = 0; i < K+1; i++) {
				for (int j = 0; j < L+1; j++) {
					x[( sx + Nx * sy ) * Nm + 0 * N + i + (K + 1) * j] = init_as[( sx + Nx * sy ) * N + i + (K + 1) * j];
					x[( sx + Nx * sy ) * Nm + 1 * N + i + (K + 1) * j] = init_bs[( sx + Nx * sy ) * N + i + (K + 1) * j];
				}
			}
		}
	}

	double sum;
	double al_sol_right, al_sol_left, ac_sol_left, ac_sol_right, ar_sol_left, ar_sol_right;
	double al_der_right, al_der_left, ac_der_left, ac_der_right, ar_der_left, ar_der_right;
	vector<double> al_tmp(K+1);
	vector<double> ac_tmp(K+1);
	vector<double> ar_tmp(K+1);

	Eigen::VectorXd fvec = Eigen::VectorXd::Zero(Nt);
	for (int ne = 0; ne < Ne; ne++) {
		for (int sx = 0; sx < Nx; sx++) {
			for (int sy = 0; sy < Ny; sy++) {
				// boundary conditions: K and K-1 mode
				for (int j = 0; j < L+1; j++) {
					for (int i = 0; i < K+1; i++) {
						ac_tmp[i] = x[( sx + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
						if ( Nx > 1 ) {
							if (sx == 0) {
								al_tmp[i] = x[( Nx - 1 + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
								ar_tmp[i] = x[( sx + 1 + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
							} else if ( sx == Nx - 1 ) {
								al_tmp[i] = x[( sx - 1 + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
								ar_tmp[i] = x[( 0 + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
							} else {
								al_tmp[i] = x[( sx - 1 + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
								ar_tmp[i] = x[( sx + 1 + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
							}
						} else {
							al_tmp[i] = x[( sx + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
							ar_tmp[i] = x[( sx + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
						}
					}
					tie(al_sol_right, al_der_right) = echebser1(1.0, al_tmp);
					tie(ac_sol_left, ac_der_left) = echebser1(-1.0, ac_tmp);
					tie(ac_sol_right, ac_der_right) = echebser1(1.0, ac_tmp);
					tie(ar_sol_left, ar_der_left) = echebser1(-1.0, ar_tmp);

					fvec[( sx + Nx * sy ) * Nm + ne * N + (K - 1) + (K + 1) * j] = ac_sol_left - al_sol_right;
					fvec[( sx + Nx * sy ) * Nm + ne * N + K + (K + 1) * j] = ac_der_right - ar_der_left;
				}
				
				// boundary conditions: L and L-1 mode
				for (int i = 0; i < K+1; i++) {
					for (int j = 0; j < L+1; j++) {
						ac_tmp[j] = x[( sx + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
						if ( Ny > 1 ) {
							if (sy == 0) {
								al_tmp[j] = x[( sx + Nx * (Ny - 1) ) * Nm + ne * N + i + (K + 1) * j];
								ar_tmp[j] = x[( sx + Nx * (sy + 1) ) * Nm + ne * N + i + (K + 1) * j];
							} else if ( sy == Ny-1 ) {
								al_tmp[j] = x[( sx + Nx * (sy - 1) ) * Nm + ne * N + i + (K + 1) * j];
								ar_tmp[j] = x[( sx + Nx * 0 ) * Nm + ne * N + i + (K + 1) * j];
							} else {
								al_tmp[j] = x[( sx + Nx * (sy - 1) ) * Nm + ne * N + i + (K + 1) * j];
								ar_tmp[j] = x[( sx + Nx * (sy + 1) ) * Nm + ne * N + i + (K + 1) * j];
							}
						} else {
							al_tmp[j] = x[( sx + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
							ar_tmp[j] = x[( sx + Nx * sy ) * Nm + ne * N + i + (K + 1) * j];
						}
						
					}
					tie(al_sol_right, al_der_right) = echebser1(1.0, al_tmp);
					tie(ac_sol_left, ac_der_left) = echebser1(-1.0, ac_tmp);
					tie(ac_sol_right, ac_der_right) = echebser1(1.0, ac_tmp);
					tie(ar_sol_left, ar_der_left) = echebser1(-1.0, ar_tmp);

					fvec[( sx + Nx * sy ) * Nm + ne * N + i + (K + 1) * (L-1)] = ac_sol_left - al_sol_right;
					fvec[( sx + Nx * sy ) * Nm + ne * N + i + (K + 1) * L] = ac_der_right - ar_der_left;
				}
			}
		}
	}

	for (int ne = 0; ne < Ne; ne++) {
		for (int sx = 0; sx < Nx; sx++) {
			for (int sy = 0; sy < Ny; sy++) {
				for (int i = 0; i < K+1; i++) {
					for (int j = 0; j < L+1; j++) {
						if (i > K-2 || j > L-2 ) {
							if ( fvec[( sx + Nx * sy ) * Nm + ne * N + i + (K + 1) * j] > 0.01 ) {
								cout << "sx = " << sx+1 << "; sy = " << sy+1 << " ; i = " << i << "; j = " << j << "; fvec = " << fvec[( sx + Nx * sy ) * Nm + ne * N + i + (K + 1) * j] << "\n";
							}
						}
					}
				}
			}
		}
	}
}