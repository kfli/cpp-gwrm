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
int K = 12, L = 12, M = 3;
int N = (K + 1) * (L + 1) * (M + 1);
int Ne = 7;
double Lx = 0, Rx = 2.0 * PI;
double Ly = 0, Ry = 2.0 * PI;
double Lt = 0, Rt = 0.2;

double BMAx = 0.5 * (Rx - Lx), BPAx = 0.5 * (Rx + Lx);
double BMAy = 0.5 * (Ry - Ly), BPAy = 0.5 * (Ry + Ly);
double BMAt = 0.5 * (Rt - Lt), BPAt = 0.5 * (Rt + Lt);

// Initial condition
vector<double> init_q((K + 1) * (L + 1));
vector<double> init_u((K + 1) * (L + 1));
vector<double> init_v((K + 1) * (L + 1));
vector<double> init_B((K + 1) * (L + 1));
vector<double> init_H((K + 1) * (L + 1));
vector<double> init_p((K + 1) * (L + 1));

double q0(double x, double y) {
	double gamma = 5.0/3.0;
	return pow(gamma,2);
}
double u0(double x, double y) {
	return -sin(y);
}
double v0(double x, double y) {
	return sin(y);
}
double B0(double x, double y) {
	return -sin(y);
}
double H0(double x, double y) {
	return sin(2.0 * y);
}
double p0(double x, double y) {
	double gamma = 5.0/3.0;
	return gamma;
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

void boundary_conditions(Eigen::VectorXd& fvec, const Eigen::VectorXd& x) {
	double sum_right, sum_left;
	double sum_der_right, sum_der_left;
	vector<double> Ktmp((K + 1));
	vector<double> Ltmp((L + 1));
	// boundary conditions: K and K-1 mode
	for (int ne = 0; ne < Ne; ne++) {
		for (int j = 0; j < L+1; j++) {
			for (int k = 0; k < M+1; k++) {
				for (int i = 0; i < K+1; i++) { Ktmp[i] = x(ne * N + i + (K + 1) * ( j + (L + 1) * k )); }
				tie(sum_right, sum_der_right) = echebser1(1.0, Ktmp);
				tie(sum_left, sum_der_left) = echebser1(-1.0, Ktmp);
				fvec(ne * N + (K - 1) + (K + 1) * ( j + (L + 1) * k )) = sum_right - sum_left;
				fvec(ne * N + K + (K + 1) * ( j + (L + 1) * k )) = sum_der_right - sum_der_left;
			}
		}

		// boundary conditions: L and L-1 mode
		for (int i = 0; i < K+1; i++) {
			for (int k = 0; k < M+1; k++) {
				for (int j = 0; j < L+1; j++) { Ltmp[j] = x(ne * N + i + (K + 1) * ( j + (L + 1) * k )); }
				tie(sum_right, sum_der_right) = echebser1(1.0, Ltmp);
				tie(sum_left, sum_der_left) = echebser1(-1.0, Ltmp);
				fvec(ne * N + i + (K + 1) * ( (L - 1) + (L + 1) * k )) = sum_right - sum_left;
				fvec(ne * N + i + (K + 1) * ( L + (L + 1) * k )) = sum_der_right - sum_der_left;
			}
		}
	}
}

// GWRM function
Eigen::VectorXd gwrm_linear(const Eigen::VectorXd x) {
	int nelem = x.size();
  	Eigen::VectorXd fvec = Eigen::VectorXd::Zero(nelem);
  	boundary_conditions(fvec, x);

  	return fvec;
}

// GWRM function
Eigen::VectorXd gwrm(const Eigen::VectorXd x) {
  int nelem = x.size();
	double sum;
  Eigen::VectorXd fvec = Eigen::VectorXd::Zero(nelem);
	Array3D q(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	Array3D u(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	Array3D v(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	Array3D B(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	Array3D H(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	Array3D p(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));
	Array3D psi(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));

	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			for (int k = 0; k < M+1; k++) {
				q[i][j][k] = x(0 * N + i + (K + 1) * ( j + (L + 1) * k ));
				u[i][j][k] = x(1 * N + i + (K + 1) * ( j + (L + 1) * k ));
				v[i][j][k] = x(2 * N + i + (K + 1) * ( j + (L + 1) * k ));
				B[i][j][k] = x(3 * N + i + (K + 1) * ( j + (L + 1) * k ));
				H[i][j][k] = x(4 * N + i + (K + 1) * ( j + (L + 1) * k ));
				p[i][j][k] = x(5 * N + i + (K + 1) * ( j + (L + 1) * k ));
				psi[i][j][k] = x(6 * N + i + (K + 1) * ( j + (L + 1) * k ));
			}
		}
  }

	// derivatives
	Array3D qx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, q, qx, BMAx);
	Array3D qy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, q, qy, BMAy);
	Array3D qt(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));

	Array3D ux(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, u, ux, BMAx);
	Array3D uxx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_x_derivative_3D_array(K, L, M, ux, uxx, BMAx);
	Array3D uy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, u, uy, BMAy);
	Array3D uyy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_y_derivative_3D_array(K, L, M, uy, uyy, BMAy);
	Array3D ut(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));

	Array3D vx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, v, vx, BMAx);
	Array3D vxx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_x_derivative_3D_array(K, L, M, vx, vxx, BMAx);
	Array3D vy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, v, vy, BMAy);
	Array3D vyy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_y_derivative_3D_array(K, L, M, vy, vyy, BMAy);
	Array3D vt(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));

	Array3D Bx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, B, Bx, BMAx);
	Array3D Bxx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, Bx, Bxx, BMAx);
	Array3D By(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, B, By, BMAy);
	Array3D Byy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, By, Byy, BMAy);
	Array3D Bt(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));

	Array3D Hx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, H, Hx, BMAx);
	Array3D Hxx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, Hx, Hxx, BMAx);
	Array3D Hy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, H, Hy, BMAy);
	Array3D Hyy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, Hy, Hyy, BMAy);
	Array3D Ht(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));

	Array3D px(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, p, px, BMAx);
	Array3D py(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, p, py, BMAy);
	Array3D pt(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));

	Array3D psix(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_x_derivative_3D_array(K, L, M, psi, psix, BMAx);
	Array3D psiy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));  chebyshev_y_derivative_3D_array(K, L, M, psi, psiy, BMAy);
	Array3D psit(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0)));

	// products
	// Eq 1
	Array3D u_qx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, u, qx, u_qx);
	Array3D v_qy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, v, qy, v_qy);

	Array3D q_ux(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, q, ux, q_ux);
	Array3D q_vy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, q, vy, q_vy);

	// Eq 2
	Array3D u_ux(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, u, ux, u_ux);
	Array3D v_uy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, v, uy, v_uy);

	Array3D q_px(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, q, px, q_px);

	Array3D H_Hx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K, L, M, H, Hx, H_Hx);
	Array3D q_H_Hx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, q, H_Hx, q_H_Hx);

	Array3D H_By(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K, L, M, H, By, H_By);
	Array3D q_H_By(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, q, H_By, q_H_By);

	// Eq 3
	Array3D u_vx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, u, vx, u_vx);
	Array3D v_vy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, v, vy, v_vy);

	Array3D q_py(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, q, py, q_py);

	Array3D B_Hx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K, L, M, B, Hx, B_Hx);
	Array3D q_B_Hx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, q, B_Hx, q_B_Hx);

	Array3D B_By(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K, L, M, B, By, B_By);
	Array3D q_B_By(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, q, B_By, q_B_By);

	// Eq 4
	Array3D B_vy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, B, vy, B_vy);
	Array3D H_ux(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, H, ux, H_ux);

	Array3D u_Bx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, u, Bx, u_Bx);
	Array3D v_By(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, v, By, v_By);

	// Eq 5
	Array3D H_uy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, H, uy, H_uy);
	Array3D H_vx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, H, vx, H_vx);
	Array3D B_vx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, B, vx, B_vx);

	Array3D u_Hx(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, u, Hx, u_Hx);
	Array3D v_Hy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, v, Hy, v_Hy);

	// Eq 6
	Array3D u_px(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, u, px, u_px);
	Array3D v_py(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, v, py, v_py);

	Array3D p_ux(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, p, ux, p_ux);
	Array3D p_vy(K+1, vector<vector<double>>(L+1, vector<double>(M+1,0))); chebyshev_product_3D_array(K-2, L-2, M, p, vy, p_vy);



	// du/dt + u.du/dx + v.du/dy - nu.( du2/dx2 +  du2/dy2 ) = 0
	// dv/dt + u.dv/dx + v.dv/dy - nu.( dv2/dx2 +  dv2/dy2 ) = 0
	double nu = 0.1;
	double gamma = 5.0/3.0;
	double c_h = 1.0;
	double c_p = sqrt(c_h * 0.18);

	chebyshev_z_derivative_3D_array(K, L, M, q, qt, BMAt);
	chebyshev_z_derivative_3D_array(K, L, M, u, ut, BMAt);
	chebyshev_z_derivative_3D_array(K, L, M, v, vt, BMAt);
	chebyshev_z_derivative_3D_array(K, L, M, B, Bt, BMAt);
	chebyshev_z_derivative_3D_array(K, L, M, H, Ht, BMAt);
	chebyshev_z_derivative_3D_array(K, L, M, p, pt, BMAt);
	chebyshev_z_derivative_3D_array(K, L, M, psi, psit, BMAt);

	for (int i = 0; i < K-1; i++) {
		for (int j = 0; j < L-1; j++) {
			for (int k = 0; k < M; k++) {
				fvec(0 * N + i + (K + 1) * ( j + (L + 1) * k )) = qt[i][j][k] + u_qx[i][j][k] + v_qy[i][j][k] - q_ux[i][j][k] - q_vy[i][j][k];
				fvec(1 * N + i + (K + 1) * ( j + (L + 1) * k )) = ut[i][j][k] + u_ux[i][j][k] + v_uy[i][j][k] + q_px[i][j][k] + q_H_Hx[i][j][k] - q_H_By[i][j][k] - nu * (uxx[i][j][k] + uyy[i][j][k]);
				fvec(2 * N + i + (K + 1) * ( j + (L + 1) * k )) = vt[i][j][k] + u_vx[i][j][k] + v_vy[i][j][k] + q_py[i][j][k] - q_B_Hx[i][j][k] + q_B_By[i][j][k] - nu * (vxx[i][j][k] + vyy[i][j][k]);
				fvec(3 * N + i + (K + 1) * ( j + (L + 1) * k )) = Bt[i][j][k] + B_vy[i][j][k] - H_uy[i][j][k] + u_Bx[i][j][k] + v_By[i][j][k] + psix[i][j][k] - 0.0 * (Bxx[i][j][k] + Byy[i][j][k]);
				fvec(4 * N + i + (K + 1) * ( j + (L + 1) * k )) = Ht[i][j][k] + H_ux[i][j][k] - B_vx[i][j][k] + u_Hx[i][j][k] + v_Hy[i][j][k] + psiy[i][j][k] - 0.0 * (Hxx[i][j][k] + Hyy[i][j][k]);
				fvec(5 * N + i + (K + 1) * ( j + (L + 1) * k )) = pt[i][j][k] + u_px[i][j][k] + v_py[i][j][k] + gamma * (p_ux[i][j][k] + p_vy[i][j][k]);
				fvec(6 * N + i + (K + 1) * ( j + (L + 1) * k )) = psit[i][j][k] + (pow(c_h,2) / pow(c_p,2)) * psi[i][j][k] + pow(c_h,2) * (Bx[i][j][k] + Hy[i][j][k]);
			}
		}
	}

	// initial condition: M mode
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			sum = 0.5 * q[i][j][0];
			for (int k = 1; k < M+1; k++) { sum += pow(-1.0,k) * q[i][j][k]; }
			fvec(0 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_q[i + (K + 1) *  j];

			sum = 0.5 * u[i][j][0];
			for (int k = 1; k < M+1; k++) { sum += pow(-1.0,k) * u[i][j][k]; }
			fvec(1 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_u[i + (K + 1) *  j];

			sum = 0.5 * v[i][j][0];
			for (int k = 1; k < M+1; k++) { sum += pow(-1.0,k) * v[i][j][k]; }
			fvec(2 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_v[i + (K + 1) *  j];

			sum = 0.5 * B[i][j][0];
			for (int k = 1; k < M+1; k++) { sum += pow(-1.0,k) * B[i][j][k]; }
			fvec(3 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_B[i + (K + 1) *  j];

			sum = 0.5 * H[i][j][0];
			for (int k = 1; k < M+1; k++) { sum += pow(-1.0,k) * H[i][j][k]; }
			fvec(4 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_H[i + (K + 1) *  j];

			sum = 0.5 * p[i][j][0];
			for (int k = 1; k < M+1; k++) { sum += pow(-1.0,k) * p[i][j][k]; }
			fvec(5 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - init_p[i + (K + 1) *  j];

			sum = 0.5 * psi[i][j][0];
			for (int k = 1; k < M+1; k++) { sum += pow(-1.0,k) * psi[i][j][k]; }
			fvec(6 * N + i + (K + 1) * ( j + (L + 1) * M )) = sum - 0.0;
		}
	}

	boundary_conditions(fvec, x);

  return fvec;
}

int main()
{
	cout << "*** STEP 1: GWRM STARTED *** \n";
	int nelem = Ne * (K + 1) * (L + 1) * (M + 1);
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nelem);
  Eigen::VectorXd x1(nelem);
	vector<double> a((K + 1) * (L + 1) * (M + 1));
	vector<double> b((K + 1) * (L + 1) * (M + 1));

	chebyshev_coefficients_2D(K+1, L+1, q0, init_q, BMAx, BPAx, BMAy, BPAy);
	chebyshev_coefficients_2D(K+1, L+1, u0, init_u, BMAx, BPAx, BMAy, BPAy);
	chebyshev_coefficients_2D(K+1, L+1, v0, init_v, BMAx, BPAx, BMAy, BPAy);
	chebyshev_coefficients_2D(K+1, L+1, B0, init_B, BMAx, BPAx, BMAy, BPAy);
	chebyshev_coefficients_2D(K+1, L+1, H0, init_H, BMAx, BPAx, BMAy, BPAy);
	chebyshev_coefficients_2D(K+1, L+1, p0, init_p, BMAx, BPAx, BMAy, BPAy);

	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			x0(0 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * init_q[i + (K + 1) *  j];
			x0(1 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * init_u[i + (K + 1) *  j];
			x0(2 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * init_v[i + (K + 1) *  j];
			x0(3 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * init_B[i + (K + 1) *  j];
			x0(4 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * init_H[i + (K + 1) *  j];
			x0(5 * N + i + (K + 1) * ( j + (L + 1) * 0 )) = 2.0 * init_p[i + (K + 1) *  j];
		}
  }

  clock_t c_start = clock();

	cout << "*** STEP 2: SOLVER STARTED *** \n";
	//x1 = newton(x0, gwrm);
	
	cout << "*** STEP 2.1: COMPUTE INITIAL JACOBIAN *** \n";
	Eigen::VectorXd dh(nelem);
	Eigen::VectorXd f0(nelem);
	Eigen::VectorXd f1(nelem);
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nelem,nelem);
	f0 = gwrm_linear(x0);
	double h = pow(10,-6);
	for (int j = 0; j < nelem; j++) {
		dh = Eigen::VectorXd::Zero(nelem);
		dh(j) = h;
		x1 = x0 + dh;
		f1 = gwrm(x1);
		for (int i = 0; i < nelem; i++) {
			H(i, j) = (f1(i) - f0(i)) / h;
		}
	}
	cout << "*** STEP 2.2: COMPUTE INVERSE *** \n";
	Eigen::MatrixXd I(nelem,nelem);
	I.setIdentity();
	H =  H.colPivHouseholderQr().solve(I);

	cout << "*** STEP 2.3: BEGIN: QUASI NEWTON *** \n";
	x1 = quasi_newton(x0, gwrm, H);
	
	//cout << "*** STEP 3.0: BEGIN: QUASI NEWTON *** \n";
	//Eigen::MatrixXd I(nelem,nelem);
	//x1 = quasi_newton(x0, gwrm, I.setIdentity());
	//cout << "*** STEP 3.0: BEGIN: ANDERSON ACCELERATION *** \n";
	//x1 = anderson_acceleration(x0, gwrm);

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
				a[i + (K + 1) * ( j + (L + 1) * k )] = x1(3 * N + i + (K + 1) * ( j + (L + 1) * k ));
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
