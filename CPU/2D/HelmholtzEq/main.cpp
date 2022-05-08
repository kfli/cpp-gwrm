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
int K = 22, L = 22;
int N = (K + 1) * (L + 1);
int Ne = 1;
double Lx = 0.0, Rx = 10.0;
double Ly = 0.0, Ry = 10.0;

double BMAx = 0.5 * (Rx - Lx), BPAx = 0.5 * (Rx + Lx);
double BMAy = 0.5 * (Ry - Ly), BPAy = 0.5 * (Ry + Ly);

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

double eval_chebyshev_series(const vector<double> x, const double xp, const double yp) {
	int nelem = x.size();
	vector<double> fp(nelem,1);
	vector<double> Tx(K);
	vector<double> Ty(L);
	fp[0] = fp[0]/2;
	Tx = chebyshev_polynomials( (xp - BPAx)/BMAx, K );
	Ty = chebyshev_polynomials( (yp - BPAy)/BMAy, L );
	double sum = 0;
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			sum += fp[i] * fp[j] * x[i + (K + 1) * j] * Tx[i] * Ty[j];
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
			for (int i = 0; i < K+1; i++) { Ktmp[i] = x(ne * N + i + (K + 1) * j); }
			tie(sum_right, sum_der_right) = echebser1(1.0, Ktmp);
			tie(sum_left, sum_der_left) = echebser1(-1.0, Ktmp);
			fvec(ne * N + K + (K + 1) * j) = sum_right - sum_left;
			fvec(ne * N + (K - 1) + (K + 1) * j) = sum_der_right - sum_der_left;
		}

		// boundary conditions: L and L-1 mode
		for (int i = 0; i < K+1; i++) {
			for (int j = 0; j < L+1; j++) { Ltmp[j] = x(ne * N + i + (K + 1) * j); }
			tie(sum_right, sum_der_right) = echebser1(1.0, Ltmp);
			tie(sum_left, sum_der_left) = echebser1(-1.0, Ltmp);
			fvec(ne * N + i + (K + 1) * L) = sum_right - sum_left;
			fvec(ne * N + i + (K + 1) * (L - 1)) = sum_der_right - sum_der_left;
		}
	}
}

// GWRM function
Eigen::VectorXd gwrm(const Eigen::VectorXd x) {
  int nelem = x.size();
	double sum;
	bool is_integration = false;
  Eigen::VectorXd fvec = Eigen::VectorXd::Zero(nelem);
	Matrix a(K+1, vector<double>(L+1,0));

	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			a[i][j] = x(0 * N + i + (K + 1) * j);
		}
  }

	// derivatives
	Matrix ax(K+1, vector<double>(L+1,0)); chebyshev_x_derivative_2D_array(K, L, a, ax, BMAx);
	Matrix axx(K+1, vector<double>(L+1,0)); chebyshev_x_derivative_2D_array(K, L, ax, axx, BMAx);

	Matrix ay(K+1, vector<double>(L+1,0)); chebyshev_y_derivative_2D_array(K, L, a, ay, BMAy);
	Matrix ayy(K+1, vector<double>(L+1,0)); chebyshev_y_derivative_2D_array(K, L, ay, ayy, BMAy);

	vector<double> alpha((K + 1) * (L + 1));
	chebyshev_coefficients_2D(K+1, L+1, alpha0, alpha, BMAx, BPAx, BMAy, BPAy);

	// ( du2/dx2 +  du2/dy2 ) = -s**2 * u + alpha(x,y)
	int s = 1.0;
	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			fvec(0 * N + i + (K + 1) * j) = axx[i][j] + ayy[i][j] + pow(s,2)*a[i][j] - alpha[i + (K + 1) * j];
		}
	}

	boundary_conditions(fvec, x);

  return fvec;
}

int main()
{
	cout << "*** STEP 1: GWRM STARTED *** \n";
	int nelem = Ne * (K + 1) * (L + 1);
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nelem);
  Eigen::VectorXd x1(nelem);
	vector<double> a((K + 1) * (L + 1));

	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			x0(i + (K + 1) * j) = 0.01;
		}
	}

  clock_t c_start = clock();
	//x1 = newton(x0, gwrm);
	/*
	cout << "*** STEP 2: CALCULATE JACOBIAN *** \n";
	Eigen::VectorXd dh(nelem);
	Eigen::VectorXd f0(nelem);
	Eigen::VectorXd f1(nelem);
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nelem,nelem);
	boundary_conditions(f0,x0);
	double h = pow(10,-6);
	for (int j = 0; j < nelem; j++) {
		dh = Eigen::VectorXd::Zero(nelem);
		dh(j) = h;
		x1 = x0 + dh;
		boundary_conditions(f1,x1);
		for (int i = 0; i < nelem; i++) {
			H(i,j) = (f1(i) - f0(i)) / h;
		}
	}

	cout << "*** STEP 3: CALCULATE INVERSE JACOBIAN *** \n";
	Eigen::MatrixXd I(nelem,nelem);
	I.setIdentity();
	H =  H.colPivHouseholderQr().solve(I);

	cout << "*** STEP 4: START QUASI NEWTON *** \n";
	x1 = quasi_newton(x0, gwrm, H);
	*/

	x1 = anderson_acceleration(x0, gwrm);
  clock_t c_end = clock();
  long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
  cout << "CPU time used: " << time_elapsed_ms << " ms\n";

	for (int i = 0; i < K+1; i++) {
		for (int j = 0; j < L+1; j++) {
			a[i + (K + 1) * j] = x1(0 * N + i + (K + 1) * j);
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
			data_array[i + x_points * j][2] = eval_chebyshev_series(a, x_grid[i], y_grid[j]);
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
