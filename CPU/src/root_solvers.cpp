#include <iostream>
#include <math.h>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/LU>
using namespace std;
typedef vector< vector<double> > Matrix;

Eigen::VectorXd quasi_newton(Eigen::VectorXd& x0, Eigen::VectorXd (*f)(Eigen::VectorXd), Eigen::MatrixXd& H) {
  int nelem = x0.size();
  Eigen::VectorXd s(nelem);
  Eigen::VectorXd y(nelem);
  Eigen::VectorXd x1(nelem);
  x1 = x0;

  Eigen::VectorXd f0(nelem);
  Eigen::VectorXd f1(nelem);

	f0 = f(x0);
  double err = 1.0;
  int k = 0;
  do {
    x1 = x0 - 0.95 * H * f0;
    f1 = f(x1);

    s = x1 - x0;
    y = f1 - f0;

    err = f1.squaredNorm();
    cout << "k = " << k << "; err = " << err << endl;
  	if (err < pow(10, -8)) {
  		break;
  	}

    // "Bad" Broyden
    H = H + (s - H * y) * y.transpose() / (y.transpose() * y);

    x0 = x1;
    f0 = f1;

    ++k;
  } while (err > pow(10, -8) && k < 1000);
    return x1;
}

Eigen::VectorXd newton(Eigen::VectorXd& x0, Eigen::VectorXd (*f)(Eigen::VectorXd)) {
  	int nelem = x0.size();
	double alpha = 1.0;
  	Eigen::VectorXd dh;
  	Eigen::VectorXd x1(nelem);
  	x1 = x0;

  	Eigen::VectorXd f0(nelem);
  	Eigen::VectorXd f1(nelem);

  	Eigen::MatrixXd J(nelem,nelem);
  	Eigen::MatrixXd J1(nelem,nelem);

  	Eigen::MatrixXd I(nelem,nelem);
	I.setIdentity();

	f0 = f(x0);
	double h = pow(10,-6);
	for (int j = 0; j < nelem; j++) {
		dh = Eigen::VectorXd::Zero(nelem);
		dh(j) = dh(j) + h;
		x1 = x0 + dh;
		f1 = f(x1);
		for (int i = 0; i < nelem; i++) {
			J(i,j) = (f1(i) - f0(i)) / h;
		}
	}

    double err = 1.0;
    int k = 0;
    do {
        x1 = x0 - 0.95 * ( J.colPivHouseholderQr().solve(I) ) * f0;
        f1 = f(x1);

        err = f1.squaredNorm();
        cout << "k = " << k << "; err = " << err << endl;
		if (err < pow(10,-8)) {
			break;
		}

        x0 = x1;
        f0 = f1;

		f0 = (1.0 - alpha) * x0 + alpha * f(x0);
		double h = pow(10,-6);
		for (int j = 0; j < nelem; j++) {
			dh = Eigen::VectorXd::Zero(nelem);
			dh(j) = dh(j) + h;
			x1 = x0 + dh;
			f1 = f(x1);
			for (int i = 0; i < nelem; i++) {
				J(i,j) = (f1(i) - f0(i)) / h;
			}
		}
        ++k;
    } while (err > pow(10,-8) && k < 1000);
    return x1;
}

Eigen::VectorXd anderson_acceleration(Eigen::VectorXd& x, Eigen::VectorXd (*f)(Eigen::VectorXd)) {
	int nelem = x.size();
  	double alpha = 1.0;
  	int mk = 0;
	int mmax = 1000;
	int kmax = 10000;
	double beta = 0.01;
	Eigen::VectorXd gamma;
	Eigen::MatrixXd D(nelem,0);
	Eigen::MatrixXd E(nelem,0);

	Eigen::VectorXd x0(nelem);
  	Eigen::VectorXd x1(nelem);
  	Eigen::VectorXd f0(nelem);
  	Eigen::VectorXd f1(nelem);
	Eigen::VectorXd g0(nelem);
  	Eigen::VectorXd g1(nelem);

	x0 = x;

  	double err = 1.0;
  	int k = 0;
	int p = 0;
  	do {
		if (alpha == 1.0){
			f0 = f(x0);
		} else {
			f0 = (1.0 - alpha) * x0 + alpha * f(x0);
		}
		g0 = f0 + x0;

		err = f0.squaredNorm();
		cout << "k = " << k << "; err = " << err << endl;
		if (err < pow(10,-8)) {
			break;
		}
		if (k == 0) {
			x0 = beta * f0 + x0;
		} else {
			if (mk < mmax) {
				D.conservativeResize(D.rows(), D.cols()+1);
				D.col(D.cols()-1) = f0 - f1;
				E.conservativeResize(E.rows(), E.cols()+1);
				E.col(E.cols()-1) = g0 - g1;
			} else {
				D = D.block(0,1,D.rows(),D.cols()-1).eval();
				D.conservativeResize(D.rows(), D.cols()+1);
				D.col(D.cols()-1) = f0 - f1;
				E = E.block(0,1,E.rows(),E.cols()-1).eval();
				E.conservativeResize(E.rows(), E.cols()+1);
				E.col(E.cols()-1) = g0 - g1;
				mk = mk - 1;
			}
			mk = mk + 1;
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(D);
			double cond = svd.singularValues()(0)
				/ svd.singularValues()(svd.singularValues().size()-1);
			cout << "condition number " << cond << "; mk = " << mk << endl;
			while  (cond > pow(10,8) && mk > 2) {
				D = D.block(0,1,D.rows(),D.cols()-1).eval();
				E = E.block(0,1,E.rows(),E.cols()-1).eval();
				mk = mk - 1;
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(D);
				cond = svd.singularValues()(0)
					/ svd.singularValues()(svd.singularValues().size()-1);
				cout << "condition number " << cond << "; mk = " << mk << endl;
			}
			//gamma = D.colPivHouseholderQr().solve(f0);
			/* QR decomposition */
			//x0 = g0 - E * gamma;
			/* SVD */
			gamma = D.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f0);
			x0 = g0 - E * gamma;
			if (beta > 0 && beta < 1) {
				x0 = x0 - (1 - beta) * (f0 - D * gamma);
			}
		}
		f1 = f0;
		g1 = g0;

		++k;
  	} while (err > pow(10,-8) && k < kmax);
		x1 = x0;
  	return x1;
}
