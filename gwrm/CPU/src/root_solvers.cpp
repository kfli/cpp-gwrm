#include <iostream>
#include <math.h>
#include <vector>
#include <Eigen/Dense>
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
        x1 = x0 - H*f0;
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
	Eigen::VectorXd dh;
    Eigen::VectorXd x1(nelem);
    x1 = x0;

    Eigen::VectorXd f0(nelem);
    Eigen::VectorXd f1(nelem);

	Eigen::MatrixXd J(nelem,nelem);
	
	f0 = f(x0);
	double h = 0.001;
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
        x1 = x0 - J.inverse()*f0;
        f1 = f(x1);

        err = f1.squaredNorm();
        cout << "k = " << k << "; err = " << err << endl;
		if (err < pow(10, -8)) {
			break;
		}

        x0 = x1;
        f0 = f1;
		
		f0 = f(x0);
		double h = 0.001;
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
    } while (err > pow(10, -8) && k < 1000);
    return x1;
}

Eigen::VectorXd AMFA(Eigen::VectorXd& x0, Eigen::VectorXd (*f)(Eigen::VectorXd)) {
    int nelem = x0.size();
	constexpr float eps = std::numeric_limits<float>::epsilon();
	Eigen::VectorXd d = Eigen::VectorXd::Ones(nelem);
    Eigen::VectorXd x1(nelem);

    Eigen::VectorXd f0(nelem);
	Eigen::VectorXd f1(nelem);
	Eigen::VectorXd p0(nelem);
	Eigen::VectorXd fp0(nelem);
	Eigen::VectorXd sp0(nelem);
	Eigen::VectorXd yp0(nelem);
	Eigen::VectorXd dp0(nelem);
    Eigen::VectorXd z0(nelem);
	Eigen::VectorXd fz0(nelem);
	Eigen::VectorXd dz0(nelem);
	Eigen::VectorXd sz0(nelem);
	Eigen::VectorXd yz0(nelem);

    double err = 1.0;
    int k = 0;
    do {
        f0 = f(x0);

		for (int i = 0; i < nelem; i++) {
			p0(i) = x0(i) - 0.5 * f0(i) / (d(i) + eps);
		}

        fp0 = f(p0);
		sp0 = p0 - x0;
		yp0 = fp0 - f0;
		if (yz0.norm() < 1.0e-8) {
			for (int i = 0; i < nelem; i++) {
				dp0(i) = yp0(i) / (sp0(i) + eps);
			}
		}
		for (int i = 0; i < nelem; i++) {
			z0(i) = x0(i) - f0(i) / (dp0(i) + eps);
		}

		fz0 = f(z0);
		for (int i = 0; i < nelem; i++) {
			dz0(i) = 1.0 / ((2.0 / (dp0(i) + eps)) - (1.0 / (d(i) + eps)));
			x1(i) = z0(i) - fz0(i) / (dz0(i) + eps);
		}
		
		f1 = f(x1);
		
		err = f1.squaredNorm();
        cout << "k = " << k << "; err = " << err << endl;
		if (err < pow(10, -8)) {
			break;
		}
		
		sz0 = x1 - z0;
		yz0 = f1 - fz0;
		if (yz0.norm() < 1.0e-8) {
			for (int i = 0; i < nelem; i++) {
				d(i) = yz0(i) / (sz0(i) + eps);
			}
		}
		
		x0 = x1;
        ++k;
    } while (err > pow(10, -8) && k < 1000);

    return x1;
}

Eigen::VectorXd anderson_acceleration(Eigen::VectorXd& x, Eigen::VectorXd (*f)(Eigen::VectorXd)) {
    int nelem = x.size();
	int mk = 0;
	int mmax = 1000;
	double beta = 1;
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
        f0 = f(x0);
		g0 = f0 + x0;
		
        err = f0.squaredNorm();
		cout << "k = " << k << "; err = " << err << endl;
		
		if (err < pow(10, -8)) {
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
			gamma = D.colPivHouseholderQr().solve(f0);
			/* QR decomposition */
			x0 = g0 - E * gamma;
			/* SVD */
			//x0 = g0 - E*D.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f0);
			if (beta > 0 && beta < 1) {
				x0 = x0 - (1 - beta) * (f0 - D * gamma);
			}
		}
		f1 = f0;
		g1 = g0;

        ++k;
    } while (err > pow(10, -8) && k < 1000);
	x1 = x0;
    return x1;
}


Eigen::VectorXd anderson_picard_acceleration(Eigen::VectorXd& x0, Eigen::VectorXd (*f)(Eigen::VectorXd)) {
    int nelem = x0.size();
	int mk = 0;
	int mmax = 1000;
	int periodic = 50;
	double omega = 0.01;
	double beta = 0.01;
	Eigen::VectorXd gamma;
    Eigen::MatrixXd D(nelem,0); 
	Eigen::MatrixXd E(nelem,0); 
	
    Eigen::VectorXd x1(nelem);
    Eigen::VectorXd f0(nelem);
    Eigen::VectorXd f1(nelem);
	Eigen::VectorXd g0(nelem);
    Eigen::VectorXd g1(nelem);

    double err = 1.0;
    int k = 0;
	int p = 0;
    do {
        f0 = f(x0);
		g0 = f0 + x0;
		
        err = f0.squaredNorm();
		cout << "k = " << k << "; err = " << err << endl;
		
		if (err < pow(10, -8)) {
			break;
		}
		
		if (k == 0) {
			x0 = x0 + omega * f0;
		} else {
			if (mk < mmax) { 
				// Concatenate column vector to the end of matrix;
				D.conservativeResize(D.rows(), D.cols()+1);
				D.col(D.cols()-1) = f0 - f1;
				E.conservativeResize(E.rows(), E.cols()+1);
				E.col(E.cols()-1) = g0 - g1;
			} else {
				// Remove first column of matrix
				D = D.block(0,1,D.rows(),D.cols()-1).eval();
				D.conservativeResize(D.rows(), D.cols()+1);
				D.col(D.cols()-1) = f0 - f1;
				E = E.block(0,1,E.rows(),E.cols()-1).eval();
				E.conservativeResize(E.rows(), E.cols()+1);
				E.col(E.cols()-1) = g0 - g1;
				mk = mk - 1;
			}
			mk = mk + 1;
			
			// Calculate the condition number of matrix
			// Delete first column each time condition number is greater than preset tolerance
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(D);
			double cond = svd.singularValues()(0) 
				/ svd.singularValues()(svd.singularValues().size()-1);
			cout << "condition number " << cond << "; mk = " << mk << endl;
			while  (cond > pow(10,12) && mk > 2) {
				D = D.block(0,1,D.rows(),D.cols()-1).eval();
				E = E.block(0,1,E.rows(),E.cols()-1).eval();
				mk = mk - 1;
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(D);
				cond = svd.singularValues()(0) 
					/ svd.singularValues()(svd.singularValues().size()-1);
				cout << "condition number " << cond << "; mk = " << mk << endl;
			} 

			if (p == periodic) {
				cout << "Fixed point iteration" << endl;
				x0 = x0 + omega * f0;
				p = 0;
			} else {
				cout << "Anderson acceleration" << endl;
				gamma = D.colPivHouseholderQr().solve(f0);
				/* QR decomposition */
				x0 = g0 - E * gamma;
				/* SVD */
				//x0 = g0 - E*D.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f0);
				if (beta > 0 && beta < 1) {
					x0 = x0 - (1 - beta) * (f0 - D * gamma);
				}
				p++;
			}
			
		}
		f1 = f0;
		g1 = g0;

        ++k;
    } while (err > pow(10, -8) && k < 100);
	x1 = x0;
    return x1;
}