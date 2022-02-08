Eigen::VectorXd anderson_picard_acceleration(Eigen::VectorXd& x0, Eigen::VectorXd (*f)(Eigen::VectorXd)) {
    int nelem = x0.size();
	int mk = 0;
	int mmax = 9;
	int periodic = 8;
	double omega = 0.6;
	double beta = 0.6;
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
		g0 = x0 - f0;
		
        err = f0.squaredNorm();
		cout << "k = " << k << "; err = " << err << endl;
		
		if (err < pow(10, -8)) {
			break;
		}
		
		if (k == 0) {
			x0 = x0 + omega * f0;
		} else {
			if (mk < mmax) { 
				D.conservativeResize(D.rows(), D.cols()+1);
				D.col(D.cols()-1) = f0 - f1;
				E.conservativeResize(E.rows(), E.cols()+1);
				E.col(E.cols()-1) = x0 - x1;
			} else {
				D = D.block(0,1,D.rows(),D.cols()-1).eval();
				D.conservativeResize(D.rows(), D.cols()+1);
				D.col(D.cols()-1) = f0 - f1;
				E = E.block(0,1,E.rows(),E.cols()-1).eval();
				E.conservativeResize(E.rows(), E.cols()+1);
				E.col(E.cols()-1) = x0 - x1;
				mk = mk - 1;
			}
			mk = mk + 1;
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
				x0 = x0 + beta * f0 - ((E + beta * D) * (D.transpose() * D).inverse() * D.transpose()) * f0;
				cout << "Anderson acceleration" << endl;
				p = 0;
			} else {
				cout << "Fixed point iteration" << endl;
				x0 = x0 + omega * f0;
				p++;
			}
			
		}
		f1 = f0;
		x1 = x0;

        ++k;
    } while (err > pow(10, -8) && k < 100);
	x1 = x0;
    return x1;
}