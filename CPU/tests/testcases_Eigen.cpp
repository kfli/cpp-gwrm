#include <iostream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <ctime>
#include <Eigen/Dense>
using namespace std;

Eigen::VectorXd f(const Eigen::VectorXd x) {
    int nelem = x.size();
    Eigen::VectorXd fvec(nelem);
	for (int i = 0; i < nelem-1; i++) {
		fvec(i) = x(i) - cos(x[i+1]);
	}
	fvec(nelem-1) = x(nelem-1) - 3 * cos(x[0]);
    return fvec;
}

int main()
{
	int numeq = 2;
    vector<double> x0(numeq,2);
    vector<double> x1(numeq);
	Eigen::VectorXd xe0 = Eigen::VectorXd::Zero(numeq);
	Eigen::VectorXd xe1 = Eigen::VectorXd::Zero(numeq);
	cout << xe1.size() << endl << endl;
	cout << xe0 << endl << endl;
	
	xe1 = f(xe0);
	cout << xe1 << endl << endl;
	
	/* Anderson Acceleration solver */
	Eigen::MatrixXd m(4,4);
	Eigen::VectorXd b(4);
	
	Eigen::MatrixXd D(4,0);
	
	b <<  1, 2, 3, 4;
	m <<  1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16;
	   
	cout << b << endl << endl;
	cout << m << endl << endl;
	
	cout << D.rows() << endl << endl;
	cout << D.cols()+1 << endl << endl;
	
	D = D.block(0,0,D.rows(), D.cols());
	D.conservativeResize(D.rows(), D.cols()+1);
	D.col(D.cols()-1) = b;
	
	cout << D << endl << endl;
	
	cout << "Delete first column" << endl;
	cout << m.block<4,3>(0,1) << endl << endl;
	m = m.block<4,3>(0,1);
	cout << m.rows() << endl;
	cout << m.cols()+1 << endl;
	m.conservativeResize(m.rows(), m.cols()+1);
	cout << m << endl << endl;
	m.col(m.cols()-1) = b;
	cout << m << endl << endl;
    
    return 0;
} 
