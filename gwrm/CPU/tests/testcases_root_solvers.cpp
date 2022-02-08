#include <iostream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <ctime>
#include <Eigen/Dense>
#include "..\inc\root_solvers.h"
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
	int numeq = 1000;
	Eigen::VectorXd x0 = Eigen::VectorXd::Zero(numeq);
	Eigen::VectorXd x1 = Eigen::VectorXd::Zero(numeq);
	
	/* Newton solver */
    clock_t c_start = clock();
    x1 = quasi_newton(x0, f);
    clock_t c_end = clock();

    long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "CPU time used: " << time_elapsed_ms << " ms\n";
	/*
    for (double elem : x1) {
        cout << elem << " ";
    }
    cout << endl;
	for (double elem : f(x1)) {
        cout << elem << " ";
    }
    cout << endl;
	*/
	
	x0 = Eigen::VectorXd::Zero(numeq);
	c_start = clock();
    x1 = AMFA(x0, f);
    c_end = clock();

    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "CPU time used: " << time_elapsed_ms << " ms\n";
	
	/* Anderson Acceleration solver */
	Eigen::MatrixXd m(4,4);
	Eigen::VectorXd b(4);
	b <<  1, 2, 3, 4;
	m <<  1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16;
	cout << "Delete first column" << endl;
	m = m.block(0,1,4,3).eval();
	cout << m << endl << endl;
	cout << "Add another column" << endl;
	m.conservativeResize(Eigen::NoChange, m.cols()+1);
	cout << m << endl << endl;
	m.col(m.cols()-1) = b;
	cout << m << endl << endl;

	x0 = Eigen::VectorXd::Zero(numeq);
	c_start = clock();
    x1 = anderson_acceleration(x0, f);
    c_end = clock();

    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "CPU time used: " << time_elapsed_ms << " ms\n";
	/*
    for (double elem : x1) {
        cout << elem << " ";
    }
    cout << endl;
	for (double elem : f(x1)) {
        cout << elem << " ";
    }
    cout << endl;
	
	x0 = Eigen::VectorXd::Zero(numeq);
	c_start = clock();
    x1 = anderson_picard_acceleration(x0, f);
    c_end = clock();

    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "CPU time used: " << time_elapsed_ms << " ms\n";
    for (double elem : x1) {
        cout << elem << " ";
    }
    cout << endl;
	for (double elem : f(x1)) {
        cout << elem << " ";
    }
    cout << endl;
    */
    return 0;
} 
