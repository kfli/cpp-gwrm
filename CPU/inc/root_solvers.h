#include <iostream>
#include <math.h>
#include <vector>
#include "../src/root_solvers.cpp"

using namespace std;
typedef vector< vector<double> > Matrix;

Eigen::VectorXd quasi_newton(Eigen::VectorXd& x0, Eigen::VectorXd (*f)(Eigen::VectorXd), Eigen::MatrixXd& H);

Eigen::VectorXd newton(Eigen::VectorXd& x0, Eigen::VectorXd (*f)(Eigen::VectorXd));

Eigen::VectorXd anderson_acceleration(Eigen::VectorXd& x0, Eigen::VectorXd (*f)(Eigen::VectorXd));
