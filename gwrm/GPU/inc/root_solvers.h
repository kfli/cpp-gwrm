#include <iostream>
#include <math.h>
#include <vector>
#include "..\src\root_solvers.cpp"

using namespace std;
typedef vector< vector<double> > Matrix;

Matrix matrix_sum(const Matrix& a, const Matrix& b);

vector<double> matrix_vector_mult(const Matrix& a, const vector<double>& b);

Matrix matrix_matrix_mult(const Matrix& a, const Matrix& b);

double det(const Matrix& vect);

Matrix transpose(const Matrix& matrix1);

Matrix cofactor(const Matrix& vect);

Matrix inverse(const Matrix A);

void bad_broyden_update(Matrix& H, const vector<double>& s, const vector<double>& y);

vector<double> quasi_newton(vector<double>& x0, vector<double> (*f)(vector<double>));

vector<double> newton(vector<double>& x0, vector<double> (*f)(vector<double>));