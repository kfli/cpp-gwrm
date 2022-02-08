#include <iostream>
#include <math.h>
#include <vector>
#include <Eigen\Dense>
using namespace std;
typedef vector< vector<double> > Matrix;

Matrix matrix_sum(const Matrix& a, const Matrix& b) {
    int nrows = a.size();
    int ncols = a[0].size();
    Matrix c(nrows, vector<double>(ncols));
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    return c;
}

// Pre: a is a non-empty n symmetric matrix, b is a non-empty n vector.
// Returns axb (an n vector).
vector<double> matrix_vector_mult(const Matrix& a, const vector<double>& b) {
    int nrows = a.size();
    int ncols = a[0].size();
    vector<double> c(nrows,0);
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            c[i] += a[i][j] * b[j];
        }
    }
    return c;
}

// Pre: a is a non-empty nxm matrix, b is a non-empty m×p matrix.
// Returns axb (an nxp matrix).
Matrix matrix_matrix_mult(const Matrix& a, const Matrix& b) {
    int n = a.size();
    int m = a[0].size();
    int p = b[0].size();
    Matrix c(n, vector<double>(p, 0));
    for (int j = 0; j < p; ++j) {
        for (int k = 0; k < m; ++k) {
            for (int i = 0; i < n; ++i) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

double det(const Matrix& vect) {
    if(vect.size() != vect[0].size()) {
        throw std::runtime_error("Matrix is not quadratic");
    } 
    int dimension = vect.size();

    if(dimension == 0) {
        return 1;
    }

    if(dimension == 1) {
        return vect[0][0];
    }

    //Formula for 2x2-matrix
    if(dimension == 2) {
        return vect[0][0] * vect[1][1] - vect[0][1] * vect[1][0];
    }

    double result = 0;
    int sign = 1;
    for(int i = 0; i < dimension; i++) {

        //Submatrix
        Matrix subVect(dimension - 1, vector<double> (dimension - 1));
        for(int m = 1; m < dimension; m++) {
            int z = 0;
            for(int n = 0; n < dimension; n++) {
                if(n != i) {
                    subVect[m-1][z] = vect[m][n];
                    z++;
                }
            }
        }

        //recursive call
        result = result + sign * vect[0][i] * det(subVect);
        sign = -sign;
    }

    return result;
}

Matrix transpose(const Matrix& matrix1) {

    //Transpose-matrix: height = width(matrix), width = height(matrix)
    Matrix solution(matrix1[0].size(), vector<double> (matrix1.size()));

    //Filling solution-matrix
    for(size_t i = 0; i < matrix1.size(); i++) {
        for(size_t j = 0; j < matrix1[0].size(); j++) {
            solution[j][i] = matrix1[i][j];
        }
    }
    return solution;
}

Matrix cofactor(const Matrix& vect) {
    if(vect.size() != vect[0].size()) {
        throw std::runtime_error("Matrix is not quadratic");
    } 

    Matrix solution(vect.size(), vector<double> (vect.size()));
    Matrix subVect(vect.size() - 1, vector<double> (vect.size() - 1));

    for(std::size_t i = 0; i < vect.size(); i++) {
        for(std::size_t j = 0; j < vect[0].size(); j++) {

            int p = 0;
            for(size_t x = 0; x < vect.size(); x++) {
                if(x == i) {
                    continue;
                }
                int q = 0;

                for(size_t y = 0; y < vect.size(); y++) {
                    if(y == j) {
                        continue;
                    }

                    subVect[p][q] = vect[x][y];
                    q++;
                }
                p++;
            }
            solution[i][j] = pow(-1, i + j) * det(subVect);
        }
    }
    return solution;
}

Matrix inverse(const Matrix A) {
    double d = 1.0/det(A);
    Matrix solution(A.size(), vector<double> (A.size()));

    if(A.size() == 1){
        vector<double> ans = {0};
        ans[0] = 1.0/det(A);
        solution[0] = ans;
        return solution;
    }

    for(size_t i = 0; i < A.size(); i++) {
        for(size_t j = 0; j < A.size(); j++) {
            solution[i][j] = A[i][j] * d; 
        }
    }

    return transpose(cofactor(solution));
}

void bad_broyden_update(Matrix& H, const vector<double>& s, const vector<double>& y) {
    int nrows = H.size();
    int ncols = H[0].size();
    double scale_factor = 0;
    Matrix H_tmp(nrows, vector<double>(ncols, 0));
    vector<double> v_tmp(nrows, 0);
    v_tmp = matrix_vector_mult(H, y);
    for (int i = 0; i < nrows; ++i) {
        v_tmp[i] = s[i] - v_tmp[i];
        scale_factor += y[i] * y[i];
    }
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            H_tmp[i][j] = v_tmp[i] * y[j] / scale_factor;
        }
    }
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            H[i][j] = H[i][j] + H_tmp[i][j];
        }
    }
}

vector<double> quasi_newton(vector<double>& x0, vector<double> (*f)(vector<double>)) {
    int nelem = x0.size();
    vector<double> Hf(nelem,0); 
    vector<double> s(nelem,0);
    vector<double> y(nelem,0);
	vector<double> dh(nelem,0);

    vector<double> x1(nelem);
    for (int i = 0; i < nelem; ++i) { x1[i] = x0[i]; }

    vector<double> f0(nelem, 0);
    vector<double> f1(nelem, 0);
    f0 = f(x0);
	
	Eigen::MatrixXd J(nelem,nelem);
	Eigen::MatrixXd Jinv(nelem,nelem);
	
	double h = 0.001;
	for (int j = 0; j < nelem; j++) {
		dh = vector<double>(nelem, 0);
		dh[j] = dh[j] + h;
		for (int i = 0; i < nelem; ++i) {
			x1[i] = x0[i] + dh[i];
			}	
		f1 = f(x1);
		for (int i = 0; i < nelem; i++) {
			J(i,j) = (f1[i] - f0[i]) / h;
		}
	}
	Jinv = J.inverse();
	
	 Matrix H(nelem, vector<double>(nelem,0));
	for (int i = 0; i < nelem; ++i) {
		for (int j = 0; j < nelem; ++j) {
			H[i][j] = Jinv(i,j);
		}
	}
    //for (int i = 0; i < nelem; ++i) { H[i][i] = 1.0; }

    double err = 1.0;
    double sum;
    int k = 0;
    do {
        Hf = matrix_vector_mult(H, f0);
        for (int i = 0; i < nelem; ++i) {
            x1[i] = x0[i] - Hf[i];
        }
        f1 = f(x1);
        
        sum = 0;
        for (int i = 0; i < nelem; ++i) {
            s[i] = x1[i] - x0[i];
            y[i] = f1[i] - f0[i];
            sum += pow(f1[i], 2);
        }
        err = sqrt(sum);
        /*
        for (double elem : x1) {
            cout << elem << " ";
        }
        cout << endl;
		*/
        cout << "k = " << k << "; err = " << err << endl;
        
        // "Bad" Broyden
        bad_broyden_update(H, s, y);

        for (int i = 0; i < nelem; ++i) {
            x0[i] = x1[i];
            f0[i] = f1[i];
        }

        ++k;
    } while (err > pow(10, -8) && k < 1000);
    return x1;
}

vector<double> newton(vector<double>& x0, vector<double> (*f)(vector<double>)) {
    int nelem = x0.size();
    Eigen::VectorXd Jinvf(nelem); 
	vector<double> dh(nelem,0);
    vector<double> x1(nelem, 0);
    for (int i = 0; i < nelem; ++i) { x1[i] = x0[i]; }

    vector<double> f0(nelem, 0);
    vector<double> f1(nelem, 0);
    f0 = f(x0);
	
	Eigen::VectorXd fa(nelem);
	for (int i = 0; i < nelem; ++i) {
		fa(i) = f0[i];
	}

    Eigen::MatrixXd J(nelem,nelem);
	Eigen::MatrixXd Jinv(nelem,nelem);
	
	double h = 0.001;
	for (int j = 0; j < nelem; j++) {
		dh = vector<double>(nelem, 0);
		dh[j] = dh[j] + h;
		for (int i = 0; i < nelem; ++i) {
            x1[i] = x0[i] + dh[i];
        }
		f1 = f(x1);
		for (int i = 0; i < nelem; i++) {
			J(i,j) = (f1[i] - f0[i]) / h;
		}
	}

    double err = 1.0;
    double sum;
    int k = 0;
    do {
		Jinv = J.inverse();
        Jinvf = Jinv*fa;

        for (int i = 0; i < nelem; ++i) {
            x1[i] = x0[i] - Jinvf(i);
        }
        f1 = f(x1);
		
        sum = 0;
        for (int i = 0; i < nelem; ++i) {
            sum += pow(f1[i], 2);
        }
        err = sqrt(sum);
		cout << "k = " << k << "; err = " << err << endl;
		
		if (err < pow(10, -8)) {
			break;
		}
		
		for (int i = 0; i < nelem; ++i) {
            x0[i] = x1[i];
            f0[i] = f1[i];
			fa(i) = f0[i];
        }
		
		for (int j = 0; j < nelem; j++) {
			dh = vector<double>(nelem, 0);
			dh[j] = dh[j] + h;
			for (int i = 0; i < nelem; ++i) {
				x1[i] = x0[i] + dh[i];
			}	
			f1 = f(x1);
			for (int i = 0; i < nelem; i++) {
				J(i,j) = (f1[i] - f0[i]) / h;
			}
		}

        ++k;
    } while (err > pow(10, -8) && k < 1000);
    return x1;
}