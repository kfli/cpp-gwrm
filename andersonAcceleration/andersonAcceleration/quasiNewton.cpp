#include <iostream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <ctime>
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

// Interchanges two values
void swap(int& a, int& b) {
    int c = a;
    a = b;
    b = c;
}
// Pre: m is a square matrix
// Post: m contains the transpose of the input matrix
void Transpose(Matrix& m) {
    int n = m.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            swap(m[i][j], m[j][i]);
        }
    }
}

vector<double> func(const vector<double> x) {
    int nelem = x.size();
    vector<double> fvec(nelem);
    for (int i = 0; i < nelem-1; ++i) {
        fvec[i] = x[i] + cos(x[i + 1]);
    }
    fvec[nelem-1] = x[nelem-1] + 3 * cos(x[0]);
    return fvec;
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

vector<double> quasi_newt(vector<double>& x0) {
    int nelem = x0.size();
    vector<double> Hf(nelem,0); 
    vector<double> s(nelem,0);
    vector<double> y(nelem,0);

    vector<double> x1(nelem);
    for (int i = 0; i < nelem; ++i) { x1[i] = x0[i]; }

    vector<double> f0(nelem, 0);
    vector<double> f1(nelem, 0);
    f0 = func(x0);

    Matrix H(nelem, vector<double>(nelem,0));
    for (int i = 0; i < nelem; ++i) { H[i][i] = 1.0; }

    double err = 1.0;
    double sum;
    int k = 0;
    do {
        Hf = matrix_vector_mult(H, f0);
        for (int i = 0; i < nelem; ++i) {
            x1[i] = x0[i] - Hf[i];
        }
        f1 = func(x1);
        
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
        cout << "k = " << k << "; err = " << err << endl;
        */
        // "Bad" Broyden
        bad_broyden_update(H, s, y);

        for (int i = 0; i < nelem; ++i) {
            x0[i] = x1[i];
            f0[i] = f1[i];
        }

        ++k;
    } while (err > pow(10, -8) && k < 100);
    return x1;
}

int main()
{
    int N = 10;
    vector<double> x0(N,0);
    vector<double> fvec(N);
    vector<double> c(N);

    clock_t c_start = clock();
    c = quasi_newt(x0);
    clock_t c_end = clock();

    long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";
    for (double elem : c) {
        cout << elem << " ";
    }
    cout << endl;
    return 0;
}