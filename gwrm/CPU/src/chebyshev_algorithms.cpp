#include <iostream>
#include <omp.h>
#include <math.h>
#include <vector>
#include <tuple>
using namespace std;
typedef vector<vector<double>> Matrix;
typedef vector<vector<vector<double>>> Array3D;
typedef vector<vector<vector<vector<double>>>> Array4D;

/* ------------------------------------------- */
/*  functions that use flattened Malloc arrays */
/* ------------------------------------------- */
tuple<double, double> echebser1(double x, vector<double>& a) {

	nelem = size(a);
	b0 = a[nelem];
	b1 = 0.0;
	b2 = 0.0;

	c0 = a[nelem];
	c1 = 0.0;
	c2 = 0.0;

	x2 = 2.0 * x;

	for (int i = nelem - 1; i > 0; i--) {
		b2 = b1;
		b1 = b0;
		b0 = a[i] - b2 + x2 * b1;

		if ( i > 1 ) {
		c2 = c1;
		c1 = c0;
		c0 = b0 - c2 + x2 * c1;
		}
	}

	y0 = 0.5 * ( b0 - b2 );
	y1 = c0 - c2;

	return {y0, y1};
}

void chebyshev_coefficients_2D(int M, int N, double (*func)(double, double), vector<double>& c, double BMA1, double BPA1, double BMA2, double BPA2) {
	const double PI = 3.141592653589793238463;
	Matrix cfac(M, vector<double>(N));

	for (int k = 0; k < M; k++) {
		for (int l = 0; l < N; l++) {
			cfac[k][l] = func(cos(PI * (k + 0.5) / M) * BMA1 + BPA1, cos(PI * (l + 0.5) / N) * BMA2 + BPA2);
		}
	}
	double sum;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			sum = 0;
			for (int k = 0; k < M; k++) {
				for (int l = 0; l < N; l++) {
					sum += cfac[k][l] * cos(PI * i * (k + 0.5) / M) * cos(PI * j * (l + 0.5) / N);
				}
			}
		c[i + K * j] = (2.0 / M) * (2.0 / N) * sum;
		}
	}
}

void chebyshev_x_derivative_1D(int l, vector<double>& a, vector<double>& b, double BMAx) {
    int R = l - 1;
	b[l - 1] = 2 * l * a[l * l] / BMAx;
	b[l - 2] = 2 * (l - 1) * a[l - 1] / BMAx;
	for (int k = R-1; k > 0; k--) {
		b[k - 1] = b[k + 1] + 2 * k * a[k] / BMAx;
	}	
}

void chebyshev_y_derivative_2D(int l, int m,  vector<double>& a,  vector<double>& b, double BMAy) {
    int R = m - 1;
#pragma omp parallel for
    for (int i = 0; i <= l; i++) {
		b[i + m * (m - 1)] = 2 * m * a[i + m * m] / BMAy;
		b[i + m * (m - 2)] = 2 * (m - 1) * a[i + m * (m - 1)] / BMAy;
		for (int k = R-1; k > 0; k--) {
			b[i + m * (k - 1)] = b[i + m * (k + 1)] + 2 * k * a[i + m * k] / BMAy;
        }
    }
}

void chebyshev_z_derivative_3D(int l, int m, int n,  vector<double>& a,  vector<double>& b, double BMAz) {
    int R = n - 1;
#pragma omp parallel for collapse(2)
    for (int i = 0; i <= l; i++) {
        for (int j = 0; j <= m; j++) {
            b[i + m * (j + n * (n - 1))] = 2 * n * a[i + m * (j + n * n)] / BMAz;
            b[i + m * (j + n * (n - 2))] = 2 * (n - 1) * a[i + m * (j + n *(n - 1))] / BMAz;
            for (int k = R-1; k > 0; k--) {
                b[i + m * (j + n * (k - 1))] = b[i + m * (j + n * (k + 1))] + 2 * k * a[i + m * (j + n * k)] / BMAz;
            }
        }
    }
}

void chebyshev_product_1D(int n,  vector<double>& a,  vector<double>& b,  vector<double>& c) {
    double sum;
    for (int i = 0; i < n; i++) {
        sum = 0.f;
        for (int j = 0; j <= i; j++) {
            sum += a[j] * b[i - j];
        }
        for (int j = 1; j <= n - i; j++) {
            sum += a[j] * b[j + i] + a[j + i] * b[j];
        }
        c[i] = 0.5f * sum;
    }
}

void chebyshev_product_2D(int l, int m,  vector<double>& a,  vector<double>& b,  vector<double>& c) {
    double sum;
#pragma omp parallel for private(sum) collapse(2)
    for (int x = 0; x <= l; x++) {
        for (int y = 0; y <= m; y++) {
            sum = 0.f;
            for (int i = 0; i <= x; i++) {
                for (int j = 0; j <= y; j++) {
                    sum += a[i + m * j] * b[(x - i) + m * (y - j)];
                }
                for (int j = 1; j <= m - y; j++) {
                    sum += a[i + m * j] * b[(x - i) + m * (j + y)]
                        + a[i + m * (j + y)] * b[(x - i) + m * j];
                }
            }
            for (int i = 1; i <= l - x; i++) {
                for (int j = 0; j <= y; j++) {
                    sum += a[i + m * j] * b[(i + x) + m * (y - j)]
                        + a[(i + x) + m * j] * b[i + m * (y - j)];
                }
                for (int j = 1; j <= m - y; j++) {
                    sum += a[i + m * j] * b[(i + x) + m * (j + y)]
                        + a[i + m * (j + y)] * b[(i + x) + m * j]

                        + a[(i + x) + m * j] * b[i + m * (j + y)]
                        + a[(i + x) + m * (j + y)] * b[i + m * j];
                }
            }
            c[x + m * y] = 0.25f * sum;
        }
    }
}

void chebyshev_product_3D(int l, int m, int n, vector<double>& a, vector<double>& b, vector<double>& c) {
    double sum;
#pragma omp parallel for private(sum) collapse(3)
    for (int x = 0; x <= l; x++) {
        for (int y = 0; y <= m; y++) { 
            for (int z = 0; z <= n; z++) {
                sum = 0.f;
                for (int i = 0; i <= x; i++) {
                    for (int j = 0; j <= y; j++) {
                        for (int k = 0; k <= z; k++) {
                            sum += a[i + m * (j + n * k)] * b[(x - i) + m * ((y - j) + n * (z - k))];
                        }
                        for (int k = 1; k <= n - z; k++) {
                            sum += a[i + m * (j + n * k)] * b[(x - i) + m * ((y - j) + n * (k + z))]
                                + a[i + m * (j + n * (k + z))] * b[(x - i) + m * ((y - j) + n * k)];
                        }
                    }
                    for (int j = 1; j <= m - y; j++) {
                        for (int k = 0; k <= z; k++) {
                            sum += a[i + m * (j + n * k)] * b[(x - i) + m * ((j + y) + n * (z - k))]
                                + a[i + m * ((j + y) + n * k)] * b[(x - i) + m * (j + n * (z - k))];
                        }
                        for (int k = 1; k <= n - z; k++) {
                            sum += a[i + m * (j + n * k)] * b[(x - i) + m * ((j + y) + n * (k + z))]
                                + a[i + m * (j + n * (k + z))] * b[(x - i) + m * ((j + y) + n * k)]

                                + a[i + m * ((j + y) + n * k)] * b[(x - i) + m * (j + n * (k + z))]
                                + a[i + m * ((j + y) + n * (k + z))] * b[(x - i) + m * (j + n * k)];
                        }
                    }
                }
                for (int i = 1; i <= l - x; i++) {
                    for (int j = 0; j <= y; j++) {
                        for (int k = 0; k <= z; k++) {
                            sum += a[i + m * (j + n * k)] * b[(i + x) + m * ((y - j) + n * (z - k))]
                                + a[(i + x) + m * (j + n * k)] * b[i + m * ((y - j) + n * (z - k))];
                        }
                        for (int k = 1; k <= n - z; k++) {
                            sum += a[i + m * (j + n * k)] * b[(i + x) + m * ((y - j) + n * (k + z))]
                                + a[i + m * (j + n * (k + z))] * b[(i + x) + m * ((y - j) + n * k)]

                                + a[(i + x) + m * (j + n * k)] * b[i + m * ((y - j) + n * (k + z))]
                                + a[(i + x) + m * (j + n * (k + z))] * b[i + m * ((y - j) + n * k)];
                        }
                    }
                    for (int j = 1; j <= m - y; j++) {
                        for (int k = 0; k <= z; k++) {
                            sum += a[i + m * (j + n * k)] * b[(i + x) + m * ((j + y) + n * (z - k))]
                                + a[i + m * ((j + y) + n * k)] * b[(i + x) + m * (j + n * (z - k))]

                                + a[(i + x) + m * (j + n * k)] * b[i + m * ((j + y) + n * (z - k))]
                                + a[(i + x) + m * ((j + y) + n * k)] * b[i + m * (j + n * (z - k))];
                        }
                        for (int k = 1; k <= n - z; k++) {
                            sum += a[i + m * (j + n * k)] * b[(i + x) + m * ((j + y) + n * (k + z))]
                                + a[i + m * (j + n * (k + z))] * b[(i + x) + m * ((j + y) + n * k)]

                                + a[i + m * ((j + y) + n * k)] * b[(i + x) + m * (j + n * (k + z))]
                                + a[i + m * ((j + y) + n * (k + z))] * b[(i + x) + m * (j + n * k)]

                                + a[(i + x) + m * (j + n * k)] * b[i + m * ((j + y) + n * (k + z))]
                                + a[(i + x) + m * (j + n * (k + z))] * b[i + m * ((j + y) + n * k)]

                                + a[(i + x) + m * ((j + y) + n * k)] * b[i + m * (j + n * (k + z))]
                                + a[(i + x) + m * ((j + y) + n * (k + z))] * b[i + m * (j + n * k)];
                        }
                    }
                }
                c[x + m * (y + n * z)] = 0.125f * sum;
            }
        }
    }
}

/* -------------------------------------------------------- */
/*  functions that use arrays created with the vector class */
/* -------------------------------------------------------- */

/*  Chebyshev coefficients */
void chebyshev_coefficients_1D(int M, double (*func)(double),  vector<double> &c, double BMA, double BPA) {
	const double PI = 3.141592653589793238463;
	vector<double> cfac(M);

	for (int k = 0; k < M; k++) {
		cfac[k] = func(cos(PI * (k + 0.5) / M) * BMA + BPA);
	}
	double sum;
	for (int i = 0; i < M; i++) {
		sum = 0;
		for (int j = 0; j < M; j++) {
			sum += cfac[j] * cos(PI * i * (j + 0.5) / M);
		}
		c[i] = (2.0 / M) * sum;
	}
}

void chebyshev_coefficients_2D_array(int M, int N, double (*func)(double, double),  Matrix &c, double BMA1, double BPA1, double BMA2, double BPA2) {
	const double PI = 3.141592653589793238463;
	Matrix cfac(M, vector<double>(N));

	for (int k = 0; k < M; k++) {
		for (int l = 0; l < N; l++) {
			cfac[k][l] = func(cos(PI * (k + 0.5) / M) * BMA1 + BPA1, cos(PI * (l + 0.5) / N) * BMA2 + BPA2);
		}
	}
	double sum;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			sum = 0;
			for (int k = 0; k < M; k++) {
				for (int l = 0; l < N; l++) {
					sum += cfac[k][l] * cos(PI * i * (k + 0.5) / M) * cos(PI * j * (l + 0.5) / N);
				}
			}
		c[i][j] = (2.0 / M) * (2.0 / N) * sum;
		}
	}
}

/*  Chebyshev derivative */
void chebyshev_x_derivative_1D_array(int l, vector<double> &a, vector<double> &b, double BMAx) {
    int R = l - 1;
	b[l - 1] = 2 * l * a[l * l] / BMAx;
	b[l - 2] = 2 * (l - 1) * a[l - 1] / BMAx;
	for (int k = R-1; k > 0; k--) {
		b[k - 1] = b[k + 1] + 2 * k * a[k] / BMAx;
	}	
}

void chebyshev_x_derivative_2D_array(int l, int m, Matrix &a, Matrix &b, double BMAx) {
    int R = l - 1;
#pragma omp parallel for
    for (int i = 0; i < m+1; i++) {
		b[l - 1][i] = 2 * l * a[l][i] / BMAx;
		b[l - 2][i] = 2 * (l - 1) * a[l - 1][i] / BMAx;
		for (int k = R-1; k > 0; k--) {
			b[k - 1][i] = b[k + 1][i] + 2 * k * a[k][i] / BMAx;
        }
    }
}

void chebyshev_y_derivative_2D_array(int l, int m, Matrix &a, Matrix &b, double BMAy) {
    int R = m - 1;
#pragma omp parallel for
    for (int i = 0; i < l+1; i++) {
		b[i][m - 1] = 2 * m * a[i][m] / BMAy;
		b[i][m - 2] = 2 * (m - 1) * a[i][m - 1] / BMAy;
		for (int k = R-1; k > 0; k--) {
			b[i][k - 1] = b[i][k + 1] + 2 * k * a[i][k] / BMAy;
        }
    }
}

void chebyshev_x_derivative_3D_array(int l, int m, int n, Array3D &a, Array3D &b, double BMAx) {
    int R = l - 1;
#pragma omp parallel for collapse(2)
    for (int i = 0; i <  m+1; i++) {
        for (int j = 0; j <  n+1; j++) {
            b[l - 1][i][j] = 2 * l * a[l][i][j] / BMAx;
            b[l - 2][i][j] = 2 * (l - 1) * a[l - 1][i][j] / BMAx;
            for (int k = R-1; k > 0; k--) {
                b[k - 1][i][j] = b[k + 1] [i][j]+ 2 * k * a[k][i][j] / BMAx;
            }
        }
    }
}

void chebyshev_y_derivative_3D_array(int l, int m, int n, Array3D &a, Array3D &b, double BMAy) {
    int R = m - 1;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < l+1; i++) {
        for (int j = 0; j < n+1; j++) {
            b[i][m - 1][j] = 2 * m * a[i][m][j] / BMAy;
            b[i][m - 2][j] = 2 * (m - 1) * a[i][m - 1][j] / BMAy;
            for (int k = R-1; k > 0; k--) {
                b[i][k - 1][j] = b[i][k + 1][j] + 2 * k * a[i][k][j] / BMAy;
            }
        }
    }
}

void chebyshev_z_derivative_3D_array(int l, int m, int n, Array3D &a, Array3D &b, double BMAz) {
    int R = n - 1;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < l+1; i++) {
        for (int j = 0; j < m+1; j++) {
            b[i][j][n - 1] = 2 * n * a[i][j][n] / BMAz;
            b[i][j][n - 2] = 2 * (n - 1) * a[i][j][n - 1] / BMAz;
            for (int k = R-1; k > 0; k--) {
                b[i][j][k - 1] = b[i][j][k + 1] + 2 * k * a[i][j][k] / BMAz;
            }
        }
    }
}

/*  Chebyshev integration */
void chebyshev_z_integration_3D_array(int l, int m, int n, Array3D &a, Array3D &b, double BMAz) {
	double sum;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < l+1; i++) {
        for (int j = 0; j < m+1; j++) {
			sum = 0;
            for (int k = 1; k < n; k++) {
                b[i][j][k] = BMAz * (a[i][j][k-1] - a[i][j][k+1]) / (2 * k);
				sum += pow(-1,k) * b[i][j][k];
            }
			b[i][j][n] = BMAz * a[i][j][n-1] / (2 * n);
			sum += pow(-1,n) * b[i][j][n];
			b[i][j][0] = -2.0 * sum;
        }
    }
}

/*  Chebyshev product */
void chebyshev_product_1D_array(int n, vector<double> &a, vector<double> &b, vector<double> &c) {
    double sum;
#pragma omp parallel for private(sum)
    for (int i = 0; i <= n; i++) {
        sum = 0;
        for (int j = 0; j <= i; j++) {
            sum += a[j] * b[i - j];
        }
        for (int j = 1; j <= n - i; j++) {
            sum += a[j] * b[j + i] + a[j + i] * b[j];
        }
        c[i] = 0.5 * sum;
    }
}

void chebyshev_product_2D_array(int l, int m, Matrix &a, Matrix &b, Matrix &c) {
    double sum;
#pragma omp parallel for private(sum) collapse(2)
    for (int x = 0; x <= l; x++) {
        for (int y = 0; y <= m; y++) {
            sum = 0;
            for (int i = 0; i <= x; i++) {
                for (int j = 0; j <= y; j++) {
                    sum += a[i][j] * b[x - i][y - j];
                }
                for (int j = 1; j <= m - y; j++) {
                    sum += a[i][j] * b[x - i][j + y]
                        + a[i][j + y] * b[x - i][j];
                }
            }
            for (int i = 1; i <= l - x; i++) {
                for (int j = 0; j <= y; j++) {
                    sum += a[i][j] * b[i + x][y - j]
                        + a[i + x][j] * b[i][y - j];
                }
                for (int j = 1; j <= m - y; j++) {
                    sum += a[i][j] * b[i + x][j + y]
                        + a[i][j + y] * b[i + x][j]

                        + a[i + x][j] * b[i][j + y]
                        + a[i + x][j + y] * b[i][j];
                }
            }
            c[x][y] = 0.25 * sum;
        }
    }
}

void chebyshev_product_3D_array(int l, int m, int n, Array3D &a, Array3D &b, Array3D &c) {
    double sum;
#pragma omp parallel for private(sum) collapse(3)
    for (int x = 0; x <= l; x++) {
        for (int y = 0; y <= m; y++) { 
            for (int z = 0; z <= n; z++) {
                sum = 0;
                for (int i = 0; i <= x; i++) {
                    for (int j = 0; j <= y; j++) {
                        for (int k = 0; k <= z; k++) {
                            sum += a[i][j][k] * b[x - i][y - j][z - k];
                        }
                        for (int k = 1; k <= n - z; k++) {
                            sum += a[i][j][k] * b[x - i][y - j][k + z]
                                + a[i][j][k + z] * b[x - i][y - j][k];
                        }
                    }
                    for (int j = 1; j <= m - y; j++) {
                        for (int k = 0; k <= z; k++) {
                            sum += a[i][j][k] * b[x - i][j + y][z - k]
                                + a[i][j + y][k] * b[x - i][j][z - k];
                        }
                        for (int k = 1; k <= n - z; k++) {
                            sum += a[i][j][k] * b[x - i][j + y][k + z]
                                + a[i][j][k + z] * b[x - i][j + y][k]

                                + a[i][j + y][k] * b[x - i][j][k + z]
                                + a[i][j + y][k + z] * b[x - i][j][k];
                        }
                    }
                }
                for (int i = 1; i <= l - x; i++) {
                    for (int j = 0; j <= y; j++) {
                        for (int k = 0; k <= z; k++) {
                            sum += a[i][j][k] * b[i + x][y - j][z - k]
                                + a[i + x][j][k] * b[i][y - j][z - k];
                        }
                        for (int k = 1; k <= n - z; k++) {
                            sum += a[i][j][k] * b[i + x][y - j][k + z]
                                + a[i][j][k + z] * b[i + x][y - j][k]

                                + a[i + x][j][k] * b[i][y - j][k + z]
                                + a[i + x][j][k + z] * b[i][y - j][k];
                        }
                    }
                    for (int j = 1; j <= m - y; j++) {
                        for (int k = 0; k <= z; k++) {
                            sum += a[i][j][k] * b[i + x][j + y][z - k]
                                + a[i][j + y][k] * b[i + x][j][z - k]

                                + a[i + x][j][k] * b[i][j + y][z - k]
                                + a[i + x][j + y][k] * b[i][j][z - k];
                        }
                        for (int k = 1; k <= n - z; k++) {
                            sum += a[i][j][k] * b[i + x][j + y][k + z]
                                + a[i][j][k + z] * b[i + x][j + y][k]

                                + a[i][j + y][k] * b[i + x][j][k + z]
                                + a[i][j + y][k + z] * b[i + x][j][k]

                                + a[i + x][j][k] * b[i][j + y][k + z]
                                + a[i + x][j][k + z] * b[i][j + y][k]

                                + a[i + x][j + y][k] * b[i][j][k + z]
                                + a[i + x][j + y][k + z] * b[i][j][k];
                        }
                    }
                }
                c[x][y][z] = 0.125 * sum;
            }
        }
    }
}