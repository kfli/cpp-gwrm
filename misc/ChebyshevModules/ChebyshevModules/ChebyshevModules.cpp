#include <stdio.h>
#include <iostream>
#include <time.h>
#include <omp.h>

void chebyshev_derivative_3D(int l, int m, int n, float* a, float* b, float BMA) {
    int R = n - 1;
#pragma omp parallel for
    for (int i = 0; i <= l; i++) {
        for (int j = 0; j <= m; j++) {
            b[i + m * (j + n * (n - 1))] = 2 * n * a[i + m * (j + n * n)] / BMA;
            b[i + m * (j + n * (n - 2))] = 2 * (n - 1) * a[i + m * (j + n *(n - 1))] / BMA;
            for (int k = R-1; k > 0; k--) {
                b[i + m * (j + n * (k - 1))] = b[i + m * (j + n * (k + 1))] + 2 * (k - 1) * a[i + m * (j + n * k)] / BMA;
            }
        }
    }
}

void chebyshev_product_1D(int n, float* a, float* b, float* c) {
    float sum;
    for (int i = 0; i < n; i++) {
        sum = 0.f;
        for (int j = 0; j <= i; j++) {
            sum += a[j] * b[j - i];
        }
        for (int j = 1; j < n - i; j++) {
            sum += a[j] * b[j + i] + a[j + i] * b[j];
        }
        c[i] = 0.5f * sum;
    }
}

void chebyshev_product_2D(int l, int m, float* a, float* b, float* c) {
    float sum;
#pragma omp parallel for private(sum) 
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

void chebyshev_product_3D(int l, int m, int n, float* a, float* b, float* c) {
    float sum;
#pragma omp parallel for private(sum) 
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

int main(void) {
    
    int xM = 20, yM = 20, zM = 20;
    int M = (xM + 1) * (yM + 1);
    int N = (xM + 1) * (yM + 1) * (zM + 1);
    float *a2,  *b2,  *c2, * a3, * b3, * c3;
    float BMA = 0.5 * (1 - 0);
    a2 = (float*)malloc((M + 1) * sizeof(float));
    b2 = (float*)malloc((M + 1) * sizeof(float));
    c2 = (float*)malloc((M + 1) * sizeof(float));

    a3 = (float*)malloc((N + 1) * sizeof(float));
    b3 = (float*)malloc((N + 1) * sizeof(float));
    c3 = (float*)malloc((N + 1) * sizeof(float));

    for (int i = 0; i < M+1; i++) {
        a2[i] = 0.1f;
        b2[i] = 0.2f;
    }

    for (int i = 0; i < N + 1; i++) {
        a3[i] = 0.1f;
        b3[i] = 0.2f;
    }

    // Perform chebyprod on N elements
    //chebyprod(N, a, b, c);
    clock_t tStart = clock();
    chebyshev_product_2D(xM, yM, a2, b2, c2);
    printf("Time taken: %.2fs\n", (float)(clock() - tStart) / CLOCKS_PER_SEC);

    tStart = clock();
    chebyshev_product_3D(xM, yM, zM, a3, b3, c3);
    printf("Time taken: %.2fs\n", (float)(clock() - tStart) / CLOCKS_PER_SEC);

    tStart = clock();
    chebyshev_derivative_3D(xM, yM, zM, a3, b3, BMA);
    printf("Time taken: %.2fs\n", (float)(clock() - tStart) / CLOCKS_PER_SEC);

    std::cout << "Vector c: [ ";
    for (int k = 0; k < 10; k++)
        std::cout << c3[k] << " ";
    std::cout << "]\n";

    free(a2);
    free(b2);
    free(c2);
    free(a3);
    free(b3);
    free(c3);
    return 0;
}
