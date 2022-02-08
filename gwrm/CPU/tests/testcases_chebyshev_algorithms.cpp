#include <stdio.h>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <vector>
#include "..\inc\chebyshev_algorithms.h"

int main(void) {
	
	double ATEST[2][2] = vector< vector<double> >(2,2);
    
    int xM = 200, yM = 200, zM = 5;
    int M = (xM + 1) * (yM + 1);
    int N = (xM + 1) * (yM + 1) * (zM + 1);
    double *a2, *b2, *c2;
	double *a3, *b3, *c3;
    double BMA = 0.5 * (1 - 0);
    a2 = (double*)malloc((M + 1) * sizeof(double));
    b2 = (double*)malloc((M + 1) * sizeof(double));
    c2 = (double*)malloc((M + 1) * sizeof(double));

    a3 = (double*)malloc((N + 1) * sizeof(double));
    b3 = (double*)malloc((N + 1) * sizeof(double));
    c3 = (double*)malloc((N + 1) * sizeof(double));
	
	vector<double> a4(xM+1);
	vector<double> b4(xM+1);
	vector<double> c4(xM+1);
	
	Matrix a5(xM+1, vector<double>(yM+1));
	Matrix b5(xM+1, vector<double>(yM+1));
	Matrix c5(xM+1, vector<double>(yM+1));
	
	Array3D a6(xM+1, vector<vector<double>>(yM+1, vector<double>(zM+1)));
	Array3D b6(xM+1, vector<vector<double>>(yM+1, vector<double>(zM+1)));
	Array3D c6(xM+1, vector<vector<double>>(yM+1, vector<double>(zM+1)));

    for (int i = 0; i < M+1; i++) {
        a2[i] = 0.1;
        b2[i] = 0.2;
    }

    for (int i = 0; i < N + 1; i++) {
        a3[i] = 0.1;
        b3[i] = 0.2;
    }
	
	for (int i = 0; i < xM+1; i++) {
		a4[i] = 0.1;
		b4[i] = 0.2;
		for (int j = 0; j < yM+1; j++) {
			a5[i][j] = 1.0;
			b5[i][j] = 1.0;
			for (int k = 0; k < zM+1; k++) {
				a6[i][j][k] = 0.1;
				b6[i][j][k] = 0.2;
			}
		}
    }
	
	printf("Start Tests...\n");
	printf("Chebyshev product: Dimensions = 2; malloc arrays\n");
    clock_t tStart = clock();
    chebyshev_product_2D(xM, yM, a2, b2, c2);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	printf("............................\n");

	printf("Chebyshev product: Dimensions = 3; malloc arrays\n");
    tStart = clock();
    chebyshev_product_3D(xM, yM, zM, a3, b3, c3);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	printf("............................\n");
	/*
	cout << "Vector c3: [ ";
    for (int k = 0; k < 10; k++)
        cout << c3[k] << " ";
    cout << "]\n";
	*/
	printf("Chebyshev product: Array dimensions = 1;\n");
    tStart = clock();
    chebyshev_product_1D_array(xM, a4, b4, c4);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	/*
	for (double elem : c4) {
        cout << elem << " ";
    }
	cout << endl;
	*/
	printf("............................\n");
	
	printf("Chebyshev product: Array dimensions = 2;\n");
    tStart = clock();
    chebyshev_product_2D_array(xM, yM, a5, b5, c5);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	/*
	for (int i = 0; i < xM+1; i++) {
		for (int j = 0; j < yM+1; j++) {
			cout << c5[i][j] << " ";
		}
	}
	cout << endl;
	*/
	printf("............................\n");
	
	printf("Chebyshev product: Array dimensions = 3;\n");
    tStart = clock();
    chebyshev_product_3D_array(xM, yM, zM, a6, b6, c6);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	printf("............................\n");
	/*
	cout << "Vector c6: [ ";
    for (int k = 0; k < 3; k++)
        cout << c6[k][0][0] << " ";
    cout << "]\n";
    */
	b5 = Matrix(xM+1, vector<double>(yM+1,0));
	printf("Chebyshev x-derivative: Dimensions = 2;\n");
    tStart = clock();
    chebyshev_x_derivative_2D_array(xM, yM, a5, b5, 0.5);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	/*
	for (int i = 0; i < xM+1; i++) {
		for (int j = 0; j < yM+1; j++) {
			cout << b5[i][j] << " ";
		}
	}
	cout << endl;
	*/
	printf("............................\n");
	
	b5 = Matrix(xM+1, vector<double>(yM+1,0));
	printf("Chebyshev y-derivative: Dimensions = 2;\n");
    tStart = clock();
    chebyshev_y_derivative_2D_array(xM, yM, a5, b5, 0.5);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	/*
	for (int i = 0; i < xM+1; i++) {
		for (int j = 0; j < yM+1; j++) {
			cout << b5[i][j] << " ";
		}
	}
	cout << endl;
	*/
	printf("............................\n");
	/*
	printf("Chebyshev derivative: Dimensions = 3; malloc arrays\n");
    tStart = clock();
    chebyshev_z_derivative_3D(xM, yM, zM, a3, b3, BMA);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	printf("............................\n");
	*/
    

    free(a2);
    free(b2);
    free(c2);
    free(a3);
    free(b3);
    free(c3);
    return 0;
}
