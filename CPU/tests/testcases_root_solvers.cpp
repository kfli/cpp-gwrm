#include <iostream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <ctime>
#include "..\inc\root_solvers.h"
using namespace std;

vector<double> f(const vector<double> x) {
    int nelem = x.size();
    vector<double> fvec(nelem);
	for (int i = 0; i < nelem-1; i++) {
		fvec[i] = x[i] - cos(x[i+1]);
	}
	fvec[nelem-1] = x[nelem-1] - 3 * cos(x[0]);
    return fvec;
}

int main()
{
	int numeq = 100;
    vector<double> x0(numeq,2);
    vector<double> x1(numeq);
	
    clock_t c_start = clock();
    x1 = quasi_newt(x0, f);
    clock_t c_end = clock();

    long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "CPU time used: " << time_elapsed_ms << " ms\n";
    for (double elem : x1) {
        cout << elem << " ";
    }
    cout << endl;
	for (double elem : f(x1)) {
        cout << elem << " ";
    }
    cout << endl;
    return 0;
} 
