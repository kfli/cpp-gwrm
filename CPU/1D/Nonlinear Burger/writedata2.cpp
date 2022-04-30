#include <iostream>
#include <fstream>
using namespace std;

int main () {
  int row = 4;
  int col = 3;
  double x[row][col] = { {1,1,4}, {2,2,3}, {3,4,5}, {4,8,2} };

  ofstream myfile ("array_data.txt");
  if (myfile.is_open())
  {
    for(int i = 0; i < row; i++) {
		for(int j = 0; j < col-1; j++) {
        myfile << x[i][j] << ", ";
		}
		myfile << x[i][col-1] << "\n";
    }
    myfile.close();
  }
  else cout << "Unable to open file";
  return 0;
}