# GWRM - Generalized Weighted Residual Method
# Install development tools 
Ubuntu
1. $ sudo apt-get update
2. $ sudo apt-get install build-essential
3. $ sudo apt-get install gfortran

Windows
1. Install MinGW
2. Install compilers (c++, python, fortran)
3. To use make in PowerShell (mingw32-make.exe or make.exe)
   $ new-item alias:make -value 'C:\Program Files (x86)\GnuWin32\bin\<mingw32-make.exe or make.exe>.exe'

# Install Eigen
1. Download and extract Eigen (Win - C://; Linux - /usr/local/include)
2. Link to Eigen folder in Makefile 
