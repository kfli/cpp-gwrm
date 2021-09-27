# gwrm
# Install development tools
$ sudo apt-get update
$ sudo apt-get install build-essential
$ sudo apt-get install gfortran

# Install LAPACK and BLAS
Tutorial :
1. Open browser and download file :http://www.netlib.org/lapack/#_lapack
2. Extract it
3. Open folder in ~/lapack-3.9.0/
4. $ cp make.inc.example make.inc
5. $ make blaslib
6. $ make lapacklib
7. Create link:
$ sudo ln -s $HOME/lapack-3.9.0/librefblas.a /usr/local/lib/libblas.a
$ sudo ln -s $HOME/lapack-3.9.0/liblapack.a /usr/local/lib/liblapack.a
