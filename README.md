# GWRM - Generalized Weighted Residual Method
# Install development tools 
Ubuntu
1. ```
   bash sudo apt-get update 
   ```
2. ```bash
   sudo apt-get install build-essential
   ```
3. ```bash
   sudo apt-get install gfortran
   ```

Windows
1. Install MinGW
2. Install compilers (c++, python, fortran)
3. To use make in PowerShell (mingw32-make.exe or make.exe)
   $ new-item alias:make -value 'C:\Program Files (x86)\GnuWin32\bin\<mingw32-make.exe or make.exe>.exe'

# Install Eigen
1. Download and extract Eigen (Win - C:\; Linux - /usr/local/include)
2. Link to Eigen folder in Makefile 

# Post-processing
```bash
python -m pip install -U pip
python -m pip install -U matplotlib
```