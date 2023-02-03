# remove pre-build directories if they exist
if [[ -d build ]]; then
  rm -rf build
fi
if [[ -d dist ]]; then
  rm -rf dist
fi
if [[ -d linkefl.egg-info ]]; then
  rm -rf linkefl.egg-info
fi

TOTAL_ERRORS=0

# obtain cpu cores via python
n_cores=$(python3 -c "import os; print(os.cpu_count())")

# build python package
python3 -m pip install --upgrade pip wheel setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 setup.py build_ext -j $n_cores --inplace # use 8 threads for parallel compilation
TOTAL_ERRORS=$((TOTAL_ERRORS + $?));
python3 setup.py sdist bdist_wheel
TOTAL_ERRORS=$((TOTAL_ERRORS + $?));

# remove the generated C files and compiled files by Cython
cd linkefl
find . -type f -name "*.c" -exec rm {} \;
find . -type f -name "*.so" -exec rm {} \; # macOS and Linux
find . -type f -name "*.pyd" -exec rm {} \; # Windows
cd ..

RED='\033[0;31m'
echo -e "${RED}The exit code of this script is ${TOTAL_ERRORS}"
exit $TOTAL_ERRORS
