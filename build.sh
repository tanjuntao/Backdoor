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

# build python package
python3 -m pip install --upgrade pip wheel setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 setup.py build_ext -j 8 --inplace # use 8 threads for parallel compilation
python3 setup.py sdist bdist_wheel

# remove the generated C files and compiled files by Cython
cd linkefl
find . -type f -name "*.c" -exec rm {} \;
find . -type f -name "*.so" -exec rm {} \; # macOS and Linux
find . -type f -name "*.pyd" -exec rm {} \; # Windows
cd ..

echo "Success."