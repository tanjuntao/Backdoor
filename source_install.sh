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

pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -e . --no-use-pep517

# remove the generated C files and compiled files by Cython
cd linkefl
find . -type f -name "*.c" -exec rm {} \;
find . -type f -name "*.so" -exec rm {} \; # macOS and Linux
find . -type f -name "*.pyd" -exec rm {} \; # Windows
cd ..
