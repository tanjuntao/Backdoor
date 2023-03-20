# LinkeFL: A Versatile Federated Learning Framework for both Academia and Industry


[![release  - v0.2.0](https://img.shields.io/badge/release_-v0.2.0-blue)](https://)
[![python - 3.8 | 3.9 | 3.10](https://img.shields.io/badge/python-3.8_|_3.9_|_3.10-blue)](https://)
[![license - Apache License 2.0](https://img.shields.io/badge/license-Apache_License_2.0-blue)](https://)
[![mypy - checked](https://img.shields.io/badge/mypy-checked-brightgreen)](https://github.com/python/mypy)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linter - flake8](https://img.shields.io/badge/linter-flake8-orange)](https://flake8.pycqa.org/en/latest/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


LinkeFL is a versatile federated learning framework including both horizonal FL and vertical FL, which is developed by [USTC Linke Lab](http://linke.ustc.edu.cn/).


## Core features
* Simulation of horizonal federated model training on one single machine
* Horizonal federated model training on multiple machines with or without GPU
* Private set intersection (PSI) using RSA Blind signature protocol
* Vertical federated logistic regression model training with Paillier homomorphic encryption
* Vertical federated XGBoost model training
* Vertical federated neural network model training
* Split learning
* Data privacy techniques such as homomorphic encryption, differential privacy





## Installation
### development mode

First install LinkeFL in development mode.

``` shell
git clone git@github.com:Linke-Data/LinkeFL.git && cd LinkeFL && pip3 install -r requirements.txt && pip3 install -e . --no-use-pep517
```

Then remove the generated `.c` files and binary files ending with `*.so` (macOS and Linux) or `*.pyd` (Windows).
``` shell
find . -type f -name "*.c" -exec rm {} \;
find . -type f -name "*.so" -exec rm {} \; # macOS and Linux
find . -type f -name "*.pyd" -exec rm {} \; # Windows
```

### local installation mode

First Install LinkeFL locally.
``` shell
git clone git@github.com:Linke-Data/LinkeFL.git && cd LinkeFL && pip3 install -r requirements.txt && pip3 install . --no-use-pep517
```
Then also remove the generated files.
``` shell
find . -type f -name "*.c" -exec rm {} \;
find . -type f -name "*.so" -exec rm {} \; # macOS and Linux
find . -type f -name "*.pyd" -exec rm {} \; # Windows
```

### build wheels and install
If you need to build Python wheel for software delivering, you can use the `build.sh` script. The generated `.whl` file be within `LinkeFL/dist` directory.
``` shell
git clone git@github.com:Linke-Data/LinkeFL.git && cd LinkeFL && bash build.sh
```
Then you can goto the `dist` directory and install LinkeFL via
``` shell
pip3 install *.whl
```

## Examples
* [Running PSI protocol alone](./examples/psi-alone/README.md)
* [Running verticalLR model training along with PSI](./examples/psi-lr/README.md)
* [Running verticalSBT model training along with PSI](./examples/psi-tree/README.md)
* [Running verticalNN model training along with PSI](./examples/psi-nn/README.md)

## API references
Follow the [API building guide](./docs/README.md) to locally build the API documentation
and then view it using your browser.

## Contribution
Please first read the [developer guide](./docs/dev_guide.md) before making any pull request.

## Contact
If you have any questions, feel free to contact the following core developers:

* Juntao Tan: tjt@mail.ustc.edu.cn
* Yihang Cheng: whcyh@mail.ustc.edu.cn
* Haoran Cheng: chr990315@mail.ustc.edu.cn
* Junhao Wang: junhaow@mail.ustc.edu.cn
