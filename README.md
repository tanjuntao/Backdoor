# LinkeFL: A Versatile Federated Learning Framework for both Academia and Industry

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
git clone git@github.com:Linke-Data/LinkeFL.git && cd LinkeFL && pip3 install Cython>=3.0.0a11 && pip3 install -e .
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
git clone git@github.com:Linke-Data/LinkeFL.git && cd LinkeFL && pip3 install Cython>=3.0.0a11 && pip3 install .
```
Then also remove the generated files.
``` shell
find . -type f -name "*.c" -exec rm {} \;
find . -type f -name "*.so" -exec rm {} \; # macOS and Linux
find . -type f -name "*.pyd" -exec rm {} \; # Windows
```

### build wheels and install
If you need to build Python wheel for software delivering, you can use the `build.sh` script. The generated `.whl` file be within `LinkeFL/sdist` directory.
``` shell
git clone git@github.com:Linke-Data/LinkeFL.git && cd LinkeFL && bash build.sh
```
Then you can goto the `sdist` directory and install LinkeFL via
``` shell
pip3 install *.whl
```

## Examples 
* [Running PSI protocol alone](./examples/psi-alone/README.md)
* [Running verticalLR model training along with PSI](./examples/psi-lr/README.md)
* [Running verticalSBT model training along with PSI](./examples/psi-tree/README.md)
* [Running verticalNN model training along with PSI](./examples/psi-nn/README.md)

## API references 
WIP...

## Contribution
Please first read the [developer guide](./docs/dev_guide.md) before making any pull request. 

## Contact 
If you have any questions, feel free to contact the following core developers:

* Juntao Tan: tjt@mail.ustc.edu.cn
* Haoran Cheng: chr990315@mail.ustc.edu.cn
* Yihang Cheng: whcyh@mail.ustc.edu.cn
* Junhao Wang: junhaow@mail.ustc.edu.cn