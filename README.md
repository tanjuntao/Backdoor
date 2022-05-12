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
We have prepared a single line command which you can copy into your terminal: 
``` shell
git clone git@github.com:Linke-Data/LinkeFL.git && cd LinkeFL && pip3 install -e .
```

## Examples 
* [Running PSI protocol alone](./examples/psi-alone/README.md)
* [Running vertical logistic regression model training](./examples/psi-lr/README.md)
* more examples WIP...

## API references 
WIP...

## Contribution
Please first read the [developer guide](./docs/dev_guide.md) before making any pull request. 

## Contact 
If you have any questions, feel free to contact the following core developers:

* Juntao Tan: tjt@mail.ustc.edu.cn
* Junhao Wang: junhaow@mail.ustc.edu.cn
* Haoran Cheng:
* Yihang Cheng: