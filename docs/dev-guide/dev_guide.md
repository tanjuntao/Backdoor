# Developer Guide to LinkeFL

## Project architecture
The source Python code is under the `LinkeFL/linkefl/` directory which is a Python package. Switch into this directory and execute the `tree` command. 
``` shell
$ tree -L 1
.
├── __init__.py       # Python package entry point file 
├── common            # including global constants and factory functions 
├── config            # algorithm configuration in yaml format 
├── crypto            # cryptosystem for encryption, decryption and others
├── data              # raw data
├── dataio            # loading raw data into numpy array or tensor
├── feature           # feature preprocessing and contribution evaluation 
├── hfl               # horizontal federated learning algorithms 
├── messenger         # for communication
├── psi               # private set intersection protocols 
├── splitnn           # split learning algorithms
├── util              # utility functions 
└── vfl               # vertical federated learning algorithms

12 directories, 1 file
```

Each sub-directory under the `linkefl` is also a Python package, and the purpose of each sub-directory is explained in the above code block. 

*If you need to create a seperate sub-directory to put you own algorithms, please contact @tanjuntao first to discuss the arrangement of these algorithms and package structures*.


## Coding style
Python coding style is critical for the whole project, so it's important to write well-styled and consistant-styled Python code. 

It's recommended to follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). There is also a [Chinese version](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents/) available online. 

Code comments is also critical, not only because it makes it easier for other collaborators to understand your code, but also because it can help you to reload your thoughts after a long time. There are several parts that you should consider to leave some comments, which are

* **Python module**: explain what this module does in one sentence with docstring
* **Python function**: explain the purpose of this function, its parameters and types, and return types
* **Python class**: explain what this class does and the parameters and types of `__init__` function
* **Class method**: same as Python function
* **Important code snippet**: use inline comments to explain some non-trivial code




## Using git 
Before you start to contribute your code, you should really learn how to use `git`, because misuse of git cannot bring you any efficiency, in the contrary, it will make you feel painful when you are facing code conflicts, branching errors, etc. 

Here are two learning resources that you can refer:
1. [Liao Xuefeng's git tutorials for beginners](https://www.liaoxuefeng.com/wiki/896043488029600)
2. [Book - Pro Git, Chinese edition](https://www.progit.cn/)

The first one is suitable for beginners to learn some basic concepts in git, e.g., branch, tag, commit. After finishing reading this tutorial, you are ready to use git in your development tour. 

The second one is a comprehensive guide book to git learning. You can view it as a dictionary book and refer to it when you have some specific problems or when you want to dive deep into some topics. 


## Using GitHub to collaborate 
GitHub privides useful functionalities for developers to collaborate with each other, the most important one is called `pull request` (`PR` for short).

A pull request is often used when you want to merge code from one branch into the `master` branch. For example, you create a branch named `alice` and commmit serveral changes, now you want to merge these changes into the `master` branch. At this time, you should *open* a pull request from the `alice` branch and submit it to the `master` branch. This pull request will be *merged* into the `master` if it has no conflicts, otherwise, you should fix these conflicts first before you submit it again. If the pull request is successfully merged, then it can be *closed* by you or other collaborators.

The following two figures demonstrate the workflow of git and GitHub. 

<p align="center">
  <img alt="Light" src="../imgs/git-flow.png" width="48%">
&nbsp;
  <img alt="Dark" src="../imgs/github-flow.png" width="48%">
</p>


The first figure shows the git workflow, which is that when you want to contribute to the repository, you should first create your branch, e.g., `alice` branch or `bob` branch from the `master` branch and then make changes in your own branch. After everything is done, you should create a pull request and merge your branch into the `master` branch. 

The second figure shows the GitHub workflow which is composed of the following steps:

1. **Clone the remote repository**. If the remote repo is not on you local computer, you should clone it first. The default cloned branch is `master`. 

``` shell
$ git clone git@github.com:Linke-Data/LinkeFL.git
```

2. **Pull the latest code**. Each time before you start coding, remember to pull the latest code into your local `master` branch. 


``` shell
$ git pull origin master
```



3. **Create your local development branch**. Make sure not to directly modify the `master` branch, instead, you should create your own branch and contribute code at that branch. The branch name should better be distinguishable from other branchs, for example, you could name it using you own name `alice`. 

``` shell
$ git checkout -b alice
```

4. **Make changes on your local development branch**. Now you are on the local `alice` branch, you can do some changes on it, e.g., adding some features, fixing some bugs. When these changes are done, you should commit it to local repository.

``` shell
$ git add . # add local changes all at once
$ git commit -m "your commit message"
```

5. **Push your local changes to GitHub**. Now it's time to push your local changes to GitHub. Note that you must not directly push local changes from local development branch `alice` to GitHub `master` branch, this is for better code reviewing. Instead, you should push local changes to a GitHub branch which has the same name as your local development branch. 
More concretly, If there is no `alice` branch in the GitHub repo, you should create one and associate it with your local `alice` branch, which can be done by:

    ``` shell
    $ git push -u origin alice:alice
    ```
    After the GitHub `alice` branch is created, you can directly push your local changes to GitHub without the `-u` parameter:
    ``` shell
    $ git push origin alice:alice
    ```


6. **Create a pull request to merge the changes**. Open a pull request in GitHub and check if there exists any conflicts. If not, you should assign the pull request to some collaborators for a review. If the review is passed, then everything is done! You have successfully contributed your code to our master branch! Congratulation!






## Code testing

Before pushing your changes to GitHub, please make sure to test the changed code locally. 
