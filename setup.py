"""Setup.py for code packaging and package installation

The template of this file can be found here:
https://github.com/navdeep-G/setup.py/blob/master/setup.py
"""
import os
from setuptools import setup, find_packages

from linkefl import __version__


# Package meta-data.
NAME = 'linkefl'
DESCRIPTION = 'LinkeFL is a federated learning framework developed by USTC LINKE Lab'
AUTHOR_EMAIL = 'iamtanjuntao@qq.com, zhanglan@ustc.edu.cn'
AUTHOR = 'Juntao Tan, Haoran Cheng, Yihang Cheng, Junhao Wang, Ju Huang, ' \
          'Xinben Gao, Anran Li, Lan Zhang'
PYTHON_REQUIRES = '>=3.6.0'


HERE = os.path.abspath(os.path.dirname(__file__))
KEYWORDS = ['federated learning', 'vertical federated learning', 'deep learning',
            'privacy-preserving machine learning', 'pytorch', 'paillier', 'gmpy2']
try:
    with open(os.path.join(HERE, 'README.md'), mode='r', encoding='utf-8') as f:
        LONG_DESCRIPTION = f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

try:
    with open(os.path.join(HERE, 'requirements.txt'), 'r') as f:
        REQUIRED = f.read().splitlines()
except FileNotFoundError:
    print('requirements.txt not found in the project root directory, exit.')
    exit(-1)


# where magic happens
setup(
    name=NAME,
    version=__version__,
    author = AUTHOR,
    authod_email=AUTHOR_EMAIL,

    keywords=KEYWORDS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    package_data={'linkefl': ['data/tabular/*.csv']},
    packages=find_packages(include=['linkefl', 'linkefl.*']),
    python_requires=PYTHON_REQUIRES,
    install_requires=REQUIRED
)