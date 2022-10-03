"""Setup.py for code packaging and package installation

The template of this file can be found here:
https://github.com/navdeep-G/setup.py/blob/master/setup.py
"""
import os
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
import sysconfig

from Cython.Build import cythonize

from linkefl import __version__


# Package meta-data.
NAME = 'linkefl'
DESCRIPTION = 'LinkeFL is a versatile federated learning framework developed by USTC LINKE Lab'
AUTHOR_EMAIL = 'tjt@mail.ustc.edu.cn, zhanglan@ustc.edu.cn'
AUTHOR = 'Juntao Tan, Yihang Cheng, Haoran Cheng, Junhao Wang, Ju Huang, ' \
          'Jiahui Huang, Xinben Gao, Anran Li, Lan Zhang'
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

def get_ext_paths(root_dir, exclude_files=None):
    """get filepaths for compilation"""
    paths = []

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != '.py':
                continue

            file_path = os.path.join(root, filename)
            if exclude_files is not None:
                if file_path in exclude_files:
                    continue

            paths.append(file_path)
    return paths

class BuildPy(_build_py):
    def find_package_modules(self, package, package_dir):
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
        modules = super().find_package_modules(package, package_dir)
        filtered_modules = []
        for (pkg, mod, filepath) in modules:
            if os.path.exists(filepath.replace('.py', ext_suffix)):
                continue
            filtered_modules.append((pkg, mod, filepath, ))
        return filtered_modules


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
    install_requires=REQUIRED,
    ext_modules=cythonize(
        get_ext_paths('linkefl'),
        compiler_directives={'language_level': 3}
    ),
    cmdclass={
        'build_py': BuildPy
    }
)