"""linkefl Python package entry point"""
from linkefl.isafe import isafelinke_sparta_validlicense


is_valid = isafelinke_sparta_validlicense()
if not is_valid:
    print('[ERR 10002] Your license is not valid and you are not allowed to use LinkeFL. '
          'Please contact isafelinke at ink@isafetech.cn')
    exit(-1)

VERSION = (0, 1, 1)

__version__ = '.'.join([str(x) for x in VERSION])


