import pickle
import socket
import struct
import time
import os
import platform
from termcolor import colored


from linkefl.base import BaseMessenger
from linkefl.config import BaseConfig
from linkefl.common.const import Const





class FastSocket_multi_v1(BaseMessenger):
    """Implement messenger using python socket

    RSAPSIPassive and RSAPSIActive will only need to maintain two pair sockets, one for RSAPSIPassive
    sending and bob receiving, the other for RSAPSIPassive receiving and RSAPSIActive sending.
    It is much more efficient than `Socket`.
    """
    def __init__(self,
                 role,
                 active_ip,
                 active_port,
                 passive_ip,
                 passive_port,
                 verbose=False,
                 world_size=1):
        """Initialize socket messenger.

        After Initialzation, a daemon socket will run in backend at both RSAPSIPassive
        and RSAPSIActive's side.
        """
        super(FastSocket_multi_v1, self).__init__()
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        self.role = role
        self.verbose = verbose

        # 主动方：是一个list，记录的是所有被动方的ip和port
        #被动方：记录的是自己的ip和port
        self.passive_ip = passive_ip
        self.passive_port = passive_port

        # self.is_connected = False
        # self.is_accepted = False
        self.world_size = world_size



        if self.role == Const.ACTIVE_NAME:
            #主动方连接多个被动方
            self.connect_allclient()
        else:
            # 创建sockt servr，并开始监听
            self.sock_daemon = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            addr = ('0.0.0.0', self.passive_port)
            self.sock_daemon.bind(addr)
            self.sock_daemon.listen(1)
            self.tcpSerSock, addr = self.sock_daemon.accept()

    @classmethod
    def from_config(cls, config):
        assert isinstance(config, BaseConfig), 'config object should be an ' \
                                               'instance of BaseConfig class.'
        return cls(role=config.ROLE,
                   active_ip=config.ACTIVE_IP,
                   active_port=config.ACTIVE_PORT,
                   passive_ip=config.PASSIVE_IP,
                   passive_port=config.PASSIVE_PORT,
                   verbose=config.VERBOSE)

    def connect_allclient(self):
        self.tcpCliSocks = []
        for i in range(self.world_size):
            ip = self.passive_ip[i]
            # print(self.passive_port[i])
            port = self.passive_port[i]

            addr = (ip,port)
            self.sock_daemon = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock_daemon.connect(addr)
            self.tcpCliSocks.append(self.sock_daemon)

    def send(self, msg, id=1):
        if self.role == Const.PASSIVE_NAME:
            self._passive_send(msg)
        else:
            self._active_send(msg,id)

    def recv(self,id=1):
        if self.role == Const.PASSIVE_NAME:
            return self._passive_recv()
        else:
            return self._active_recv(id)

    def close(self):
        if self.role == Const.ACTIVE_NAME:
            for i in range(self.world_size):
                self.tcpCliSocks[i].close()
        else:
            self.sock_daemon.close()

    def _passive_send(self, msg):
        try:
            msg_pickled = pickle.dumps(msg)
            # prefix is the binary representation of the length of pickled message
            prefix = self._msg_prefix(len(msg_pickled))
            msg_send = prefix + msg_pickled
            # self.sock_send.sendall(msg_send)
            # self.sock_daemon.send(msg_send)
            self.tcpSerSock.send(msg_send)
            if self.verbose:
                print('[SOCKET-PASSIVE]: Send message to active party.')
        except pickle.PickleError:
            raise pickle.PickleError(
                "Can't pickle object of type {}".format(type(msg)))

    def _active_send(self,msg,id):
        try:
            Sock = self.tcpCliSocks[id]
            msg_pickled = pickle.dumps(msg)
            prefix = self._msg_prefix(len(msg_pickled))
            msg_send = prefix + msg_pickled
            Sock.sendall(msg_send)
            if self.verbose:
                print('[SOCKET-ACTIVE]: Send message to passive party.')
        except pickle.PickleError:
            raise pickle.PickleError(
                "Can't pickle object of type {}".format(type(msg)))


    def _passive_recv(self):
        raw_msglen = self._recvall(self.tcpSerSock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0] # unpack always returns a tuple
        raw_data = self._recvall(self.tcpSerSock, msglen)
        if self.verbose:
            print('[SOCKET-PASSIVE]: Receive meesage from active party.')

        return pickle.loads(raw_data)

    def _active_recv(self,id):
        raw_msglen = self._recvall(self.tcpCliSocks[id], 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        raw_data = self._recvall(self.tcpCliSocks[id], msglen)
        if self.verbose:
            print('[SOCKET-ACTIVE]: Receive message from passive party.')

        return pickle.loads(raw_data)

    def _msg_prefix(self, msg_len):
        """Prefix each message with its length

        Args:
            msg_len: length of binary message
            '>I': `>` means Big Endian(networking order), `I` means 4
                 bytes unsigned integer

        Returns:
            4 bytes data representing length of a binary message, so maximum
            message size if 4GB
        """
        return struct.pack('>I', msg_len)

    def _recvall(self, sock, n_bytes):
        """Receive specific number of bytes from a socket connection.

        Args:
            sock: Client's side socket object.
            n_bytes: number of bytes to be received.

        Returns:
            Raw data which is a bytearray.
        """
        raw_data = bytearray()
        while len(raw_data) < n_bytes:
            packet = sock.recv(n_bytes - len(raw_data))
            if not packet:
                break
            raw_data.extend(packet)

        return raw_data


class FastSocket_multi_disconnection_v1(FastSocket_multi_v1):
    """Implement messenger using python socket

    RSAPSIPassive and RSAPSIActive will only need to maintain two pair sockets, one for RSAPSIPassive
    sending and bob receiving, the other for RSAPSIPassive receiving and RSAPSIActive sending.
    It is much more efficient than `Socket`.
    """
    def __init__(self,
                 role,
                 model_type,
                 active_ip,
                 active_port,
                 passive_ip,
                 passive_port,
                 verbose=False,
                 world_size=1):
        """Initialize socket messenger.

        After Initialzation, a daemon socket will run in backend at both RSAPSIPassive
        and RSAPSIActive's side.
        """
        super(FastSocket_multi_disconnection_v1, self).__init__(role,
                 active_ip=active_ip,
                 active_port=active_port,
                 passive_ip=passive_ip,
                 passive_port=passive_port,
                 verbose=verbose,
                 world_size=world_size)
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        self.model_type = model_type

    def send(self, msg, id=0,passive_party=True):
        if self.role == Const.PASSIVE_NAME:
            self._passive_send(msg)
        else:
            if passive_party:
                try:
                    self._active_send(msg,id)
                    return True
                except Exception:
                    print(colored("Passive party {} is disconnected", 'red').format(id))
                    return False
            else:
                return False

    def recv(self,id=0,passive_party=True):
        if self.role == Const.PASSIVE_NAME:
            return self._passive_recv()
        else:
            if passive_party:
                try:
                    rec_data = self._active_recv(id)
                except Exception:
                    print(colored("Passive party {} is disconnected", 'red').format(id))
                    return None, False
                passive_party = self._verify_data_integrtiy(rec_data)
                if not passive_party:
                    print(colored("Passive party {} has been disconnected", 'red').format(id))
                return rec_data, passive_party
            else:
                return None, False
            # return self._active_recv()

    def _verify_data_integrtiy(self, data):
        """Judge data integrity by looking at the first data
        """
        try:
            if self.model_type == 'Tree':
                data["name"]  # verify in decision tree
            else:
                data[0]  # verify in NN or LR
            return True
        except Exception:
            return False

    def _close_client(self,id):
        self.tcpCliSocks[id].close()

    def try_reconnect(self, reconnection_port,id=0):

        """Attempt to reconnect, return True if successful
        """
        self._close_client(id)
        try:
            ip = self.passive_ip[id]
            port = reconnection_port[id]
            addr = (ip,port)
            self.sock_daemon = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock_daemon.connect(addr)
            self.tcpCliSocks[id] = self.sock_daemon
            return True
        except Exception:
            return False


