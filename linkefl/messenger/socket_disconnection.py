import pickle
import socket
import struct
import time
import zlib

import blosc
from termcolor import colored

from linkefl.base import BaseMessenger
from linkefl.common.const import Const
from linkefl.config import BaseConfig


class FastSocket_disconnection_v1(BaseMessenger):
    def __init__(self,
                 role,
                 model_type,
                 active_ip,
                 active_port,
                 passive_ip,
                 passive_port,
                 verbose=False):
        """Initialize socket messenger.

        After Initialzation, a daemon socket will run in backend at both RSAPSIPassive
        and RSAPSIActive's side.
        """
        socket.setdefaulttimeout(120)
        super(FastSocket_disconnection_v1, self).__init__()
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'

        self.role = role
        self.model_type = model_type
        self.active_ip = active_ip
        self.active_port = active_port
        self.passive_ip = passive_ip
        self.passive_port = passive_port
        self.verbose = verbose

        self.sock_daemon = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.is_connected = False
        self.is_accepted = False

        if self.role == Const.PASSIVE_NAME:
            self.sock_daemon.bind((passive_ip, passive_port))
        else:
            self.sock_daemon.bind((active_ip, active_port))

        self.sock_daemon.listen(120)

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

    def send(self, msg, compress=False, cname='blosc', passive_party_connected=True):

        assert cname in (Const.BLOSC, Const.ZLIB), "invalid compression name"

        if self.role == Const.PASSIVE_NAME:
            self._send(msg, role='passive', compress=compress, cname=cname)
        else:
            # active party makes a disconnection judgment
            if passive_party_connected:
                try:
                    self._send(msg, role='active', compress=compress, cname=cname)
                    return True
                except Exception:
                    print(colored("Passive party is disconnected", 'red'))
                    return False
            else:
                return False

    def recv(self, passive_party_connected=True):
        """Returns receive message and passive party state.
        """
        if self.role == Const.PASSIVE_NAME:
            return self._recv(role='passive')
        else:
            # active party makes a disconnection judgment
            if passive_party_connected:
                try:
                    rec_data = self._recv(role='active')
                except Exception:
                    print(colored("Passive party is disconnected", 'red'))
                    return None, False
                else:
                    # execute after the try part is executed
                    passive_party_connected = self._verify_data_integrtiy(rec_data)
                    if not passive_party_connected:
                        print(colored("Passive party has been disconnected", 'red'))
                    return rec_data, passive_party_connected
            else:
                return None, False

    def close(self):
        self.sock_send.close()
        self.sock_daemon.close()

    def _close_send(self):
        self.sock_send.close()

    def try_reconnect(self, reconnect_port):
        """Attempt to reconnect, return True if successful
        """
        # preprocessing operation
        self._close_send()
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.is_connected, self.is_accepted = False, False
        self.passive_port = reconnect_port

        # try to reconnect
        try:
            self.sock_send.connect((self.passive_ip, self.passive_port))
            self.is_connected = True
            return True
        except Exception:
            return False

    def _verify_data_integrtiy(self, data):
        """Judge data integrity by looking at the first data
        """
        try:
            if self.model_type == 'Tree':
                data["name"]    # verify in decision tree
            else:
                data[0]         # verify in NN or LR
            return True
        except Exception:
            return False

    def _send(self, msg, role, compress=False, cname='blosc'):
        if not self.is_connected:
            # connection channel
            if role == 'active':
                self.sock_send.connect((self.passive_ip, self.passive_port))
            elif role == 'passive':
                self.sock_send.connect((self.active_ip, self.active_port))
            else:
                raise ValueError("Not supported role.")
            self.is_connected = True
        try:
            # compressed data
            msg_binary = pickle.dumps(msg)

            if compress:
                if cname == Const.BLOSC:
                    msg_binary = blosc.compress(msg_binary)
                elif cname == Const.ZLIB:
                    msg_binary = zlib.compress(msg_binary)
                else:
                    pass
            msglen_prefix = self._msglen_prefix(len(msg_binary))
            compress_prefix = self._compress_prefix(compress)
            cname_prefix = self._cname_prefix(cname)

            msg_send = msglen_prefix + compress_prefix + cname_prefix + msg_binary
            self.sock_send.sendall(msg_send)

            if self.verbose:
                if role == 'active':
                    print('[SOCKET-ACTIVE]: Send message to passive party.')
                else:
                    print('[SOCKET-PASSIVE]: Send message to active party.')

        except pickle.PickleError:
            raise pickle.PickleError(
                "Can't pickle object of type {}".format(type(msg)))

    def _recv(self, role):
        if not self.is_accepted:
            self.conn, addr = self.sock_daemon.accept()
            self.is_accepted = True

        msglen, compress, cname = self._recv_prefixes()
        binary_data = self._recvall(self.conn, msglen)
        if compress:
            if cname == Const.BLOSC:
                binary_data = blosc.decompress(binary_data)
            elif cname == Const.ZLIB:
                binary_data = zlib.decompress(binary_data)
            else:
                pass
        msg = pickle.loads(binary_data)

        if self.verbose:
            if role == 'active':
                print('[SOCKET-ACTIVE]: Receive message from passive party.')
            else:
                print('[SOCKET-PASSIVE]: Receive message from active party.')
        return msg


    def _msglen_prefix(self, msg_len):
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

    def _compress_prefix(self, compress):
        """
        Args:
            compress[bool], whether using data compression algo
            '>?': '>' means Big Endian, '?' means python bool type (1 byte)
        """
        return struct.pack('>?', compress)

    def _cname_prefix(self, cname):
        """
        Args:
            cname[str], compression type
            '>B': 'B" means unsigned char (1 byte)
        """
        value = Const.COMPRESSION_DICT[cname]
        return struct.pack('>B', value)

    def _recv_prefixes(self):
        # first 4 bytes means length of msg
        raw_msglen = self._recvall(self.conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]  # unpack always returns a tuple

        # second 1 byte means whether using data compression algorotihm
        raw_compress = self._recvall(self.conn, 1)
        if not raw_compress:
            return None
        compress = struct.unpack('>?', raw_compress)[0]

        # third 1 byte means compresssion type
        raw_cname_value = self._recvall(self.conn, 1)
        if not raw_cname_value:
            return None
        cname_value = struct.unpack('>B', raw_cname_value)[0]
        cname = None
        for key, value in Const.COMPRESSION_DICT.items():
            if value == cname_value:
                cname = key

        return msglen, compress, cname

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
        

class FastSocket_disconnection(BaseMessenger):
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
                 verbose=False):
        """Initialize socket messenger.

        After Initialzation, a daemon socket will run in backend at both RSAPSIPassive
        and RSAPSIActive's side.
        """
        socket.setdefaulttimeout(20)
        super(FastSocket_disconnection, self).__init__()
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        self.role = role
        self.active_ip = active_ip
        self.active_port = active_port
        self.passive_ip = passive_ip
        self.passive_port = passive_port
        self.verbose = verbose

        self.sock_daemon = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.is_connected = False
        self.is_accepted = False

        if self.role == Const.PASSIVE_NAME:
            self.sock_daemon.bind((passive_ip, passive_port))
        else:
            self.sock_daemon.bind((active_ip, active_port))

        self.sock_daemon.listen(10)

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

    def send(self, msg, passive_party=True):
        if self.role == Const.PASSIVE_NAME:
            self._passive_send(msg)
        else:
            if passive_party:
                try:
                    self._active_send(msg)
                    return True
                except Exception:
                    print(colored("Passive party is disconnected", 'red'))
                    return False
            else:
                return False

    def recv(self, passive_party=True):
        if self.role == Const.PASSIVE_NAME:
            return self._passive_recv()
        else:
            if passive_party:
                try:
                    rec_data = self._active_recv()
                except Exception:
                    print(colored("Passive party is disconnected", 'red'))
                    return None, False
                passive_party = self.verify_data_integrtiy(rec_data)
                if not passive_party:
                    print(colored("Passive party has been disconnected", 'red'))
                return rec_data, passive_party
            else:
                return None, False
            # return self._active_recv()


    def close(self):
        self.sock_send.close()
        self.sock_daemon.close()

    def try_to_reconnect(self):
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock_send.connect((self.passive_ip, self.passive_port))
            self.is_connected = True
            return True
        except Exception:
            return False

    def _passive_send(self, msg):
        if not self.is_connected:
            self.sock_send.connect((self.active_ip, self.active_port))
            self.is_connected = True
        try:
            msg_pickled = pickle.dumps(msg)
            # prefix is the binary representation of the length of pickled message
            prefix = self._msg_prefix(len(msg_pickled))
            msg_send = prefix + msg_pickled
            self.sock_send.sendall(msg_send)
            if self.verbose:
                print('[SOCKET-PASSIVE]: Send message to active party.')
        except pickle.PickleError:
            raise pickle.PickleError(
                "Can't pickle object of type {}".format(type(msg)))

    def _active_send(self, msg):
        if not self.is_connected:
            self.sock_send.connect((self.passive_ip, self.passive_port))
            self.is_connected = True
        try:
            msg_pickled = pickle.dumps(msg)
            prefix = self._msg_prefix(len(msg_pickled))
            msg_send = prefix + msg_pickled
            self.sock_send.sendall(msg_send)
            if self.verbose:
                print('[SOCKET-ACTIVE]: Send message to passive party.')
        except pickle.PickleError:
            raise pickle.PickleError(
                "Can't pickle object of type {}".format(type(msg)))

    def verify_data_integrtiy(self, data):
        try:
            data[0]
            return True
        except Exception:
            return False

    def close_send(self):
        self.sock_send.close()

    def _passive_recv(self):
        if not self.is_accepted:
            self.conn, addr = self.sock_daemon.accept()
            self.is_accepted = True
        # first 4 bytes means length of msg
        raw_msglen = self._recvall(self.conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]  # unpack always returns a tuple
        raw_data = self._recvall(self.conn, msglen)
        if self.verbose:
            print('[SOCKET-PASSIVE]: Receive meesage from active party.')

        return pickle.loads(raw_data)

    def _active_recv(self):
        if not self.is_accepted:
            self.conn, addr = self.sock_daemon.accept()
            self.is_accepted = True
        raw_msglen = self._recvall(self.conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        raw_data = self._recvall(self.conn, msglen)
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


class Socket_disconnection(BaseMessenger):
    """Using python socket to implement messenger."""

    def __init__(self,
                 role,
                 active_ip,
                 active_port,
                 passive_ip,
                 passive_port,
                 interval=400,
                 sleep_time=0.5,
                 verbose=False):
        """Initialize socket messenger.

        After initialization, a dameon socket will run in backend waiting for
        new connections.

        Args:
            role: federated learning role, only "passive_party" and "active_party" are valid.
            interval: Wnen #interval sockets are created, the OS will sleep
                for one second, which is to wait for old sockets to be released.
                Otherwise, there will be no available ports to create new socket
                because it's been run out. Default 800.
            verbose: Whether to print communication status, default False.
        """
        socket.setdefaulttimeout(20)
        super(Socket_disconnection, self).__init__()
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        self.role = role
        self.active_ip = active_ip
        self.active_port = active_port
        self.passive_ip = passive_ip
        self.passive_port = passive_port
        self.interval = interval
        self.sleep_time = sleep_time
        self.verbose = verbose

        self.count = 1

        self.sock_daemon = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.role == Const.PASSIVE_NAME:
            self.sock_daemon.bind((passive_ip, passive_port))
        else:
            self.sock_daemon.bind((active_ip, active_port))
        self.sock_daemon.listen(10)

    @classmethod
    def from_config(cls, config):
        assert isinstance(config, BaseConfig), 'config object should be an ' \
                                               'instance of BaseConfig class.'
        return cls(role=config.ROLE,
                   active_ip=config.ACTIVE_IP,
                   active_port=config.ACTIVE_PORT,
                   passive_ip=config.PASSIVE_IP,
                   passive_port=config.PASSIVE_PORT,
                   interval=config.INTERVAL,
                   sleep_time=config.SLEEP_TIME,
                   verbose=config.VERBOSE)

    def send(self, msg):
        if self.count % self.interval == 0:
            time.sleep(self.sleep_time)  # sleep to wait for old sockets to be released

        if self.role == Const.PASSIVE_NAME:
            self._passive_send(msg)
        else:
            self._active_send(msg)

        self.count += 1

    def recv(self):
        if self.role == Const.PASSIVE_NAME:
            return self._passive_recv()
        else:
            return self._active_recv()

    def close(self):
        self.sock_daemon.close()

    def _passive_send(self, msg):
        """Create a new socket to send message.

        Every time when new message need to be send, a new socket will be
        created, so this solution is not that efficient.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.active_ip, self.active_port))
        try:
            s.send(pickle.dumps(msg))
            if self.verbose:
                print('[SOCKET-PASSIVE]: Send message to active party.')
            s.close()
        except pickle.PickleError:
            raise pickle.PickleError("Can't pickle object of type {}".format(type(msg)))

    def _active_send(self, msg):
        """Create a new socket to send message."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.passive_ip, self.passive_port))
        try:
            s.send(pickle.dumps(msg))
            if self.verbose:
                print('[SOCKET-ACTIVE]: Send message to passive party.')
            s.close()
        except pickle.PickleError:
            raise pickle.PickleError("Can't pickle object of type {}".format(type(msg)))

    def _passive_recv(self):
        conn, addr = self.sock_daemon.accept()  # blocking
        msg = self._recv_bytes_stream(conn)
        if self.verbose:
            print('[SOCKET-PASSIVE]: Receive meesage from active party.')
        return msg

    def _active_recv(self):
        conn, addr = self.sock_daemon.accept()  # blocking
        msg = self._recv_bytes_stream(conn)
        if self.verbose:
            print('[SOCKET-ACTIVE]: Receive message from passive party.')
        return msg

    def _recv_bytes_stream(self, conn):
        """Receive data from stream until there's no more packet."""
        packets = []
        while True:
            packet = conn.recv(1024)
            if not packet: break
            packets.append(packet)

        return pickle.loads(b''.join(packets))
