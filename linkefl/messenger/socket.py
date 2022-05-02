import pickle
import socket
import struct
import time
from abc import ABC, abstractmethod

from linkefl.config import BaseConfig
from linkefl.common.const import Const


class Messenger(ABC):
    """Base class of messenger.

    Messenger provides sending and receiving APIs for communication.
    """
    def __init__(self):
        pass

    @abstractmethod
    def send(self, msg):
        """Send message."""
        pass

    @abstractmethod
    def recv(self):
        """Receive message."""
        pass

    @abstractmethod
    def close(self):
        """Close connection."""
        pass


class FastSocket(Messenger):
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
        super(FastSocket, self).__init__()
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

    def send(self, msg):
        if self.role == Const.PASSIVE_NAME:
            self._passive_send(msg)
        else:
            self._active_send(msg)

    def recv(self):
        if self.role == Const.PASSIVE_NAME:
            return self._passive_recv()
        else:
            return self._active_recv()

    def close(self):
        self.sock_send.close()
        self.sock_daemon.close()

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

    def _passive_recv(self):
        if not self.is_accepted:
            self.conn, addr = self.sock_daemon.accept()
            self.is_accepted = True
        # first 4 bytes means length of msg
        raw_msglen = self._recvall(self.conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0] # unpack always returns a tuple
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


class Socket(Messenger):
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
            role: federated learning role, only "alice" and "bob" are valid.
            interval: Wnen #interval sockets are created, the OS will sleep
                for one second, which is to wait for old sockets to be released.
                Otherwise, there will be no available ports to create new socket
                because it's been run out. Default 800.
            verbose: Whether to print communication status, default False.
        """
        super(Socket, self).__init__()
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
            time.sleep(self.sleep_time) # sleep to wait for old sockets to be released

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
        conn, addr = self.sock_daemon.accept() # blocking
        msg = self._recv_bytes_stream(conn)
        if self.verbose:
            print('[SOCKET-PASSIVE]: Receive meesage from active party.')
        return msg

    def _active_recv(self):
        conn, addr = self.sock_daemon.accept() # blocking
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
