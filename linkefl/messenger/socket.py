import socket
import pickle
import struct
import time
from abc import ABC, abstractmethod


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


class FastSocketMessenger(Messenger):
    """Implement messenger using python socket.

    Alice and Bob will only need to maintain two pair sockets, one for Alice
    sending and bob receiving, the other for Alice receiving and Bob sending.
    It is much more efficient than `SocketMessenger`.
    """
    def __init__(self, role, config, verbose=False):
        """Initialize socket messenger.

        After Initialzation, a daemon socket will run in backend at both Alice
        and Bob's side.
        """
        super(FastSocketMessenger, self).__init__()
        assert role in ('alice', 'bob'), 'invalid role'
        self.role = role
        self.config = config
        self.verbose = verbose
        self.sock_daemon = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.accepted = False

        if self.role == 'alice':
            self.sock_daemon.bind((config.ALICE_HOST, config.ALICE_PORT))
        else:
            self.sock_daemon.bind((config.BOB_HOST, config.BOB_PORT))
        self.sock_daemon.listen(10)

    def send(self, msg):
        if self.role == 'alice':
            self._alice_send(msg)
        else:
            self._bob_send(msg)

    def recv(self):
        if self.role == 'alice':
            return self._alice_recv()
        else:
            return self._bob_recv()

    def close(self):
        self.sock_send.close()
        self.sock_daemon.close()

    def _alice_send(self, msg):
        if not self.connected:
            self.sock_send.connect((self.config.BOB_HOST, self.config.BOB_PORT))
            self.connected = True
        try:
            msg_pickled = pickle.dumps(msg)
            # prefix is the binary representation of the length of pickled message
            prefix = self._msg_prefix(len(msg_pickled))
            msg_send = prefix + msg_pickled
            self.sock_send.sendall(msg_send)
            if self.verbose:
                print('[SOCKET-ALICE]: Send message to Bob.')
        except pickle.PickleError:
            raise pickle.PickleError(
                "Can't pickle object of type {}".format(type(msg)))

    def _bob_send(self, msg):
        if not self.connected:
            self.sock_send.connect((self.config.ALICE_HOST, self.config.ALICE_PORT))
            self.connected = True
        try:
            msg_pickled = pickle.dumps(msg)
            prefix = self._msg_prefix(len(msg_pickled))
            msg_send = prefix + msg_pickled
            self.sock_send.sendall(msg_send)
            if self.verbose:
                print('[SOCKET-BOB]: Send message to Alice.')
        except pickle.PickleError:
            raise pickle.PickleError(
                "Can't pickle object of type {}".format(type(msg)))

    def _alice_recv(self):
        if not self.accepted:
            self.conn, addr = self.sock_daemon.accept()
            self.accepted = True
        # first 4 bytes means length of msg
        raw_msglen = self._recvall(self.conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0] # unpack always returns a tuple
        raw_data = self._recvall(self.conn, msglen)
        if self.verbose:
            print('[SOCKET-ALICE]: Receive meesage from Bob.')

        return pickle.loads(raw_data)

    def _bob_recv(self):
        if not self.accepted:
            self.conn, addr = self.sock_daemon.accept()
            self.accepted = True
        raw_msglen = self._recvall(self.conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        raw_data = self._recvall(self.conn, msglen)
        if self.verbose:
            print('[SOCKET-BOB]: Receive message from Alice.')

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


class SocketMessenger(Messenger):
    """Using python socket to implement messenger."""
    def __init__(self, role, config, verbose=False):
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
        super(SocketMessenger, self).__init__()
        assert role in ('alice', 'bob'), 'invalid role'
        self.role = role
        self.config = config
        self.verbose = verbose
        self.count = 1
        self.sock_daemon = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        if self.role == 'alice':
            self.sock_daemon.bind((config.ALICE_HOST, config.ALICE_PORT))
        else:
            self.sock_daemon.bind((config.BOB_HOST, config.BOB_PORT))

        self.sock_daemon.listen(10)

    def send(self, msg):
        if self.count % self.config.SOCK_WAIT_INTERVAL == 0:
            time.sleep(1) # sleep for one second so that old sockets can be released

        if self.role == 'alice':
            self._alice_send(msg)
        else:
            self._bob_send(msg)

        self.count += 1

    def recv(self):
        if self.role == 'alice':
            return self._alice_recv()
        else:
            return self._bob_recv()

    def close(self):
        self.sock_daemon.close()

    def _alice_send(self, msg):
        """Create a new socket to send message.

        Every time when new message need to be send, a new socket will be
        created, so this solution is not that efficient.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.config.BOB_HOST, self.config.BOB_PORT))
        try:
            s.send(pickle.dumps(msg))
            if self.verbose:
                print('[SOCKET-ALICE]: Send message to Bob.')
            s.close()
        except pickle.PickleError:
            raise pickle.PickleError("Can't pickle object of type {}".format(type(msg)))

    def _bob_send(self, msg):
        """Create a new socket to send message."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.config.ALICE_HOST, self.config.ALICE_PORT))
        try:
            s.send(pickle.dumps(msg))
            if self.verbose:
                print('[SOCKET-BOB]: Send message to Alice.')
            s.close()
        except pickle.PickleError:
            raise pickle.PickleError("Can't pickle object of type {}".format(type(msg)))

    def _alice_recv(self):
        conn, addr = self.sock_daemon.accept() # blocking
        msg = self._recv_bytes_stream(conn)
        if self.verbose:
            print('[SOCKET-ALICE]: Receive meesage from Bob.')
        return msg

    def _bob_recv(self):
        conn, addr = self.sock_daemon.accept() # blocking
        msg = self._recv_bytes_stream(conn)
        if self.verbose:
            print('[SOCKET-BOB]: Receive message from Alice.')
        return msg

    def _recv_bytes_stream(self, conn):
        """Receive data from stream until there's no more packet."""
        packets = []
        while True:
            packet = conn.recv(1024)
            if not packet: break
            packets.append(packet)

        return pickle.loads(b''.join(packets))
