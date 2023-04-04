import pickle
import socket
import struct
import zlib

import blosc

from linkefl.base import BaseMessenger
from linkefl.common.const import Const


class EasySocketServer:
    def __init__(self, active_ip, active_port, passive_num, verbose=False):
        """Initialize socket messenger.

        After Initialzation, a daemon socket will run in backend
        """
        self.active_ip = active_ip
        self.active_port = active_port

        self.sock_daemon = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_daemon.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_daemon.bind((self.active_ip, self.active_port))
        self.sock_daemon.listen(passive_num)

        self.messengers = []
        print(f"Waiting for {passive_num} passive party to connect...")
        for _ in range(passive_num):
            conn, addr = self.sock_daemon.accept()
            messenger = EasySocket(role=Const.ACTIVE_NAME, conn=conn, verbose=verbose)
            self.messengers.append(messenger)
            print(f"Accept connection from {addr}.")
        print("All connected.")

    def get_messengers(self):
        return self.messengers

    def close(self):
        self.sock_daemon.close()


class EasySocket(BaseMessenger):
    """Implement messenger using python socket"""

    def __init__(self, role, conn, verbose=False):
        """Initialize socket messenger.

        After Initialzation, a daemon socket will run in backend
        """
        super(EasySocket, self).__init__()
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), "Invalid role"
        self.role = role
        self.conn = conn
        self.verbose = verbose

    @classmethod
    def init_passive(cls, active_ip, active_port, verbose=False):
        sock_passive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_passive.connect((active_ip, active_port))
        return cls(role=Const.PASSIVE_NAME, conn=sock_passive, verbose=verbose)

    def send(self, msg, compress=False, cname="blosc"):
        assert cname in (Const.BLOSC, Const.ZLIB), "invalid compression name"
        self._send(msg, compress=compress, cname=cname)

    def recv(self):
        return self._recv()

    def close(self):
        self.conn.close()

    def _send(self, msg, compress=False, cname="blosc"):
        try:
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
            self.conn.sendall(msg_send)
            if self.verbose:
                if self.role == Const.ACTIVE_NAME:
                    other = Const.PASSIVE_NAME
                else:
                    other = Const.ACTIVE_NAME
                print(f"[SOCKET-{self.role}]: Send message to {other} party.")
        except pickle.PickleError:
            raise pickle.PickleError("Can't pickle object of type {}".format(type(msg)))

    def _recv(self):
        msglen, compress, cname = self._recv_prefixes(self.conn)
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
            if self.role == Const.ACTIVE_NAME:
                other = Const.PASSIVE_NAME
            else:
                other = Const.ACTIVE_NAME
            print(f"[SOCKET-{self.role}]: Receive message from {other} party.")

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
        return struct.pack(">I", msg_len)

    def _compress_prefix(self, compress):
        """
        Args:
            compress[bool], whether using data compression algo
            '>?': '>' means Big Endian, '?' means python bool type (1 byte)
        """
        return struct.pack(">?", compress)

    def _cname_prefix(self, cname):
        """
        Args:
            cname[str], compression type
            '>B': 'B" means unsigned char (1 byte)
        """
        value = Const.COMPRESSION_DICT[cname]
        return struct.pack(">B", value)

    def _recv_prefixes(self, conn):
        # first 4 bytes means length of msg
        raw_msglen = self._recvall(conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack(">I", raw_msglen)[0]  # unpack always returns a tuple

        # second 1 byte means whether using data compression algorotihm
        raw_compress = self._recvall(conn, 1)
        if not raw_compress:
            return None
        compress = struct.unpack(">?", raw_compress)[0]

        # third 1 byte means compresssion type
        raw_cname_value = self._recvall(conn, 1)
        if not raw_cname_value:
            return None
        cname_value = struct.unpack(">B", raw_cname_value)[0]
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
