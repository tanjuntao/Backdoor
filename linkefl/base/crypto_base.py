from abc import ABC, abstractmethod


class BasePartialCryptoSystem(ABC):
    def __init__(self):
        self.pub_key = None
        self.type = None

    @abstractmethod
    def encrypt(self, plaintext):
        pass

    @abstractmethod
    def encrypt_vector(self, plain_vector, *, pool=None, num_workers=1):
        pass


class BaseCryptoSystem(ABC):
    """Base class of cryptosystem"""

    def __init__(self, key_size=1024):
        """Initialize a cryptosystem.

        Args:
            key_size: Key size of cryptosystem, default 1024 bits.
        """
        self.key_size = key_size
        self.type = None

    @abstractmethod
    def _gen_key(self, key_size):
        """Generate public key and private key."""
        pass

    @abstractmethod
    def encrypt(self, plaintext):
        """Encrypt single plaintext message."""
        pass

    @abstractmethod
    def decrypt(self, ciphertext):
        """Decrypt single ciphertext"""
        pass

    @abstractmethod
    def encrypt_vector(self, plain_vector, *, pool=None, num_workers=1):
        """Encrypt a vector with more than one plaintext message"""
        pass

    @abstractmethod
    def decrypt_vector(self, cipher_vector, *, pool=None, num_workers=1):
        """Decrypt a vector of ciphertext"""
        pass
