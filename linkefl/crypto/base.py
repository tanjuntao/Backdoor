from abc import ABC, abstractmethod


class PartialCryptoSystem(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def encrypt(self, plaintext):
        pass

    @abstractmethod
    def encrypt_vector(self, plain_vector,
                       using_pool=False, n_workers=None, pool=None):
        pass


class CryptoSystem(ABC):
    """Base class of cryptosystem"""
    def __init__(self, key_size=1024):
        """Initialize a cryptosystem.

        Args:
            key_size: Key size of cryptosystem, default 1024 bits.
        """
        self.key_size = key_size

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
    def encrypt_vector(self, plain_vector,
                       using_pool=False, n_workers=None, pool=None):
        """Encrypt a vector with more than one plaintext message"""
        pass

    @abstractmethod
    def decrypt_vector(self, cipher_vector,
                       using_pool=False, n_workers=None, pool=None):
        """Decrypt a vector of ciphertext"""
        pass