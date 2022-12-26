from abc import ABC, abstractmethod


class BaseMessenger(ABC):
    """Base class of messenger.

    Messenger provides sending and receiving APIs for communication.
    """
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
