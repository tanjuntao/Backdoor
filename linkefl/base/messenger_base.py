from abc import ABC, abstractmethod


class BaseMessenger(ABC):
    """Base class of messenger.

    Messenger provides sending and receiving APIs for communication.
    """

    def __init__(self):
        self.active_ip = None
        self.active_port = None
        self.passive_ip = None
        self.passive_port = None

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
