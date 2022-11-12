
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class DisconnectedError(Error):
    """Exception raised for errors in communication.

        Attributes:
            drop_party_id -- disconnected party id
            message -- explanation of the error
    """
    def __init__(self, drop_party_id):
        self.drop_party_id = drop_party_id

    def __str__(self):
        return repr(self.drop_party_id)