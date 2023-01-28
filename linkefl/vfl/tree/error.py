class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class DisconnectedError(Error):
    """Exception raised for errors in communication.

    Attributes:
        drop_party_id -- disconnected party id
        message -- explanation of the error
    """

    def __init__(self, disconnect_phase: str = "", disconnect_party_id: int = -1):
        self.disconnect_phase = disconnect_phase
        self.disconnect_party_id = disconnect_party_id

    def __str__(self):
        return (
            f"Disconnect Error Message: \ndisconnect_phase: {self.disconnect_phase},"
            f" disconnect_party: passive_party_{self.disconnect_party_id}."
        )
