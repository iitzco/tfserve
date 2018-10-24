"""
Defines handler interface.

Protocol specific handlers should extend this this class.

This interface is used by `tfserve.main` to handle HTTP request POST
bodies.
"""

class EncodeDecodeHandler():
    """
    Abstract class for encode/decode handlers.
    """

    def get_description(self):
        """
        Return a string that describes the handler.
        """
        raise NotImplementedError()

    def encode(self, input_bytes):
        """
        Encode input bytes as model inputs.
        """
        raise NotImplementedError()

    def decode(self, outputs):
        """
        Decode model outputs to JSON serializable Python object.
        """
        raise NotImplementedError()
