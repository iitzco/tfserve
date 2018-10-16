import os
import sys

def _check_tensorflow():
    """Check tensorflow status.

    Premptively import tensorflow and exit with an error message if
    it's not installed.

    """
    try:
        import tensorflow
    except ImportError:
        sys.stderr.write(
            "tfserve: tensorflow is not installed\n"
            "Try installing it by running 'pip install tensorflow' or\n"
            "'pip install tensorflow-gpu' if your system supports GPUs.\n"
            "For more information see https://www.tensorflow.org/install/\n")
        sys.exit(1)

_check_tensorflow()

from tfserve.tfserve import TFServeApp
from tfserve.tfserve import BadInput
