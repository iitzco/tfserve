"""
JSON encode / decode support.
"""

import json

from tfserve.tfserve import BadInput
from tfserve.handler import EncodeDecodeHandler

class JSONHandler(EncodeDecodeHandler):
    """
    JSON handler for encode and decode.
    """

    def __init__(self, inputs, outputs, batch=False):
        self.input_tensors = inputs
        self.output_tensors = outputs
        self.batch_mode = batch

    def get_description(self):
        return "JSON handler"

    def encode(self, request_bytes):
        """
        Encode request bytes as model inputs.

        `request_bytes` is a byte array that can be decoded as
        JSON. The decoded value must be a Python dict that contains
        values for each input tensor.

        """
        inputs = self._decode_request(request_bytes)
        self._validate_inputs(inputs)
        return inputs

    @staticmethod
    def _decode_request(request_bytes):
        if not request_bytes:
            raise BadInput("empty request")
        try:
            request_str = request_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            raise BadInput(str(e))
        else:
            try:
                return json.loads(request_str)
            except json.decoder.JSONDecodeError as e:
                raise BadInput(str(e))

    def _validate_inputs(self, inputs):
        if not isinstance(inputs, dict):
            raise BadInput("inputs must be a JSON object")
        missing_inputs = [
            name for name in self.input_tensors
            if name not in inputs]
        if missing_inputs:
            raise BadInput(
                "missing inputs: %s"
                % ','.join(missing_inputs))

    @staticmethod
    def decode(outputs):
        """
        Decode model outputs to a JSON serializable Python dict.

        """
        return {
            name: outputs[name].tolist()
            for name in outputs
        }

def create_handler(**kw):
    """
    Create a JSONHandler instance.
    """
    return JSONHandler(**kw)
