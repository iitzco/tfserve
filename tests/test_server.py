"""Tests TFServeApp server functionality.
"""

import io
import json
import os
import sys

import numpy as np
from werkzeug.exceptions import MethodNotAllowed

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tfserve

class RequestProxy():
    """Proxy for a Werkzeug request object."""

    def __init__(self, req_data=None, method='POST'):
        if req_data:
            req_encoded = json.dumps(req_data).encode()
        else:
            req_encoded = b''
        self.stream = io.BytesIO(req_encoded)
        self.method = method

class TestRun():
    """Tests server run."""

    model_path = './tests/models/graph.pb'
    in_t = 'import/x:0'
    out_t = 'import/out:0'

    examples = [
        ([1.0, 1.0, 1.0, 1.0, 1.0], 0.2677996287397143),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 0.339625029168856),
    ]

    bad_input = [
        [1.0, 2.0, 3.0, 4.0],
        1234,
        'foo',
    ]

    @classmethod
    def setup_class(cls):
        """Setup for tests by creating a server.

        """
        cls.server = tfserve.TFServeApp(
            cls.model_path,
            [cls.in_t],
            [cls.out_t],
            cls._encode,
            cls._decode,
            False)

    @classmethod
    def _encode(cls, req_bytes):
        return {
            cls.in_t: json.loads(req_bytes)
        }

    @classmethod
    def _decode(cls, outputs):
        return np.float_(outputs[cls.out_t])

    def test_examples(self):
        """Test inference on examples.

        Fails if outputs don't match expected results.

        """
        for example_in, example_out in self.examples:
            req = RequestProxy(example_in)
            resp = self.server._handle_inference(req)
            assert resp.status == '200 OK'
            assert resp.headers.get('Content-Type') == 'application/json'
            decoded = json.loads(b''.join(resp.response))
            assert decoded == example_out

    def test_bad_requests(self):
        """Test unsupported HTTP methods.

        Anything but 'POST' should raise MethodNotAllowed.

        """
        invalid_methods = ('GET', 'PUT', 'HEAD', 'FOOBAR')
        for method in invalid_methods:
            req = RequestProxy(method=method)
            with pytest.raises(MethodNotAllowed):
                self.server._handle_inference(req)

    def test_bad_input(self):
        """Test various invalid inputs.

        Bad values are passed through to TensorFlow, which in turn
        raises ValueError.

        """
        for example_in in self.bad_input:
            req = RequestProxy(example_in)
            with pytest.raises(ValueError):
                self.server._handle_inference(req)
