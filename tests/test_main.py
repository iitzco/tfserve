"""Test module for tflearn.main.

"""

import errno
import json
import os
import random
import socket
import sys
import threading
import time

from urllib import request
from urllib.error import HTTPError
from urllib.error import URLError

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tfserve import main

class Args():
    """Proxy for main args."""

    def __init__(self, **kw):
        for name in kw:
            setattr(self, name, kw[name])

class Server(threading.Thread):
    """Thread to run server used by tests."""

    def __init__(self, args):
        super(Server, self).__init__()
        self.args = args

    def run(self):
        main._serve_model(self.args)

    def wait_for_ready(self, timeout=15):
        """Wait for the server to accept connections.

        Uses '/_ping' path to check server status.
        """
        timeout_at = time.time() + timeout
        while time.time() < timeout_at:
            try:
                self.post("/_ping")
            except URLError as e:
                if not e.reason.errno == 111: # connection refused
                    raise
                time.sleep(0.5)
            else:
                return
        raise RuntimeError("timeout")

    def stop(self):
        """Stop the server.

        Server is stopped by posting to `/_shutdown`.
        """
        self.post('/_shutdown')

    def post(self, path, data=None):
        """Post to the server at `path`.

        `data` is a JSON serializable value.
        """
        url = self._url(path)
        if data is not None:
            encoded_data = json.dumps(data).encode()
            headers = {'Content-Type': 'application/json'}
        else:
            encoded_data = b''
            headers = {}
        req = request.Request(url, encoded_data, headers=headers)
        return request.urlopen(req)

    def decode_post(self, path, data):
        """Post to server at `path` and decode a JSON response.

        Response must have a content type of 'application/json'.
        """
        resp = self.post(path, data)
        assert resp.headers['Content-Type'] == 'application/json'
        return json.load(resp.fp)

    def _url(self, path):
        return 'http://%s:%i%s' % (self.args.host, self.args.port, path)

class TestMain():
    """Tests the main module by starting a server and running tests against it.
    """

    model_path = './tests/models/graph.pb'
    in_t = 'import/x:0'
    out_t = 'import/out:0'

    examples = [
        ([1.0, 1.0, 1.0, 1.0, 1.0], 0.2677996287397143),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 0.339625029168856),
    ]

    unsupported_examples = [
        ([1.0, 2.0, 3.0, 4.0],
         (b"Cannot feed value of shape (1, 4) for Tensor "
          b"'import/x:0', which has shape '(?, 5)'")),
        (1234,
         (b"Cannot feed value of shape (1,) for Tensor "
          b"'import/x:0', which has shape '(?, 5)'")),
        ('foo', b"could not convert string to float: 'foo'"),
    ]

    @classmethod
    def setup_class(cls):
        """Setup for tests.

        Starts the server on a randomly selected free port.

        """
        port = _free_port()
        args = Args(
            model=cls.model_path,
            inputs=cls.in_t,
            outputs=cls.out_t,
            host='localhost',
            port=port,
            batch=False)
        cls.server = server = Server(args)
        server.start()
        server.wait_for_ready()

    @classmethod
    def teardown_class(cls):
        """Teardown for tests.

        Stops the server by posting to '/_shutdown'. Note that if this
        fails the server thread will continue running and the test run
        will not exit.

        """
        cls.server.stop()

    def test_empty_request(self):
        """Test an empty request.

        We expect 400 responses with meaningful messages.

        """
        with pytest.raises(HTTPError) as e:
            self.server.post('/')
        assert e.value.getcode() == 400
        assert b"empty request" in e.value.file.read()

    def test_non_json_object(self):
        """Test a non-JSON object input.

        We expect 400 responses with meaningful messages.

        """
        with pytest.raises(HTTPError) as e:
            self.server.post('/', data=123)
        assert e.value.getcode() == 400
        assert b"inputs must be a JSON object" in e.value.file.read()

    def test_missing_input(self):
        """Test missing input.

        We expect 400 responses with meaningful messages.

        """
        with pytest.raises(HTTPError) as e:
            self.server.post('/', data={})
        assert e.value.getcode() == 400
        error_msg = b"missing inputs: %b" % self.in_t.encode()
        assert error_msg in e.value.file.read()

    def test_examples(self):
        """Test inference for known examples.

        We expect 200 responses that can be decoded to the expected
        outputs.

        """
        for example, expected_output in self.examples:
            inputs = {
                self.in_t: example
            }
            outputs = self.server.decode_post('/', data=inputs)
            assert outputs[self.out_t] == expected_output

    def test_unsupported_examples(self):
        """Test examples that aren't supported by model.

        We expect 400 responses with meaningful messages.

        """
        for val, expected_error_fragment in self.unsupported_examples:
            inputs = {
                self.in_t: val
            }
            with pytest.raises(HTTPError) as e:
                self.server.post('/', data=inputs)
            assert e.value.getcode() == 400
            assert expected_error_fragment in e.value.file.read()

    def test_non_root_url(self):
        """Test paths other than '/'.

        We expect 404 (not found) responses.

        """
        with pytest.raises(HTTPError) as e:
            self.server.post('/foobar')
        assert e.value.getcode() == 404

def _free_port():
    attempts = 0
    while True:
        if attempts > 100:
            raise RuntimeError("too many free port attempts")
        port = random.randint(49152, 65535)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        try:
            sock.connect(('localhost', port))
        except socket.timeout:
            return port
        except socket.error as e:
            if e.errno == errno.ECONNREFUSED:
                return port
        else:
            sock.close()
        attempts += 1
