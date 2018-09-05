import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tfserve import graph_utils as utils
from tfserve import loader


class TestUtils():
    sess = loader.load_model("./tests/models/graph.pb")
    g = sess.graph

    def test_none(self):
        with pytest.raises(ValueError):
            utils.check_placeholders(None, [])

        with pytest.raises(ValueError):
            utils.check_placeholders(TestUtils.g, None)

        with pytest.raises(ValueError):
            utils.check_tensors(None, [])

        with pytest.raises(ValueError):
            utils.check_tensors(TestUtils.g, None)

        with pytest.raises(ValueError):
            utils.check_input(None, [], "")

        with pytest.raises(ValueError):
            utils.check_input([], None, "")

    def test_correct_tensors(self):
        utils.check_placeholders(TestUtils.g, ["import/x:0"])
        utils.check_tensors(TestUtils.g, ["import/x:0"])
        utils.check_tensors(TestUtils.g, ["import/out:0"])

    def test_incorrect_tensors(self):
        with pytest.raises(ValueError):
            utils.check_placeholders(TestUtils.g, ["pcodmsocs:0"])

        with pytest.raises(ValueError):
            utils.check_tensors(TestUtils.g, ["pcodmsocs:0"])

        with pytest.raises(ValueError):
            utils.check_tensors(TestUtils.g, ["import/x:0", "pcodmsocs:0"])

    def test_smart_tensor(self):
        assert utils.smart_tensor_name("carlitos") == "carlitos:0"
        assert utils.smart_tensor_name("carlitos:0") == "carlitos:0"
        assert utils.smart_tensor_name("carlitos:1") == "carlitos:1"
        assert utils.smart_tensor_name("carlitos:tevez") == "carlitos:tevez:0"
