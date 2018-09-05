import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tfserve import loader

class TestLoadModel():

    def test_none(self):
        with pytest.raises(ValueError):
            loader.load_model(None)

    def test_inexistent_path(self):
        with pytest.raises(ValueError):
            loader.load_model("./non_existant.pb")

    def test_inexistent_dir(self):
        with pytest.raises(ValueError):
            loader.load_model("./non_existant/")

    def test_dir(self):
        assert loader.load_model("./tests/models/") is not None

    def test_pb(self):
        assert loader.load_model("./tests/models/graph.pb") is not None
