import tensorflow as tf
import numpy as np
import os
from apistar import http, App, Route

from loader import load_model


class TFServeApp():
    def __init__(self, model_path, in_t, out_t, encode, decode, batch=False):
        self.sess = load_model(model_path)
        self.graph = self.sess.graph

        self.check_tensors(in_t)
        self.check_tensors(out_t)
        self.check_placeholders(in_t)

        self.in_t = in_t
        self.out_t = out_t

        self.encode = encode
        self.decode = decode

        self.batch = batch

    def run(self, *args, **kwargs):
        routes = [Route('/', method='POST', handler=self.make_inference)]

        app = App(routes=routes)
        app.serve(*args, **kwargs)

    def make_inference(self, request: http.Request):
        feed_dict = self.encode(request.body)
        if not self.batch:
            feed_dict = {k: np.expand_dims(v, axis=0) for k, v in feed_dict.items()}
        out_map = {}
        ret = self.sess.run(self.out_t, feed_dict=feed_dict)
        for i, e in enumerate(self.out_t):
            out_map[e] = ret[i] if self.batch else np.squeeze(ret[i])

        return self.decode(out_map)

    def check_placeholders(self, tensors):
        ph = [op for op in self.graph.get_operations() if op.type == "Placeholder"]
        ph_names = [p.outputs[0].name for p in ph]

        for t in tensors:
            if t not in ph_names:
                raise ValueError("Input node must be placeholder: {}".format(t))

    def check_tensors(self, tensors):
        for t in tensors:
            try:
                self.graph.get_tensor_by_name(t)
            except ValueError:
                raise ValueError("Non existent tensor in graph: {}".format(t))
            except KeyError:
                raise ValueError("Non existent tensor in graph: {}".format(t))

    def check_input(provided, required):
        for x in provided:
            if x not in required:
                raise ValueError("Encoded tensor that is not required: {}".format(x))

        for x in required:
            if x not in provided:
                raise ValueError("Input tensor missing: {}".format(x))

