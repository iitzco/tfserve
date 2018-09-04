import numpy as np
from apistar import http, App, Route

from tfserve.loader import load_model
import tfserve.graph_utils as graph_utils


class TFServeApp():

    def __init__(self, model_path, in_t, out_t, encode, decode, batch=False):
        self.sess = load_model(model_path)
        self.graph = self.sess.graph

        graph_utils.check_tensors(self.graph, in_t)
        graph_utils.check_tensors(self.graph, out_t)

        graph_utils.check_placeholders(self.graph, in_t)

        self.in_t = in_t
        self.out_t = out_t

        self.encode = encode
        self.decode = decode

        self.batch = batch

    def run(self, *args, **kwargs):
        routes = [Route('/', method='POST', handler=self._make_inference)]

        app = App(routes=routes)
        app.serve(*args, **kwargs)

    def _make_inference(self, request: http.Request):
        feed_dict = self.encode(request.body)
        if not self.batch:
            feed_dict = {k: np.expand_dims(v, axis=0) for k, v in feed_dict.items()}

        feed_dict = {graph_utils.smart_tensor_name(k): v for k, v in feed_dict.items()}

        graph_utils.check_input(feed_dict.keys(), self.in_t, "Encode function must generate all and only input tensors")

        out_map = {}
        ret = self.sess.run(self.out_t, feed_dict=feed_dict)
        for i, e in enumerate(self.out_t):
            out_map[e] = ret[i] if self.batch else np.squeeze(ret[i])

        return self.decode(out_map)

