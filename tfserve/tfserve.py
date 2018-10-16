import json

import numpy as np
from apistar import http, App, Route

from werkzeug import routing
from werkzeug import serving
from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import MethodNotAllowed
from werkzeug.wrappers import Request, Response

from tfserve.loader import load_model
import tfserve.graph_utils as graph_utils


class BadInput(Exception):
    """
    Raised by decode when input is invalid.
    """

    def __init__(self, description):
        super(BadInput, self).__init__(description)
        self.description = description


class TFServeApp():
    """
    This class will handle all server functionality for one-to-one models.
    A TFServeApp object is able to run a web server to serve the model and
    will also be responsible for encoding and decoding it's input/output.

    Each TFServeApp will be running it's own tensorflow session and graph.

    The user only needs to create the objects giving the appropiate parameters
    and execute the `run` method that loads everything up. From that moment,
    the server will be up and listening for input data.


    Examples of user use:

    app = TFServeApp("logistic_regression_model.pb", ["input:0"], ["sigmoid:0"]
                        lambda x: {"input:0": np.array(x)},
                        lambda y: {"prob": double(y["sigmoid:0"])})
    app.run("127.0.0.1", 5000)

    A web server should be up and running, being POST to / it's only supported method.
    In this request, the model input should be provided as the request body data. This data
    will be received by the encode function that will process it to feed the model. The server will
    run the model, giving it's output to the decode function that will prepare the reponse data.
    """

    def __init__(self, model_path, in_t, out_t, encode, decode, batch=False):
        """
        When constructing, the method checks that all in_t tensors are valid
        placeholders and all out_t tensors are valid tensors that exist in
        the graph.

        The objects is constructed by providing:
            * A model path
            * List of input placeholder that will be used to feed the model.
            * List of expected outputs tensors (the desired predictions).
            * encode function mapping request data to input numpy values.
            * decode function mapping output numpy values to request response data.

        :param str model_path: It can be a `.pb` file or directory containing checkpoint files.
        :param list[str] in_t: List of placeholder tensor names. Something like: ["input/image:0"]
        :param list[str] out_t: List of output tensor names. Something like: ["output/Softmax:0"]
        :param encode: python function that receives the request body data and returns a `dict` mapping
                        in_t to numpy values.
        :param decode: python function that receives a `dict` mapping out_t to numpy values and returns
                        the response data (for example, a `dict` object that will be transformed to JSON).
                        The return object of this method will be the response to the request.
                        Read it's docs for more information on how to return certain objects (for example, images).
        :param list[str] out_t: List of output tensor names. Something like: ["output/Softmax:0"]
        :param boolean batch: If False, batch dimension (required by tensorflow) will be automatically
                               handled (that is, you don't need to handle it yourself in the encode/decode functions).
                               This option is ideal when dealing with single inferences.
                               If True, you can run multiple inferences at the same time by dealing
                               with the batch dimension yourself in the encode/decode functions.

        :raises ValueError: if in_t are not all placeholder or out_t contains non-existent graph tensors
        """
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

    def _make_inference(self, request: http.Request):
        """
        This method is the request handler. It deals with the logic of encoding the input, running the model
        and decoding it's output for final response.

        When encoding the input, it also makes sure that all the encode function creates numpy values for all
        in_t placeholders.

        :raises ValueError: if the encoded data provided by the request does not include all
                            the in_t placeholders.
        """
        return self._make_inference_impl(request.body)

    def _make_inference_impl(self, req_bytes):
        """
        Implementation of _make_inference to decouple data interface from HTTP server interface.

        req_bytes must be the unencoded HTTP request body as bytes.

        Returns a Python dict that can be encoded as a JSON HTTP response.
        """
        feed_dict = self.encode(req_bytes)
        if not self.batch:
            feed_dict = {k: np.expand_dims(v, axis=0) for k, v in feed_dict.items()}

        feed_dict = {graph_utils.smart_tensor_name(k): v for k, v in feed_dict.items()}


        graph_utils.check_input(feed_dict.keys(), self.in_t, "Encode function must generate all and only input tensors")

        out_map = {}
        ret = self.sess.run(self.out_t, feed_dict=feed_dict)
        for i, e in enumerate(self.out_t):
            out_map[e] = ret[i] if self.batch else np.squeeze(ret[i])

        return self.decode(out_map)

    def run(self, host, port, middleware=None):
        """Werkzeug run implementation.

        `middleware` may be provided as a function to handle
        requests. It must accept the arguments `(handler, req)` where
        `handler` is the TFServeApp request handler and `req` is the
        request.

        """
        app = self._init_app(middleware)
        server = serving.make_server(
            host, port, app, threaded=True,
            request_handler=serving.WSGIRequestHandler)
        server.serve_forever()

    def _init_app(self, middleware=None):
        """Initialize a WSGI application for handling POST to '/'.

        `middleware` may be provided as WSGI middleware.

        """
        routes = routing.Map([
            routing.Rule('/', endpoint=self._handle_inference),
            routing.Rule('/ping', endpoint=self._handle_ping),
            routing.Rule('/shutdown', endpoint=self._handle_shutdown),
        ])
        def app(env, start_resp):
            """WSGI application to handle server requests.

            """
            urls = routes.bind_to_environ(env)
            try:
                handler, _kw = urls.match()
                req = Request(env)
                if middleware:
                    return middleware(handler, req)(env, start_resp)
                return handler(req)(env, start_resp)
            except HTTPException as e:
                return e(env, start_resp)
        return app

    def _handle_inference(self, req):
        """Handle inference request.

        """
        if req.method != 'POST':
            raise MethodNotAllowed(valid_methods=['POST'])
        req_bytes = req.stream.read()
        try:
            resp_val = self._make_inference_impl(req_bytes)
        except BadInput as e:
            raise BadRequest(e.description)
        return Response(json.dumps(resp_val), content_type='application/json')

    @staticmethod
    def _handle_ping(_req):
        """Handles ping request.

        """
        return Response()

    @staticmethod
    def _handle_shutdown(req):
        """Handles shutown request.

        """
        if req.method != 'POST':
            raise MethodNotAllowed(valid_methods=['POST'])
        shutdown = req.environ.get('werkzeug.server.shutdown')
        if not shutdown:
            raise BadRequest("server does not support shutdown")
        shutdown()
        return Response()

    def run_apistar(self, *args, **kwargs):
        """
        LEGACY CODE THAT RUNS UNDER APISTAR

        After building the object, this method needs to be called to run the server. This method
        will build the apistar `app` with the appropiate handlers and will serve it.

        It receives the same parameters as apistar App.serve method. More information here:
        https://github.com/encode/apistar/blob/master/apistar/server/app.py#L161

        :raises ValueError: if the encoded data provided by the request does not include all
                            the in_t placeholders.
        """
        routes = [Route('/', method='POST', handler=self._make_inference)]

        app = App(routes=routes)
        app.serve(*args, **kwargs)

