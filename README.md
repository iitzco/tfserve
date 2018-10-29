# TFServe

[![Downloads](https://pepy.tech/badge/tfserve)](https://pepy.tech/project/tfserve)  [![PyPI version](https://badge.fury.io/py/tfserve.svg)](https://badge.fury.io/py/tfserve)

TFServe is a framework designed to serve tensorflow models in a simple and easy way as an HTTP API server. It's built on top of [Werkzeug](http://werkzeug.pocoo.org/).

## How to install

```bash
$ pip install tfserve
```

After installing `tfserve`, install either `tensorflow` of `tensorflow-gpu` (the latter if you have GPU available).

```bash
$ pip install tensorflow
```
or
```bash
$ pip install tensorflow-gpu
```

## How to use

### Python API

You will need 5 parts:

1. **Model**: it can be a `.pb` file or a model directory containing ckpt files.
2. **Input tensor names**: name of the input tensors of the graph.
3. **Output tensor names**: name of the output tensors of the graph.
4. **`encode`**: python function that receives the request body data and outputs a `dict` mapping input tensor names to input numpy values.
5. **`decode`**: python function that receives a `dict` mapping output tensor names to output numpy values and returns the HTTP response.

Follow the example to learn how to combine these parts...

#### Example

Deploy image classification service that receives a binary jpg image and returns the class of the object found in the image alongside it's probability.

```python

# 1. Model: trained mobilenet on ImageNet that can be downloaded from
#           https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
MODEL_PATH = "mobilenet_v2_1.4_224/mobilenet_v2_1.4_224_frozen.pb"

# 2. Input tensor names:
INPUT_TENSORS = ["import/input:0"]

# 3. Output tensor names:
OUTPUT_TENSORS = ["import/MobilenetV2/Predictions/Softmax:0"]

# 4. encode function: Receives raw jpg image as request data. Returns dict
#                     mappint import/input:0 to numpy value.
def encode(request_data):
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jpg") as f:
        f.write(request_data)
        # Model receives 224x224 normalized RGB image.
        img = Image.open(f.name).resize((224, 224)) 
        img = np.asarray(img) / 255.

    return {INPUT_TENSORS[0]: img}

# 5. decode function: Receives `dict` mapping import/MobilenetV2/Predictions/Softmax:0 to
#                     numpy value and builds dict with for json response.
def decode(outputs):
    p = outputs[OUTPUT_TENSORS[0]] # 1001 vector with probabilities for each class.
    index = np.argmax(p)
    return {"class": index_to_class_map[index-1], "prob": float(p[index])}
```

That's it! Now create TFServeApp object and run it!

```python
app = TFServeApp(MODEL_PATH, INPUT_TENSORS, OUTPUT_TENSORS, encode, decode)
app.run('127.0.0.1', 5000)  # Host and port where the server will be running
```

See `client.py` for full example.

#### How to consume server

![img](imgs/screen.gif)

> The server supports only `POST` method to `/` with the input information as part of the request body.

The input will be proccessed in the encode function to produce the feed_dict object that will be passed to the graph. The graph output will be processed in the decode function and the server will return whatever the decode function returns.

### CLI

`tfserve` also provides a CLI program with built-in encode/decode handlers:

```bash
tfserve -m PATH [-i INPUTS] [-o OUTPUTS] [-h HANDLER] [-b] [-H HOST] [-p PORT]

  -m PATH, --model PATH
                        path to pb file or directory containing checkpoint
  -i INPUTS, --inputs INPUTS
                        a comma separated list of input tensors
  -o OUTPUTS, --outputs OUTPUTS
                        a comma separated list of output tensors
  -h HANDLER, --handler HANDLER
                        encode/decode handler (deault is 'json')
  -b, --batch           process multiple inputs (default is to process
                        one input per request)
  -H HOST, --host HOST  host interface to bind to (0.0.0.0)
  -p PORT, --port PORT  port to listen on (5000)
```

#### Example

```bash
$ tfserve -m models/graph.pb -i x:0 -o out:0 -h json -H localhost
```

Run `tfserve` with `models/graph.pb` model that takes as input the tensor with name `x:0` (dimesion: [?,5]) and outputs a tensor named `out:0`. The server will run on http://localhost:5000/ and will receive `POST` requests to `/`.

By using the **json handler**, you can provide the input data as a `json` object in the request body:

```json
{
  "x:0": [1,1,3,4,5]
}
```

You will receive a `json` output object as:

```json
{
  "out:0": 0.48
}
```

#### More information about CLI

Run:

```bash
$ tfserve --help
```

## Help

* **What if I don't know the tensor names?**

> You can use `tfserve.helper.estimate_io_tensors(model_path)` function to get a list of possible input/output tensor names. Also, you can use the CLI by running: `tfserve -m [model_path] --help-model`

* **What if I want to run multiple inferences at the same time?**

> You can use `batch=True` when building tfserve.TFServeApp. You will then need to handle the batch dimension yourself in the `encode` and `decode` function.
> Also, if using the CLI, just add the `--batch` flag.


## Limitation

> It only works with one-to-one models. That is, models that need to run the graph only once to get the inference.
> Other architectures of inference will be supported soon. Help is appreciated!
