from PIL import Image
from labels import LABEL_MAP
import numpy as np
import tempfile

from tfserve import TFServeApp


MODEL_PATH = "./mobilenet_v2_1.4_224/mobilenet_v2_1.4_224_frozen.pb"
INPUT_TENSORS = ["import/input:0"]
OUTPUT_TENSORS = ["import/MobilenetV2/Predictions/Softmax:0"]


def encode(request_data):
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jpg") as f:
        f.write(request_data)
        img = Image.open(f.name).resize((224, 224))
        img = np.asarray(img) / 255.

    return {INPUT_TENSORS[0]: img}


def decode(outputs):
    p = outputs[OUTPUT_TENSORS[0]]
    index = np.argmax(p)
    return {"class": LABEL_MAP[index-1], "prob": float(p[index])}


app = TFServeApp(MODEL_PATH, INPUT_TENSORS, OUTPUT_TENSORS, encode, decode)
app.run('127.0.0.1', 5000, debug=True)
