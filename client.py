from PIL import Image
from labels import label_map
import numpy as np
import tempfile

from tfserve import TFServeApp


# 1. Model: trained mobilenet on ImageNet that can be downloaded from
#           https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
MODEL_PATH = "./mobilenet_v2_1.4_224/mobilenet_v2_1.4_224_frozen.pb"


# 2. Input tensor names:
INPUT_TENSORS = ["import/input:0"]

# 3. Output tensor names:
OUTPUT_TENSORS = ["import/MobilenetV2/Predictions/Softmax:0"]


# 4. encode function: Receives raw jpg image as request_data. Returns dict
#                     mappint import/input:0 to numpy value.
#                     Model expects 224x224 normalized RGB image.
#                     That is, [224, 224, 3]-size numpy array.
def encode(request_data):
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jpg") as f:
        f.write(request_data)
        img = Image.open(f.name).resize((224, 224))
        img = np.asarray(img) / 255.

    return {INPUT_TENSORS[0]: img}


# 5. decode function: Receives `dict` mapping import/MobilenetV2/Predictions/Softmax:0 to
#                     numpy value and builds dict with for json response.
def decode(outputs):
    p = outputs[OUTPUT_TENSORS[0]]
    # p will be a 1001 numpy array with all class probabilities.
    index = np.argmax(p)
    # This `dict` will result in a JSON response (courtesy of apistar).
    return {"class": label_map[index-1], "prob": float(p[index])}


# Run the server
app = TFServeApp(MODEL_PATH, INPUT_TENSORS, OUTPUT_TENSORS, encode, decode)
app.run('127.0.0.1', 5000, debug=True)
