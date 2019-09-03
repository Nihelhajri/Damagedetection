from PIL import Image
import labels
import numpy as np
import tempfile

from tfserve import TFServeApp


# 1. Model: trained mobilenet on ImageNet that can be downloaded from
#           https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
MODEL_PATH = "/home/hajrinihel/learning/damage_inference_graph/frozen_inference_graph.pb"


# 2. Input tensor names:
INPUT_TENSORS = ["import/image_tensor:0"]

# 3. Output tensor names:
OUTPUT_TENSORS = ["import/image_tensor:0"]


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
    index = np.argmax(p)
    print(OUTPUT_TENSORS[0])
    print(p[index])
    return {
               "class": "damage",
               "prob": float(index)
           }

# Run the server
app = TFServeApp(MODEL_PATH, INPUT_TENSORS, OUTPUT_TENSORS, encode, decode)
app.run('localhost', 5000)
