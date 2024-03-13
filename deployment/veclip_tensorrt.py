# wget https://docs-assets.developer.apple.com/ml-research/models/veclip/vecapdfn_clip_l14.zip
import os
import re
import requests
import argparse
import time
import numpy as np
import torch
from PIL import Image

import tensorrt as trt
from torch2trt import TRTModule

from transformers import CLIPModel, T5Tokenizer
from veclip_preprocessor import image_preprocess

MODEL_DIR = "/home/xuanlin/project_soledad/ml-veclip/checkpoints/vecapdfn_clip_l14"

parser = argparse.ArgumentParser(description="Export VeClip model to ONNX")
parser.add_argument("--model-dir", type=str, default="/home/xuanlin/project_soledad/ml-veclip/checkpoints/vecapdfn_clip_l14", help="Model name")
parser.add_argument("--encoder-engine", type=str, default="/home/xuanlin/project_soledad/ml-veclip/checkpoints/vecapdfn_clip_l14_image_encoder.engine", help="Output ONNX model name")
args = parser.parse_args()

# load tokenizer and model
# Note: The T5 tokenizer does not enforce a fixed maximum input length. Therefore, during usage, 
# if any warnings related to sequence length exceedance appear, they can generally be ignored.
tokenizer = T5Tokenizer.from_pretrained("t5-base")
print(f"Loading model {MODEL_DIR} ...")
model = CLIPModel.from_pretrained(MODEL_DIR)
text_model, text_projection = model.text_model, model.text_projection
text_model, text_projection = text_model.eval(), text_projection.eval()
logit_scale = model.logit_scale.exp().item()

texts = ["a photo of car", "a photo of two cats"]
text_inputs = tokenizer(texts, return_tensors="pt", padding=True)
tt = time.time()
with torch.no_grad():
    text_outputs = model.text_model(**text_inputs) # text encoder a lot faster on cpu since it's t5...
print("text encoder time", time.time() - tt)

crop_size = 224

# vision model
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
pixel_values = image_preprocess([np.asarray(image)], crop_size=crop_size).to("cuda")

with trt.Logger() as logger, trt.Runtime(logger) as runtime:
    with open(args.encoder_engine, 'rb') as f:
        engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
encoder_engine = TRTModule(
    engine,
    input_names=["input_image"],
    output_names=["last_hidden_state", "pooled_output", "norm_pooled_output"]
)
torch.cuda.synchronize()
tt = time.time()
vision_outputs = encoder_engine(pixel_values)
torch.cuda.synchronize()
print("visual encoder time", time.time() - tt)

tt = time.time()
with torch.no_grad():
    text_embeds = text_projection(text_outputs[1])
    text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
    print("text postprocess time", time.time() - tt)
    visual_embeds = vision_outputs[2].cpu()
    torch.cuda.synchronize()
    print("visual cpu postprocess time", time.time() - tt)
    probs = (logit_scale * visual_embeds @ text_embeds.T).softmax(dim=-1)
    print("visual dot product postprocess time", time.time() - tt)
    probs = probs.numpy()
    print("total postprocess time", time.time() - tt)
print(probs)

for prob, text in zip(probs[0], texts):
    # Format and print the message
    print("Probability for '{}' is {:.2%}".format(text, prob))