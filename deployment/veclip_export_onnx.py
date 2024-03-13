"""
/usr/src/tensorrt/bin/trtexec --onnx=/home/xuanlin/project_soledad/ml-veclip/checkpoints/vecapdfn_clip_l14_image_encoder.onnx \
    --minShapes=input_image:1x3x224x224 --optShapes=input_image:16x3x224x224 --maxShapes=input_image:16x3x224x224 \
    --saveEngine=/home/xuanlin/project_soledad/ml-veclip/checkpoints/vecapdfn_clip_l14_image_encoder.engine
"""
import os
import warnings
import argparse
import numpy as np
import torch, torch.nn as nn
from PIL import Image

from transformers import CLIPModel, T5Tokenizer

parser = argparse.ArgumentParser(description="Export VeClip model to ONNX")
parser.add_argument("--model-dir", type=str, default="/home/xuanlin/project_soledad/ml-veclip/checkpoints/vecapdfn_clip_l14", help="Model name")
parser.add_argument("--encoder-output", type=str, default="/home/xuanlin/project_soledad/ml-veclip/checkpoints/vecapdfn_clip_l14_image_encoder.onnx", help="Output ONNX model name")
args = parser.parse_args()

crop_size = 224 if not "336" in args.model_dir else 336


class EncoderOnnxModel(nn.Module):
    def __init__(
        self,
        image_encoder,
        visual_projection
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.visual_projection = visual_projection

    @torch.no_grad()
    def forward(self, input_image):
        hidden_states = self.image_encoder.embeddings(input_image)
        hidden_states = self.image_encoder.pre_layrnorm(hidden_states)

        for idx, encoder_layer in enumerate(self.image_encoder.encoder.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        last_hidden_state = hidden_states
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.image_encoder.post_layernorm(pooled_output)
        norm_pooled_output = self.visual_projection(pooled_output)
        norm_pooled_output = norm_pooled_output / norm_pooled_output.norm(dim=-1, keepdim=True)
        return last_hidden_state, pooled_output, norm_pooled_output


def run_export(
    model,
    encoder_output: str,
    opset: int,
) -> None:
    print("Loading model...")

    onnx_model = EncoderOnnxModel(model.vision_model, model.visual_projection)

    dummy_input = {"input_image": torch.randn((1, 3, crop_size, crop_size), dtype=torch.float).to("cuda")}
    dynamic_axes = {
        "input_image": {0: "batch_size"},
    }

    _ = onnx_model(**dummy_input)

    output_names = ["last_hidden_state", "pooled_output", "norm_pooled_output"]

    if not os.path.exists(os.path.dirname(encoder_output)):
        os.makedirs(os.path.dirname(encoder_output))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting onnx model to {encoder_output}...")
        with open(encoder_output, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_input.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_input.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

# load tokenizer and model
# Note: The T5 tokenizer does not enforce a fixed maximum input length. Therefore, during usage, 
# if any warnings related to sequence length exceedance appear, they can generally be ignored.
tokenizer = T5Tokenizer.from_pretrained("t5-base")
print(f"Loading model {args.model_dir} ...")
model = CLIPModel.from_pretrained(args.model_dir).to("cuda")


run_export(model, args.encoder_output, 17)