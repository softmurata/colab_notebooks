import os
import argparse
import torch
from lora_diffusion import tune_lora_scale, patch_pipe
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1-base", help="model type from hugging face")
parser.add_argument("--patch_weight", type=str, default="/content/output/lora_weight.safetensors", help="patch weight which was learned with lora")
parser.add_argument("--unet_scale", type=float, default=1.0)
parser.add_argument("--te_scale", type=float, default=1.0, help="text_encoder scale")
parser.add_argument("--prompt", type=str, default="anya")
parser.add_argument("--infer_steps", type=int, default=50)
parser.add_argument("--guidance_scale", type=int, default=7)
parser.add_argument("--output", type=str, default="/content/output.jpg")
args = parser.parse_args()


# パイプラインの準備
model_id = args.model_id
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    scheduler=EulerDiscreteScheduler.from_pretrained(
        model_id, 
        subfolder="scheduler"
    ),     
    torch_dtype=torch.float16
).to("cuda")

patch_pipe(
    pipe,
    args.patch_weight,
    patch_text=True,
    patch_ti=True,
    patch_unet=True,
)

# LoRAの重みの調整
tune_lora_scale(pipe.unet, args.unet_scale)
tune_lora_scale(pipe.text_encoder, args.te_scale)

# 推論の実行
image = pipe(
    args.prompt, 
    num_inference_steps=args.infer_steps, 
    guidance_scale=args.guidance_scale
).images[0]
image.save("/content/output.jpg")