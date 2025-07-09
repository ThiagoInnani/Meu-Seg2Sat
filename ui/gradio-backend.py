from PIL import Image
from io import BytesIO
import base64

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

import gradio as gr

#controlnet = ControlNetModel.from_pretrained("rgres/sd-controlnet-aerialdreams", torch_dtype=torch.float16)

caminho_do_modelo = "../finetuning/controlnet_training/model_pinus_finetuned"
controlnet = ControlNetModel.from_pretrained(caminho_do_modelo, torch_dtype=torch.float16)


pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=torch.float16
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

# CPU offloading
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()

def generate_map(image, prompt, steps, seed):
    #image = Image.open(BytesIO(base64.b64decode(image_base64)))
    generator = torch.manual_seed(seed)

    #image = Image.fromarray(image)

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        image=image
    ).images[0]

    return image

demo = gr.Interface(
  fn=generate_map,
  inputs=[gr.Image(type="pil", tool='color-sketch'), "text", gr.Slider(0,100), "number"],
  outputs="image")

demo.launch()
