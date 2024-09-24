import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image
from utils.controlnet_union import ControlNetModel_Union
from utils.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

device = "cpu"

config_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="config_promax.json",
)

config = ControlNetModel_Union.load_config(config_file)
controlnet_model = ControlNetModel_Union.from_config(config)
model_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="diffusion_pytorch_model_promax.safetensors",
)
state_dict = load_state_dict(model_file)
model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
)
model.to(device=device, dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to(device)

pipe = StableDiffusionXLFillPipeline.from_pretrained(
    "./model_weights/photopediaXL_45",
    torch_dtype=torch.float16,
    vae=vae,
    controlnet=model,
    variant="fp16",
).to(device)

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

prompt = "high quality"
(
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
) = pipe.encode_prompt(prompt, device, True)


def fill_image(source, mask, output_path):
    alpha_channel = mask.split()[3]
    inverted_binary_mask = alpha_channel.point(lambda p: p == 0 and 255 or 0)
    source.paste(0, (0, 0), inverted_binary_mask)
    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=source,
    ):
        yield source
    image = image.convert("RGBA")
    source.paste(image, (0, 0), inverted_binary_mask)
    source.save(output_path)
    yield source

if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(description="sample argument parser")
    parser.add_argument(
        "--input_image", type=str, required=True,
        help="Image",
    )
    parser.add_argument(
        "--input_mask", type=str, required=True,
        help="Mask",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output",
    )
    args=parser.parse_args()
    
    source, mask = Image.open(args.input_image), Image.open(args.input_mask)
    fill_image(source, mask, args.output_path)