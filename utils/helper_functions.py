from typing import Tuple, Optional, List
import gradio as gr
import supervision as sv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import torch, os, re, ast, random, base64, glob, PIL.Image, cv2, gc, requests, copy, time
import subprocess as sp, gradio as gr, pandas as pd, numpy as np
from diffusers import AutoPipelineForText2Image
from background_removal.bgrem import bgrem_inference



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("thwri/CogFlorence-2.1-Large", trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained("thwri/CogFlorence-2.1-Large", trust_remote_code=True)

neg_prompt = """
(((duplicate))), (bad legs), bra, one leg, extra leg, (bad face), (bad eyes), ((bad hands, bad anatomy, bad feet, missing fingers, cropped:1.0)), worst quality, jpeg artifacts, signature, (((watermark))), (username), blurry, ugly, old, wide face, ((fused fingers)), ((too many fingers)), amateur drawing, odd, fat, out of frame, (cloned face:1.3), (mutilated:1.3), (deformed:1.3), (gross proportions:1.3), (disfigured:1.3), (mutated hands:1.3), (bad hands:1.3), (extra fingers:1.3), long neck, extra limbs, broken limb, asymmetrical eyes, bad feet, necklace, drawn, illustrated
"""
pipe = AutoPipelineForText2Image.from_pretrained(
    "./model_weights/photopediaXL_45",
    torch_dtype=torch.float16,
    variant="fp32",
    use_safetensors=True,
)
pipe = pipe.to("cuda")
pipe.enable_vae_slicing()

lora_dirs = [
    "./model_weights/style_lora_realis.safetensors",
    "./model_weights/pastel_colors_xl_v3.safetensors",
    "./model_weights/high_key_lighting_style.safetensors",
    "./model_weights/FilmStockFootageStyle.safetensors",
            ]
lora_scales = [1.0, 0.8, 0.8, 0.6]
for ldir, lsc in zip(lora_dirs, lora_scales):
    pipe.load_lora_weights(ldir)
    pipe.fuse_lora(lora_scale = lsc)


def flush():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
    
def remove_iccp_profile(image_path):
    with PIL.Image.open(image_path) as img:
        img.save(image_path, icc_profile=None)
        
# def single_dominant_color_picker(image_path):
#     color_thief = ColorThief(image_path)
#     dominant_color = color_thief.get_color(quality=1)
#     return dominant_color

def run_example(task_prompt, image):
    prompt = task_prompt
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=True
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    parsed_answer = list(parsed_answer.values())[0]
    return parsed_answer

def generate_assets(scene, image, BATCH_SIZE):
    
    generated_desc = run_example("Describe only the main product in the image" , image)
    print(generated_desc)
    prompt = generated_desc[:100] + " in the background setting: " + scene
    images = pipe(prompt=prompt, negative_prompt=neg_prompt, num_inference_steps=35, width=768, height=512, num_images_per_prompt=BATCH_SIZE).images
    flush()
    # for idx, image in enumerate(images):
    #     image.save(f"/app/{idx+1}.png")
    
    return images, generated_desc


def combine_images(original_image, binary_mask, product_image, safety_overlap_factor = 1.42, overlap_choice = "Opaque"):
    
    bg_path, mask_path = "./bg.png", "./mask.png"
    original_image.save(bg_path)
    PIL.Image.fromarray(binary_mask).save(mask_path)
    sp.call(f"python3 inpaint.py --input_image {bg_path} --input_mask {mask_path} --output_path {bg_path}", shell=True)
    background_image = PIL.Image.open(bg_path).convert("RGBA")
    mask_image = PIL.Image.open(mask_path).convert("L")
    product_image_rgba = product_image.convert("RGBA")
    
    if overlap_choice == "Transparent":
    
        bbox = mask_image.getbbox()
        product_aspect_ratio = product_image_rgba.width / product_image_rgba.height
        if (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) > product_aspect_ratio:
            new_height = int((bbox[3] - bbox[1]) * safety_overlap_factor)
            new_width = int(product_aspect_ratio * new_height)
        else:
            new_width = int((bbox[2] - bbox[0]) * safety_overlap_factor)
            new_height = int(new_width / product_aspect_ratio)

        resized_product_image = product_image_rgba.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
        paste_x = bbox[0] + ((bbox[2] - bbox[0]) - new_width) // 2
        paste_y = bbox[1] + ((bbox[3] - bbox[1]) - new_height) // 2
        background_image.paste(resized_product_image, (paste_x, paste_y), resized_product_image)
    
    else:
        
        r, g, b, alpha = product_image_rgba.split()
        new_alpha = alpha.point(lambda p: 255 if p > 0 else 0)  # Make the product fully opaque
        product_image_rgba = PIL.Image.merge("RGBA", (r, g, b, new_alpha))
        bbox = mask_image.getbbox()
        product_aspect_ratio = product_image_rgba.width / product_image_rgba.height
        if (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) > product_aspect_ratio:
            new_height = int((bbox[3] - bbox[1]) * safety_overlap_factor)
            new_width = int(product_aspect_ratio * new_height)
        else:
            new_width = int((bbox[2] - bbox[0]) * safety_overlap_factor)
            new_height = int(new_width / product_aspect_ratio)

        resized_product_image = product_image_rgba.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
        paste_x = bbox[0] + ((bbox[2] - bbox[0]) - new_width) // 2
        paste_y = bbox[1] + ((bbox[3] - bbox[1]) - new_height) // 2
        background_image.paste(resized_product_image, (paste_x, paste_y), resized_product_image)
        

    return background_image


def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    # output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    # output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image