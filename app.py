from typing import Tuple, Optional, List
import gradio as gr
import supervision as sv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import torch, os, re, ast, random, base64, glob, PIL.Image, cv2, gc, requests, copy, time
import subprocess as sp, gradio as gr, pandas as pd, numpy as np
from diffusers import AutoPipelineForText2Image
from utils.video import generate_unique_name, create_directory, delete_directory
from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_DETAILED_CAPTION_TASK, \
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, \
    IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
from utils.sam import load_sam_image_model, run_sam_inference
from utils.helper_functions import *
from background_removal.bgrem import bgrem_inference


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("thwri/CogFlorence-2.1-Large", trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained("thwri/CogFlorence-2.1-Large", trust_remote_code=True)


torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=device)
SAM_IMAGE_MODEL = load_sam_image_model(device=device)
# COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLORS = ['#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5
)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX
)

MARKDOWN = """
# Product Consistent Image Generation
"""
image_path = "mascot.png" 
base64_image = convert_image_to_base64(image_path)
html_code = f"""
<style>
  .bottom-left-container {{
    position: relative;
    min-height: 100vh;
  }}
  .bottom-left-image {{
    position: fixed;
    bottom: -30px;
    left: -20px;
    width: 400px;
    height: 400px;
  }}
</style>
<div class="bottom-left-container">
  <img src="data:image/jpeg;base64,{base64_image}" alt="My Image" class="bottom-left-image">
</div>
"""

flush()



@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(
    image_input, text_input, safety_overlap_factor, overlap_choice
) -> List[PIL.Image.Image]:
    if not image_input:
        gr.Info("Please upload an image.")
        return None, None
    
    if not text_input:
        gr.Info("Please enter a text prompt.")
        return None, None
    
    images, text_input = generate_assets(text_input, image_input, BATCH_SIZE=3)
    texts = [prompt.strip() for prompt in text_input.split(",")]
    
    input_image_path = "/app/bg_input.png"
    sp.call(f"rm -rf {input_image_path}", shell=True)
    image_input.save(input_image_path)
    product_image = PIL.Image.open(bgrem_inference(input_image_path))
    annotations = []
    for image_input in images:
        detections_list = []
        for text in texts:
            print(text)
            _, result = run_florence_inference(
                model=FLORENCE_MODEL,
                processor=FLORENCE_PROCESSOR,
                device=device,
                image=image_input,
                task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                text=text
            )
            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2,
                result=result,
                resolution_wh=image_input.size
            )
            
            if detections.xyxy.size != 0:
                detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
                detections_list.append(detections)
            else:
                print(f"Detection XYXY size 0")
            
        detections = sv.Detections.merge(detections_list)
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        original_image = image_input
        mask = np.any(detections.mask, axis=0)
        binary_mask = (mask * 255).astype(np.uint8)
        
        annotations.append(combine_images(original_image, binary_mask, product_image, safety_overlap_factor = safety_overlap_factor, overlap_choice = overlap_choice))
        flush()
        
    return annotations


with gr.Blocks(theme="NoCrypt/miku") as demo:
    gr.Markdown(MARKDOWN)
    with gr.Tab("Image"):
        
        with gr.Row():
            with gr.Column():
                image_processing_image_input_component = gr.Image(
                    type='pil', label='Upload image')
                with gr.Row():
                    image_processing_text_input_component = gr.Textbox(
                        label='Describe the background scene',
                        placeholder='Enter a prompt')
                    safety_overlap_factor = gr.Number(value=1.3, label='Resize Ratio')
                with gr.Row():
                    overlap_choice = gr.Radio(choices=["Transparent", "Opaque"], value="Opaque", label="Product Transparency")
                image_processing_submit_button_component = gr.Button(
                    value='Submit', variant='primary')
            with gr.Column():
                image_processing_image_output_component = gr.Gallery(label="Image Output", interactive=False)


    image_processing_submit_button_component.click(
        fn=process_image,
        inputs=[
            image_processing_image_input_component,
            image_processing_text_input_component,
            safety_overlap_factor,
            overlap_choice
        ],
        outputs=[
            image_processing_image_output_component
        ]
    )
    image_processing_text_input_component.submit(
        fn=process_image,
        inputs=[
            image_processing_image_input_component,
            image_processing_text_input_component,
            safety_overlap_factor,
            overlap_choice
        ],
        outputs=[
            image_processing_image_output_component
        ]
    )
    gr.HTML(html_code)

demo.launch(debug=False, show_error=True, share=True)