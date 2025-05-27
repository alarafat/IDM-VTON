import os
import sys
import numpy as np
from typing import List
from PIL import Image, ImageDraw
import streamlit as st

import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

# Import the required modules from the original code
sys.path.append('./')
import apply_net
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from src.unet_hacked_tryon import UNet2DConditionModel
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

from transformers import (CLIPImageProcessor,
                          CLIPVisionModelWithProjection,
                          CLIPTextModel,
                          CLIPTextModelWithProjection)
from diffusers import DDPMScheduler, AutoencoderKL

from describer import Describer
from utils_mask import get_mask_location
from mask_generator import ObjectSegmentor

import gc
torch.cuda.empty_cache()
gc.collect()

# Set page config
st.set_page_config(
    page_title="Virtual Try-On",
    page_icon="👕",
    layout="wide"
)

# Initialize device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


@st.cache_resource
def load_models():
    """Load all the required models with caching"""
    base_path = 'yisol/IDM-VTON'

    # Load UNet models
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)

    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    )
    UNet_Encoder.requires_grad_(False)

    # Load tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )

    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    # Load text encoders
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )

    # Load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        base_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )

    # Set requires_grad to False
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Create pipeline
    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe.unet_encoder = UNet_Encoder

    # Load the Qwen Model for image descriptions and SAM for mask generation
    cloth_describer = Describer()
    # mask_predictor = load_sam_model()

    # Load parsing and openpose models (for original method)
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)

    # Load the Florence2+SAM2 model for mask generation
    florence2_model_name = "microsoft/Florence-2-large"
    sam2_checkpoint = "gradio_demo/sam2_files/checkpoints/sam2.1_hiera_large.pt"
    # sam2_checkpoint = "sam2_files/checkpoints/sam2.1_hiera_large.pt"
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    mask_predictor = ObjectSegmentor(florence_model_name=florence2_model_name, sam2_config=sam2_cfg, sam2_checkpoint=sam2_checkpoint)

    return pipe, cloth_describer, mask_predictor, parsing_model, openpose_model


def perform_tryon(human_img,
                  garment_img,
                  mask_method,
                  pipe,
                  cloth_describer,
                  mask_predictor,
                  parsing_model,
                  openpose_model,
                  denoise_steps=30,
                  seed=42):
    cloth_describer.model.to(device)
    mask_predictor.to(device)
    openpose_model.preprocessor.body_estimation.model.to(device)

    # Get description of the cloth
    question = ("What is this dress? Is it upper body clothing like a shirt or t-shirt, or lower body clothing like pants? "
                "Classify between [upper/lower], write only [upper or lower], no extra word. Add only one sentence description about the cloth, it's color, cloth type, etc.")
    cloth_position, garment_desc = cloth_describer.describe_image(garment_img, question=question)

    if 'upper' in cloth_position:
        text_prompt = "Upper body cloth of the person"
        # mask_prompt = "Segment the entire upper body garment in the image. Make sure the segmentation includes the full shirt or t-shirt, including sleeves, shoulders, and neckline. It’s okay if the mask includes parts of the mannequin or background, but do not exclude any visible part of the clothing."
    elif 'lower' in cloth_position:
        text_prompt = "Lower body cloth of the person"
        # mask_prompt = "Segment the entire lower body garment in the image. Include the full pants, skirt, or shorts from waist to hem. The mask should cover the complete garment even if it overlaps with background or legs. Avoid excluding any part of the clothing."
    else:
        text_prompt = "Upper body cloth of the person"
        # mask_prompt = 'Segment the main garment in the image. Ensure that the segmentation includes the entire clothing item as shown, even if some parts are partially occluded or overlap with background or body. The mask can be slightly larger than the garment, but must not miss any part of it.'

    print("\n📌 Qwen Output:")
    print(cloth_position, " Cloth Description: ", garment_desc)



    # Prepare images
    garm_img = garment_img.convert("RGB").resize((768, 1024))
    human_img_orig = human_img.convert("RGB")
    human_img = human_img_orig.resize((768, 1024))

    # Generate mask based on selected method
    if mask_method == "sam":
        # text_prompt = "Upper body cloth of the person"
        results = mask_predictor.process(in_image=human_img,
                                         prompt=text_prompt,
                                         box_enlargement=0,
                                         mask_dilation=20,
                                         use_multimask=False)
        # mask = results['masks'][0]
        # mask = mask.resize((768, 1024))

        binary_mask = results['masks'][0]
        mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8), mode='L')
        mask = mask_pil.resize((768, 1024))
    else:
        openpose_model.preprocessor.body_estimation.model.to(device)
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))

    # Prepare tensor transforms
    tensor_transfrom = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    # Prepare pose image (required for the pipeline)
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args((
        'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
        './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v',
        '--opts', 'MODEL.DEVICE', 'cuda'
    ))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))


    # Release GPU memory for pipe loading
    cloth_describer.model.to("cpu")
    mask_predictor.unload_from_gpu()
    openpose_model.preprocessor.body_estimation.model.to("cpu")


    # Generate try-on result
    pipe.to(device)
    pipe.unet_encoder.to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Encode prompts for human
                prompt = "model is wearing " + garment_desc
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * 1

                with torch.inference_mode():
                    (prompt_embeds,
                     negative_prompt_embeds,
                     pooled_prompt_embeds,
                     negative_pooled_prompt_embeds,) = pipe.encode_prompt(prompt,
                                                                          num_images_per_prompt=1,
                                                                          do_classifier_free_guidance=True,
                                                                          negative_prompt=negative_prompt, )

                    # Encode prompts for garment
                    prompt_c = "a photo of " + garment_desc
                    (prompt_embeds_c, _, _, _,) = pipe.encode_prompt([prompt_c],
                                                                     num_images_per_prompt=1,
                                                                     do_classifier_free_guidance=False,
                                                                     negative_prompt=negative_prompt)

                    # Prepare tensors
                    pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
                    garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

                    # Generate images
                    images = pipe(prompt_embeds=prompt_embeds.to(device, torch.float16),
                                  negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                                  pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                                  negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                                  num_inference_steps=denoise_steps,
                                  generator=generator,
                                  strength=1.0,
                                  pose_img=pose_img_tensor.to(device, torch.float16),
                                  text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                                  cloth=garm_tensor.to(device, torch.float16),
                                  mask_image=mask,
                                  image=human_img,
                                  height=1024,
                                  width=768,
                                  ip_adapter_image=garm_img.resize((768, 1024)),
                                  guidance_scale=2.0, )[0]

    return images[0], mask_gray, mask


def main():
    st.title("👕 Virtual Try-On")
    st.markdown("Upload a person's photo and a garment image to see the virtual try-on result!")

    # Load models
    pipe, cloth_describer, mask_predictor, parsing_model, openpose_model = load_models()

    # Create columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Person Image")

        # Option to use webcam or upload
        input_method = st.radio("Choose input method:",
                                ["Upload Image", "Use Webcam (Coming Soon)"],
                                key="input_method")

        if input_method == "Upload Image":
            human_img = st.file_uploader("Upload person image",
                                         type=['png', 'jpg', 'jpeg'],
                                         key="human")

            if human_img is not None:
                human_image = Image.open(human_img)
                # st.image(human_image, caption="Person Image", use_column_width=True)
                st.image(human_image, caption="Person Image", width=300)
        else:
            st.info("Webcam functionality will be added in the next update!")
            human_image = None

    with col2:
        st.subheader("👔 Garment Image")
        garment_img = st.file_uploader("Upload garment image",
                                       type=['png', 'jpg', 'jpeg'],
                                       key="garment")

        if garment_img is not None:
            garment_image = Image.open(garment_img)
            # st.image(garment_image, caption="Garment Image", use_column_width=True)
            st.image(garment_image, caption="Garment Image", width=300)


    # Masking method selection
    st.subheader("🎯 Mask Generation Method")
    mask_method = st.selectbox("Choose masking method:",
                               ["sam", "original"],
                               format_func=lambda x: {"sam": "SAM - Segment Anything Model (Best Quality)",
                                                      "original": "Original Human Parsing (Slow but Accurate)"}[x],
                               help="SAM provides the best results but requires additional setup")

    # Try-on button
    if st.button("🎯 Try On!", type="primary", use_container_width=True):
        if input_method == "Upload Image" and human_img is not None and garment_img is not None:
            with st.spinner("Generating try-on result... This may take a few moments."):
                try:
                    result_img, mask_img, generated_mask = perform_tryon(human_image,
                                                                         garment_image,
                                                                         mask_method,
                                                                         pipe,
                                                                         cloth_describer,
                                                                         mask_predictor,
                                                                         parsing_model,
                                                                         openpose_model)

                    # Display results
                    st.subheader("✨ Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(result_img, caption="Try-On Result", use_column_width=True)
                    with col2:
                        st.image(generated_mask, caption="Generated Mask", use_column_width=True)
                    with col3:
                        st.image(mask_img, caption="Processed Mask", use_column_width=True)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please make sure all model files are properly installed and accessible.")
        else:
            st.warning("Please upload both images and provide a garment description.")

    # Information boxes
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.info("💡 **Masking Methods:**\n"
                "- **SAM**: Best quality, requires setup\n"
                "- **Original**: Uses human parsing, most accurate for complex poses")

    with col2:
        st.info("🚀 **Coming Soon:**\n"
                "- Webcam integration\n"
                "- Qwen-powered garment description\n"
                "- Real-time try-on preview\n"
                "- Multiple garment categories")

    # Footer
    st.markdown("---")
    st.markdown("📚 **Setup Requirements:** This demo requires IDM-VTON model files. "
                "Check out the [source code](https://github.com/yisol/IDM-VTON) and "
                "[model](https://huggingface.co/yisol/IDM-VTON) for setup instructions.")


if __name__ == "__main__":
    main()
