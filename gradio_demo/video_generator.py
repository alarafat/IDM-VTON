import torch
import numpy as np
from PIL.Image import Image
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel


class VideoGenerator:
    def __init__(self):
        print('Loading models.............')
        # Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
        model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
        print('✅ Models loaded successfully')
        self.pipe.to("cuda")

    def process(self, image: Image, prompt: str):
        max_area = 720 * 1280
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        print("⏳ Generating video frames...")
        output = self.pipe(image=image,
                           prompt=prompt,
                           negative_prompt=negative_prompt,
                           height=height, width=width,
                           num_frames=81,
                           guidance_scale=5.0).frames[0]

        print("✅ Video generation complete.")
        export_to_video(output, "output.mp4", fps=16)

        return output.frames[0]

def preview_video(frames, fps=16):
    from matplotlib import pyplot as plt
    from matplotlib import animation

    fig = plt.figure(figsize=(6, 6))
    plt.axis('off')
    im = plt.imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=True)
    plt.show(block=True)

def main():
    image = load_image('example/human/00034_00.jpg')
    video_gen = VideoGenerator()

    prompt = (
        "A full-body person slowly turning around in place on a white seamless background, as if walking in a circle. "
        "The person is wearing fashionable clothes. "
        "Studio lighting, high quality, realistic shadows, ultra high resolution, detailed clothing texture, realistic movement, "
        "smooth camera tracking from front to back, centered composition, cinematic look."
    )

    frames = video_gen.process(image, prompt)
    preview_video(frames)


if __name__ == '__main__':
    main()
