import numpy as np
import torch
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


class Describer:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            # torch_dtype="auto",
            device_map="auto"
        )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(model_name)

    @staticmethod
    def parse_output_description(response: str):
        # Remove any leading/trailing whitespace and brackets
        lines = response.strip().split("\n")

        if not lines:
            return "unknown", "no description"

        # Extract position from the first line
        cloth_position = lines[0].strip().lower().replace('[', '').replace(']', '')

        # Extract description from the second line or concatenate the rest
        garment_desc = " ".join(line.strip() for line in lines[1:] if line.strip())

        return cloth_position, garment_desc

    def describe_image(self, image, question):
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt",
                                )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(output_text)

        cloth_position, cloth_desc = self.parse_output_description(output_text[0])

        return cloth_position, cloth_desc
