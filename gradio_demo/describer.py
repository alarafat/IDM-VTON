import re
import numpy as np
from PIL import Image

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


class Describer:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            # quantization_config=bnb_config,
            torch_dtype="auto",
            device_map="auto"
        )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(model_name)

    @staticmethod
    def parse_output_description(response: str):
        """
        Parses Qwen response like:
        "['[upper] This is an upper body clothing item, specifically a shirt...']"
        Returns: cloth_position = "upper", garment_desc = "This is ..."
        """
        # Normalize the input string (strip brackets/quotes if it's a list-like string)
        response = response.strip().lower()
        response = response.replace("['", "").replace("']", "").replace('"', '').strip()

        # Try to find 'upper' or 'lower' position
        match = re.search(r'\b(upper|lower)\b', response)
        cloth_position = match.group(1) if match else "unknown"

        # Try to extract a full sentence as description (after position)
        desc_start = response.find(cloth_position) + len(cloth_position)
        garment_desc = response[desc_start:].strip(" []:,.")

        # Fallback to default description
        if not garment_desc:
            if cloth_position == "upper":
                garment_desc = "It's an upper part cloth"
            elif cloth_position == "lower":
                garment_desc = "It's a lower part cloth"
            else:
                garment_desc = "It's a cloth item"

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
