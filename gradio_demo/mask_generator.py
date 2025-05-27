import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any


import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# from utils.visualization import show_masks



class ObjectSegmentor:
    """
    A class that combines Florence-2 for object detection and SAM2 for segmentation.
    Given an image and a text prompt, it detects objects and SAM2 uses that bbox to return segmented mask.
    """

    def __init__(self,
                 florence_model_name: str = "microsoft/Florence-2-large",
                 sam2_config: str = "sam2_hiera_l.yaml",
                 sam2_checkpoint: str = "sam2_hiera_large.pt",
                 device: str = "auto"):
        """
        Initialize the ObjectSegmentor with Florence-2 and SAM2 models.

        Args:
            florence_model_name: Name of the Florence-2 model to use
            sam2_config: SAM2 configuration file
            sam2_checkpoint: SAM2 checkpoint file
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.device = self._get_device(device)

        # Initialize Florence-2
        print("Loading Florence-2 model...")
        self.florence_model = AutoModelForCausalLM.from_pretrained(florence_model_name,
                                                                   # torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                                                                   torch_dtype='auto',
                                                                   trust_remote_code=True).eval().to(self.device)
        self.florence_processor = AutoProcessor.from_pretrained(florence_model_name, trust_remote_code=True)

        # Initialize SAM2
        print("Loading SAM2 model...")
        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        print(f"Models loaded successfully on {self.device}")

    @staticmethod
    def _get_device(device: str) -> torch.device:
        """Determine the appropriate device for inference."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _prepare_image(in_image) -> np.ndarray:
        """Convert PIL image formats to numpy array and make sure it's 3-channel."""
        if isinstance(in_image, Image.Image):
            # PIL Image
            in_image = in_image.convert('RGB')
            in_image = np.array(in_image)
        elif isinstance(in_image, np.ndarray):
            # Already numpy array
            if len(in_image.shape) == 3 and in_image.shape[2] == 3:
                pass  # RGB format
            else:
                raise ValueError("Numpy array should be in RGB format with shape (H, W, 3)")
        else:
            raise ValueError("Image should be PIL Image, or numpy array")

        return in_image

    @staticmethod
    def _enlarge_bounding_box(box: np.ndarray,
                              image_shape: Tuple[int, int],
                              enlargement_factor: float = 0.15) -> List[float]:
        """
        Enlarge bounding box to ensure complete clothing coverage.

        Args:
            box: Original bounding box [x1, y1, x2, y2]
            image_shape: Image dimensions (height, width)
            enlargement_factor: How much to enlarge (0.15 = 15% larger on each side)

        Returns:
            Enlarged bounding box
        """
        x1, y1, x2, y2 = box
        height, width = image_shape[:2]

        # Calculate current box dimensions
        box_width = x2 - x1
        box_height = y2 - y1

        # Calculate enlargement amounts
        width_expand = box_width * enlargement_factor
        height_expand = box_height * enlargement_factor

        # Enlarge the box
        new_x1 = int(max(0, x1 - width_expand))
        new_y1 = int(max(0, y1 - height_expand))
        new_x2 = int(min(width, x2 + width_expand))
        new_y2 = int(min(height, y2 + height_expand))

        return [new_x1, new_y1, new_x2, new_y2]

    @staticmethod
    def _enlarge_mask(mask: np.ndarray,
                      dilation_iterations: int = 5,
                      kernel_size: int = 5) -> np.ndarray:
        """
        Enlarge the segmentation mask to cover more of the clothing area.

        Args:
            mask: Original binary mask
            dilation_iterations: Number of dilation iterations
            kernel_size: Size of the dilation kernel

        Returns:
            Enlarged mask
        """
        # Create dilation kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Convert to uint8 if needed
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Apply morphological dilation to enlarge the mask
        enlarged_mask = cv2.dilate(mask_uint8, kernel, iterations=dilation_iterations)

        # Apply Gaussian blur for smoother edges
        enlarged_mask = cv2.GaussianBlur(enlarged_mask, (3, 3), 0)

        # Convert back to boolean
        enlarged_mask = (enlarged_mask > 127).astype(np.float32)

        return enlarged_mask

    @staticmethod
    def _convert_to_od_format(data):
        """
        Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.

        Parameters:
        - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.

        Returns:
        - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.
        """
        # Extract bounding boxes and labels
        bboxes = data.get('bboxes', [])
        labels = data.get('bboxes_labels', [])

        # Construct the output format
        od_results = {
            'bboxes': bboxes,
            'labels': labels
        }

        return od_results

    def detect_objects(self, in_image: Image, od_text_prompt: str) -> Dict[str, Any]:
        """
        Use Florence-2 to detect objects based on the prompt.

        Args:
            in_image: Input image as numpy array
            od_text_prompt: Text prompt describing what to detect

        Returns:
            Dictionary containing detection results
        """
        # Convert numpy array to PIL Image for Florence-2
        if isinstance(in_image, np.ndarray):
            in_image = Image.fromarray(in_image)

        task_prompt = f"<OPEN_VOCABULARY_DETECTION>"
        if od_text_prompt is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + od_text_prompt

        # Process inputs
        inputs = self.florence_processor(text=prompt,
                                         images=in_image,
                                         return_tensors="pt").to(self.device)

        # Generate predictions
        with torch.no_grad():
            generated_ids = self.florence_model.generate(input_ids=inputs["input_ids"].to(self.device),
                                                         pixel_values=inputs["pixel_values"].to(self.device, dtype=torch.float16),
                                                         max_new_tokens=1024,
                                                         early_stopping=False,
                                                         do_sample=False,
                                                         num_beams=3)

        # Decode results
        generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        generated_text = generated_text.replace("</s>", "").replace("<s>", "").strip()
        # cleaned_text = "<OPEN_VOCABULARY_DETECTION>" + generated_text

        # Parse the results
        parsed_answer = self.florence_processor.post_process_generation(generated_text, task=task_prompt, image_size=(in_image.width, in_image.height))
        bbox_results = self._convert_to_od_format(parsed_answer['<OPEN_VOCABULARY_DETECTION>'])

        return bbox_results

    def segment_objects(self,
                        in_image: np.ndarray,
                        in_bboxes: np.ndarray,
                        enlargement_factor: float = 0.15,
                        use_multimask: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use SAM2 to segment objects based on bounding boxes.

        Args:
            in_image: Input image as numpy array
            in_bboxes: Bounding boxes from Florence-2
            enlargement_factor: Factor to enlarge the bbox
            use_multimask: Whether to return multiple mask options

        Returns:
            Tuple of (masks, scores, logits)
        """
        if len(in_bboxes) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Set image for SAM2 predictor
        self.sam2_predictor.set_image(in_image)

        # Enlarge bounding boxes
        enlarged_box = self._enlarge_bounding_box(in_bboxes, in_image.shape, enlargement_factor)
        enlarged_box = np.array(enlarged_box)

        # Predict masks
        masks, scores, logits = self.sam2_predictor.predict(point_coords=None, point_labels=None, box=enlarged_box, multimask_output=use_multimask, )

        return masks, scores, logits, np.array(enlarged_box)

    def process(self,
                in_image,
                prompt: str,
                box_enlargement: float = 0.15,
                mask_dilation: int = 5,
                use_multimask: bool = True) -> Dict[str, Any]:
        image_np = self._prepare_image(in_image)

        # -------------------------------------------------------
        # Detect objects with Florence-2
        # -------------------------------------------------------
        print(f"Detecting objects with prompt: '{prompt}'")
        detection_results = self.detect_objects(image_np, prompt)

        # Extract bounding boxes and labels
        boxes = detection_results.get('bboxes', [])

        print(f"Found {len(boxes)} objects")

        if not boxes:
            return {'image': image_np,
                    'detections': detection_results,
                    'mask': np.array([]),
                    'score': 0.0,
                    'box': np.array([]), }
        best_box = boxes[0]

        # -----------------------------------------------------
        # Segment objects with SAM2
        # -----------------------------------------------------
        print("Segmenting detected objects...")
        masks, scores, _, boxes = self.segment_objects(image_np, best_box, enlargement_factor=box_enlargement, use_multimask=use_multimask)

        if len(masks) == 0:
            return {
                'image': image_np,
                'detections': detection_results,
                'mask': np.array([]),
                'score': 0.0,
                'box': np.array(best_box),
                'enlarged_box': np.array([]),
            }

        # Select the best mask (highest score)
        best_mask_idx = np.argmax(scores[0])
        best_mask = masks[best_mask_idx]

        # Further enlarge the mask using morphological operations
        if mask_dilation > 0:
            best_mask = self._enlarge_mask(best_mask, dilation_iterations=mask_dilation)

        best_mask = np.expand_dims(best_mask, axis=0)
        return {'image': image_np,
                'detections': detection_results,
                'masks': best_mask,
                'scores': scores,
                'boxes': boxes, }


    # @staticmethod
    # def save_masks(results: Dict[str, Any], output_dir: str = "./masks/") -> List[str]:
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     masks = results['masks']
    #     saved_files = []
    #
    #     for i, mask in enumerate(masks):
    #         if len(mask.shape) == 3:
    #             mask = mask[0]  # Take first mask if multimask
    #
    #         # Convert boolean mask to 0-255 uint8
    #         mask_img = (mask * 255).astype(np.uint8)
    #
    #         # Save mask
    #         filename = f"output_mask_{i}.png"
    #         filepath = os.path.join(output_dir, filename)
    #         cv2.imwrite(filepath, mask_img)
    #         saved_files.append(filepath)
    #
    #     return saved_files
    #
    # @staticmethod
    # def get_mask_statistics(results: Dict[str, Any]) -> Dict[str, float]:
    #     """
    #     Get statistics about the segmentation mask coverage.
    #     """
    #     mask = results['masks'][0]
    #
    #     if len(mask) == 0:
    #         return {'coverage_percentage': 0.0, 'mask_area': 0, 'image_area': 0}
    #
    #     mask_area = np.sum(mask > 0)
    #     image_area = mask.shape[0] * mask.shape[1]
    #     coverage_percentage = (mask_area / image_area) * 100
    #
    #     return {
    #         'coverage_percentage': coverage_percentage,
    #         'mask_area': int(mask_area),
    #         'image_area': int(image_area)
    #     }
    #

# if __name__ == "__main__":
#     # Initialize the parameters
#     florence2_model_name = "microsoft/Florence-2-large"
#     sam2_checkpoint = "sam2_fils/checkpoints/sam2.1_hiera_large.pt"
#     sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
#     input_image_path = "sample_data/people/00330_00.jpg"
#
#     box_extend_factor = 0       # 20% larger bounding box
#     mask_dilation = 20          # More mask dilation for better coverage
#     use_multimask = False
#     text_prompt = "Upper body cloth of the person"
#
#     # Load the input image
#     image = cv2.imread(input_image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Load the Mask generator class
#     segmentor = ObjectSegmentor(florence_model_name=florence2_model_name, sam2_config=sam2_cfg, sam2_checkpoint=sam2_checkpoint)
#
#     # Segment upper clothing with enlarged mask
#     results = segmentor.process(in_image=image,
#                                 prompt=text_prompt,
#                                 box_enlargement=box_extend_factor,
#                                 mask_dilation=mask_dilation,
#                                 use_multimask=False)
#
#     # Print results
#     if len(results['masks']) > 0:
#         # print(f"Successfully segmented: {results['labels']}")
#         print(f"Segmentation confidence: {results['scores'][0]:.3f}")
#
#         # Get mask statistics
#         stats = segmentor.get_mask_statistics(results)
#         print(f"Mask covers {stats['coverage_percentage']:.1f}% of image area")
#         print(f"Mask area: {stats['mask_area']} pixels")
#
#         # Visualize results
#         show_masks(image, results['masks'], results['scores'], box_coords=results['boxes'])
#
#         # Save the enlarged mask
#         mask_path = segmentor.save_masks(results, "enlarged_upper_clothing_mask.png")
#
#     else:
#         print("No clothing items detected")
