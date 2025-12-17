import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Sam3Processor, Sam3Model, Sam3TrackerProcessor, Sam3TrackerModel
import torch.nn.functional as F


class SAM3Predictor:
    def __init__(self, model_path, device='cuda'):
        print(f"[SAM3] Loading model from {model_path}...")
        self.device = device
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model = Sam3Model.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model.half()
        print("[SAM3] Model loaded successfully.")

    def predict(self, image_array, text_prompt, conf=0.4):
        """
        Text-based prediction
        Args:
            image_array: numpy array (H, W, 3) - BGR format (Opencv)
            text_prompt: str (ex: "car", "wheel")
            conf: float threshold
        Returns:
            polygons: List of numpy arrays [[x, y], [x, y], ...]
            labels: List of label indices (or names)
            texts: List of text labels
        """
        if not text_prompt:
            print("[SAM3] Error: No text prompt provided.")
            return None, None, None

        pil_img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        
        texts = [t.strip() for t in text_prompt.split(',')]
        
        inputs = self.processor(
            images=pil_img, 
            text=texts, 
            return_tensors="pt"
        ).to(self.device)

        if torch.cuda.is_available():
             inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=conf,
            mask_threshold=0.5,
            target_sizes=[(pil_img.height, pil_img.width)]
        )[0]

        masks = results['masks'].cpu().numpy() # (N, H, W)
        
        if 'labels' in results:
            class_indices = results['labels'].cpu().numpy()
        elif 'classes' in results:
            class_indices = results['classes'].cpu().numpy()
        else:
            class_indices = np.zeros(len(masks), dtype=int)

        final_polygons = []
        final_labels = []

        for i, mask in enumerate(masks):
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 50:
                    continue
                
                polygon = contour.squeeze().astype(float)
                
                if len(polygon) < 3:
                    continue
                
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True).squeeze().astype(float)
                
                if len(approx) < 3: 
                    continue

                final_polygons.append(approx)
                final_labels.append(class_indices[i])

        return final_polygons, final_labels, texts

class SAM3TrackerPointPredictor:
    """
    진짜 Point Prompt 전용 SAM3 Tracker 모델
    - 텍스트 프롬프트용 SAM3Predictor와 별도로 사용
    - 클릭 좌표 (x, y)를 그대로 point prompt로 넣어서 segmentation
    """
    def __init__(self, model_path, device="cuda"):
        print(f"[SAM3-Tracker] Loading tracker model from {model_path}...")
        self.device = device
        self.processor = Sam3TrackerProcessor.from_pretrained(model_path)
        self.model = Sam3TrackerModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.half()

    def predict_by_point(self, image_array, point_x, point_y, multimask_output=False):
        """
        Args:
            image_array: (H, W, 3) BGR (OpenCV)
            point_x, point_y: 원본 이미지 좌표 (float 또는 int)
        Returns:
            approx: (N, 2) polygon (x, y) / 없으면 None
        """
        pil_img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        w, h = pil_img.size

        input_points = [[[[float(point_x), float(point_y)]]]]
        input_labels = [[[1]]]  # 1 = positive click

        inputs = self.processor(
            images=pil_img,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)

        if torch.cuda.is_available():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    inputs[k] = v.half()

        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=multimask_output)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]

        if masks.dim() == 4:
            masks = masks[0]
        elif masks.dim() != 3:
            print(f"[SAM3-Tracker] Unexpected mask dim: {masks.shape}")
            return None

        masks_np = masks.numpy()  # (num_masks, H, W)
        num_masks, H, W = masks_np.shape

        if num_masks == 0:
            print("[SAM3-Tracker] No mask returned.")
            return None

        cx = int(np.clip(point_x, 0, W - 1))
        cy = int(np.clip(point_y, 0, H - 1))

        chosen_idx = None
        for i in range(num_masks):
            if masks_np[i, cy, cx]:
                chosen_idx = i
                break

        if chosen_idx is None:
            areas = masks_np.reshape(num_masks, -1).sum(axis=1)
            chosen_idx = int(areas.argmax())

        mask = masks_np[chosen_idx]  # (H, W)

        mask_uint8 = (mask.astype(np.uint8)) * 255

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 50:
            return None

        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True).squeeze().astype(float)

        if approx.ndim == 1:
            approx = approx.reshape(-1, 2)
        if len(approx) < 3:
            return None

        return approx
