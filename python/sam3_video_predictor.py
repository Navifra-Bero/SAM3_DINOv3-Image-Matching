import sys
import os

# [중요] Linux 환경 충돌 방지
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# 그 다음 OpenCV
import cv2

# 나머지
import numpy as np
import torch
from PIL import Image

import sam3
from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor

# -----------------------------
# Utils
# -----------------------------
def qpix_from_bgr(frame_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def overlay_mask_bgr(frame_bgr: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    color = np.array([0, 0, 255], dtype=np.float32)  # red in BGR
    if mask_bool is not None and mask_bool.any():
        out[mask_bool] = (out[mask_bool].astype(np.float32) * 0.5 + color * 0.5).astype(np.uint8)
    return out

def resize_mask_to(mask_bool: np.ndarray, w: int, h: int) -> np.ndarray:
    if mask_bool is None:
        return None
    if mask_bool.shape[0] == h and mask_bool.shape[1] == w:
        return mask_bool
    m = mask_bool.astype(np.uint8) * 255
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 127)

def ensure_chw_uint8(frame_bgr: np.ndarray, size: int) -> torch.Tensor:
    """
    tracker가 기대하는 inference_state["images"][i] 형태로 만들기.
    - (C,H,W) uint8
    - 내부에서 tracker가 .cuda().float().unsqueeze(0)로 처리함
    """
    fr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(fr).to(torch.uint8)        # (H,W,3)
    t = t.permute(2, 0, 1).contiguous()             # (3,H,W)
    return t


# -----------------------------
# Core Wrapper
# -----------------------------
class SAM3Wrapper:
    """
    - Image Mode: build_sam3_image_model + Sam3Processor (기존 기능)
    - Video Mode: build_sam3_video_model + Sam3TrackerPredictor 기반 트래킹
      * 중요: video_model이 이미 Sam3TrackerPredictor면 그대로 tracker로 사용
      * 프롬프트 전에 frame0 feature cache 강제 생성
      * 긴 영상은 전체 로드 대신 max_frames만 cv2로 읽어서 state 구성
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

        checkpoint_dir = "/home/park/work/SAM3_DINOv3-Image-Matching/sam3/sam3_models"
        checkpoint_name = "sam3.pt"
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        self.bpe_path = f"{self.sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

        # -------- Image model --------
        print(f"[SAM3] Loading IMAGE model from: {self.checkpoint_path} ...")
        with torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.no_grad():
            self.image_model = build_sam3_image_model(
                checkpoint_path=self.checkpoint_path,
                bpe_path=self.bpe_path
            ).to(self.device).eval()
        self.image_processor = Sam3Processor(self.image_model, confidence_threshold=0.5)
        print("[SAM3] IMAGE Model loaded successfully.")

        # -------- Image state --------
        self.inference_state = None
        self.cv_image = None
        self.width = 0
        self.height = 0

        # -------- Video model / tracker --------
        self.video_model = None
        self.tracker = None
        self.tracker_state = None
        self.track_gen = None

        self.is_tracking_ready = False
        self.track_obj_id = 1
        self.video_path = None
        self.video_cap = None
        self.video_frame0_bgr = None
        self.video_orig_w = None
        self.video_orig_h = None
        self.video_num_frames_loaded = 0

    # -------------------------
    # Image Mode
    # -------------------------
    def set_image(self, image_path: str):
        pil_image = Image.open(image_path).convert("RGB")
        self.width, self.height = pil_image.size
        self.cv_image = cv2.imread(image_path)  # BGR

        with torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.no_grad():
            self.inference_state = self.image_processor.set_image(pil_image)

    def reset_prompts(self):
        if self.inference_state:
            with torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.no_grad():
                self.image_processor.reset_all_prompts(self.inference_state)
        if self.cv_image is not None:
            return qpix_from_bgr(self.cv_image)
        return None

    def run_box_prompt(self, x, y, w, h):
        if self.inference_state is None:
            return None, None, None
        cx = (x + w / 2) / self.width
        cy = (y + h / 2) / self.height
        nw = w / self.width
        nh = h / self.height
        norm_box = [cx, cy, nw, nh]
        print(f"[IMAGE] Prompting Box (xywh normalized center): {norm_box}")
        return self._run_image_inference(prompt_type="box", box=norm_box)

    def run_point_prompt(self, points, labels):
        if self.inference_state is None or not points:
            return None, None, None
        norm_points = []
        for (px, py) in points:
            norm_points.append([px / self.width, py / self.height])
        print(f"[IMAGE] Prompting {len(points)} Points")
        return self._add_point_prompt_manual(norm_points, labels)

    def run_text_prompt(self, text: str):
        if self.inference_state is None or not text:
            return None, None, None
        print(f"[IMAGE] Prompting Text: '{text}'")
        return self._run_image_inference(prompt_type="text", text=text)

    def _run_image_inference(self, prompt_type, **kwargs):
        if self.inference_state is None:
            return None, None, None
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.no_grad():
                self.image_processor.reset_all_prompts(self.inference_state)

                if prompt_type == "box":
                    final_result = self.image_processor.add_geometric_prompt(
                        box=kwargs["box"], label=True, state=self.inference_state
                    )
                elif prompt_type == "text":
                    final_result = self.image_processor.set_text_prompt(
                        state=self.inference_state, prompt=kwargs["text"]
                    )
                else:
                    return None, None, None

                masks = None
                if final_result and isinstance(final_result, dict):
                    masks = final_result.get("pred_masks", None) or final_result.get("masks", None)
                if masks is None:
                    return None, None, None

            return self.visualize_result_opencv(masks)

        except Exception as e:
            print(f"[IMAGE] Inference Error ({prompt_type}): {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def _add_point_prompt_manual(self, points, labels):
        state = self.inference_state
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.no_grad():
                if "backbone_out" not in state:
                    raise ValueError("You must call set_image before prompting")

                if "language_features" not in state["backbone_out"]:
                    dummy_text_outputs = self.image_model.backbone.forward_text(["visual"], device=self.device)
                    state["backbone_out"].update(dummy_text_outputs)

                state["geometric_prompt"] = self.image_model._get_dummy_prompt()

                points_tm = torch.tensor(points, device=self.device, dtype=torch.float32).view(1, -1, 2)
                labels_tm = torch.tensor(labels, device=self.device, dtype=torch.int32).view(1, -1)

                if hasattr(state["geometric_prompt"], "append_points"):
                    state["geometric_prompt"].append_points(points_tm, labels_tm)
                else:
                    print("[IMAGE] Error: missing append_points.")
                    return None, None, None

                final_result = self.image_processor._forward_grounding(state)
                masks = final_result.get("pred_masks", None) or final_result.get("masks", None)
                if masks is None:
                    return None, None, None

            return self.visualize_result_opencv(masks)

        except Exception as e:
            print(f"[IMAGE] Manual Point Prompt Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def visualize_result_opencv(self, masks_tensor):
        """
        return:
          - QPixmap overlay
          - final_mask (H,W) bool
          - instance_masks (N,H,W) bool
        """
        if self.cv_image is None or masks_tensor is None:
            return None, None, None

        masks = masks_tensor.float().cpu().numpy()
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks.ndim == 2:
            masks = masks[None, ...]

        inst_list = []
        for m in masks:
            if m.shape[:2] != (self.height, self.width):
                m = cv2.resize(m, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            inst_list.append(m > 0.0)

        instance_masks = np.stack(inst_list, axis=0).astype(bool)
        final_mask = np.any(instance_masks, axis=0)

        vis_img = overlay_mask_bgr(self.cv_image, final_mask)
        return qpix_from_bgr(vis_img), final_mask, instance_masks

    # -------------------------
    # Video Mode
    # -------------------------
    
    def _is_tracker_like(self, obj) -> bool:
        if obj is None:
            return False
        need = [
            "init_state",
            "add_new_points_or_box",
            "propagate_in_video_preflight",
            "propagate_in_video",
            "_get_image_feature",
        ]
        return all(hasattr(obj, k) for k in need)

    def _unwrap_tracker(self, video_model):
        """
        build_sam3_video_model()이 반환한 wrapper 내부에서 tracker/predictor를 찾아 꺼낸다.
        """
        print(f"[DEBUG] _unwrap_tracker received type: {type(video_model)}")
        print(f"[DEBUG] video_model class name: {video_model.__class__.__name__}")
        
        # 0) 이미 tracker면 바로 반환
        if self._is_tracker_like(video_model):
            print("[DEBUG] video_model is already tracker-like, returning as-is")
            return video_model
        if isinstance(video_model, Sam3TrackerPredictor):
            print("[DEBUG] video_model is Sam3TrackerPredictor, returning as-is")
            return video_model

        # 1) 모든 attributes 출력
        all_attrs = [x for x in dir(video_model) if not x.startswith('_')]
        print(f"[DEBUG] All non-private attributes: {all_attrs[:20]}")  # 처음 20개만
        
        # 2) 흔히 있는 attribute 후보들
        candidates = []
        attr_names = [
            "tracker", "predictor", "video_predictor", "sam3_tracker", "sam3_predictor",
            "model", "net", "network", "module", "core", "core_model",
            "video_model", "video_net"
        ]
        
        print("[DEBUG] Checking candidate attributes...")
        for name in attr_names:
            if hasattr(video_model, name):
                attr = getattr(video_model, name)
                print(f"[DEBUG]   Found '{name}': {type(attr)}")
                candidates.append((name, attr))

        # 3) 1단계 후보에서 tracker-like 찾기
        for name, c in candidates:
            if self._is_tracker_like(c):
                print(f"[DEBUG] Found tracker-like object at '{name}'")
                return c
            if isinstance(c, Sam3TrackerPredictor):
                print(f"[DEBUG] Found Sam3TrackerPredictor at '{name}'")
                return c

        # 4) 그래도 못 찾으면 2-depth로 한번 더 뒤져봄
        print("[DEBUG] Checking 2-depth attributes...")
        for parent_name, c in candidates:
            if c is None:
                continue
            for name in attr_names:
                if hasattr(c, name):
                    cc = getattr(c, name)
                    print(f"[DEBUG]   Found '{parent_name}.{name}': {type(cc)}")
                    if self._is_tracker_like(cc):
                        print(f"[DEBUG] Found tracker-like at '{parent_name}.{name}'")
                        return cc
                    if isinstance(cc, Sam3TrackerPredictor):
                        print(f"[DEBUG] Found Sam3TrackerPredictor at '{parent_name}.{name}'")
                        return cc

        # 5) 실패 - 더 자세한 정보 출력
        print("[ERROR] Cannot find tracker!")
        print(f"[DEBUG] video_model type: {type(video_model)}")
        print(f"[DEBUG] video_model has backbone? {hasattr(video_model, 'backbone')}")
        print(f"[DEBUG] video_model has forward_image? {hasattr(video_model, 'forward_image')}")
        
        # build_sam3_video_model이 직접 tracker를 반환했을 가능성
        if hasattr(video_model, 'backbone') and hasattr(video_model, 'forward_image'):
            print("[DEBUG] video_model itself has backbone and forward_image, using it directly")
            return video_model
        
        raise RuntimeError(
            f"Cannot unwrap tracker from video_model of type {type(video_model)}. "
            f"Please check the output of build_sam3_video_model()."
        )

    def load_video(self, video_path: str, max_frames: int = 300, stride: int = 1):
        """
        JPEG 프레임 폴더로 변환해서 메모리 효율적으로 로드
        """
        if self.video_model is None:
            from sam3.model_builder import build_sam3_video_predictor
            print(f"[SAM3] Loading VIDEO predictor...")
            self.video_model = build_sam3_video_predictor(gpus_to_use=[0])
            print("[SAM3] VIDEO predictor loaded.")
        
        # 1. 임시 폴더에 JPEG 프레임 저장
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        saved_idx = 0
        
        print(f"[VIDEO] Extracting frames (max={max_frames}, stride={stride})...")
        
        while saved_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (frame_idx % stride) == 0:
                frame_path = os.path.join(temp_dir, f"{saved_idx:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                if saved_idx == 0:
                    self.video_frame0_bgr = frame.copy()
                    h, w = frame.shape[:2]
                
                saved_idx += 1
            
            frame_idx += 1
        
        cap.release()
        
        print(f"[VIDEO] Saved {saved_idx} frames to {temp_dir}")
        
        # 2. JPEG 폴더로 세션 시작 (메모리 효율적)
        response = self.video_model.handle_request(
            request=dict(
                type="start_session",
                resource_path=temp_dir,  # 폴더 경로
            )
        )
        self.session_id = response["session_id"]
        self.temp_dir = temp_dir
        
        self.cv_image = self.video_frame0_bgr.copy()
        self.height, self.width = h, w
        
        # Image processor용
        pil0 = Image.fromarray(cv2.cvtColor(self.video_frame0_bgr, cv2.COLOR_BGR2RGB))
        with torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.no_grad():
            self.inference_state = self.image_processor.set_image(pil0)
        
        self.is_tracking_ready = False
        
        return self.video_frame0_bgr

    def video_ensure_frame0_cached(self):
        """
        ✅ Video tracker의 forward_image()로 frame0 feature를 생성하고 캐시에 저장
        tracker.forward_image()는 backbone 호출 + SAM decoder preprocessing을 수행
        """
        if self.tracker_state is None:
            return

        cf = self.tracker_state.setdefault("cached_features", {})
        
        # 이미 캐시되어 있으면 스킵
        if 0 in cf and cf[0] is not None and cf[0][1] is not None:
            print("[VIDEO] frame0 already cached, skipping.")
            return

        print("[VIDEO] Creating frame0 features cache using tracker.forward_image()...")
        
        with torch.inference_mode():
            try:
                # (1) frame0 이미지 텐서 준비
                img0 = self.tracker_state["images"][0]  # (3,H,W) uint8
                img0_batch = img0.to(self.device).float().unsqueeze(0)  # (1,3,H,W) float
                backbone_out = self.tracker.forward_image(img0_batch)
                
                # (3) cached_features에 직접 저장
                cf[0] = (img0_batch, backbone_out)
                
                print(f"[VIDEO] frame0 cached successfully. Keys in backbone_out: {list(backbone_out.keys())}")
                
            except Exception as e:
                print(f"[ERROR] Failed to create frame0 cache: {e}")
                import traceback
                traceback.print_exc()
                raise

    def video_add_box_prompt(self, x, y, w, h):
        """공식 API로 box 프롬프트 추가"""
        if self.session_id is None:
            return None, None, None
        
        # 1) 프리뷰용 (image processor로 첫 프레임에서 마스크 미리보기)
        pix, final_mask, inst = self.run_box_prompt(x, y, w, h)
        
        # 2) 공식 API로 tracker에 box 등록
        # ✅ 중요: boxes_xywh 형식 (center_x, center_y, width, height)로 상대 좌표
        cx = (x + w/2) / self.width
        cy = (y + h/2) / self.height
        bw = w / self.width
        bh = h / self.height
        
        boxes_xywh = np.array([[cx, cy, bw, bh]], dtype=np.float32)
        
        try:
            response = self.video_model.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.session_id,
                    frame_index=0,
                    boxes=boxes_xywh,  # ✅ 'box'가 아니라 'boxes'
                )
            )
            
            print(f"[VIDEO] Box prompt added. Response keys: {response.keys()}")
            
            # response에서 outputs 확인
            if "outputs" in response:
                out = response["outputs"]
                print(f"[VIDEO] Got outputs with obj_ids: {list(out.keys())}")
            
            self.is_tracking_ready = True
            
        except Exception as e:
            print(f"[ERROR] Failed to add box prompt: {e}")
            import traceback
            traceback.print_exc()
        
        return pix, final_mask, inst


    def video_add_point_prompt(self, points, labels):
        """공식 API로 point 프롬프트 추가"""
        if self.session_id is None or not points:
            return None, None, None
        
        # 1) 프리뷰
        pix, final_mask, inst = self.run_point_prompt(points, labels)
        
        # 2) 상대 좌표로 변환
        points_rel = [[x / self.width, y / self.height] for x, y in points]
        points_tensor = torch.tensor(points_rel, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int32)
        
        try:
            response = self.video_model.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.session_id,
                    frame_index=0,
                    points=points_tensor,
                    point_labels=labels_tensor,
                )
            )
            
            print(f"[VIDEO] Point prompt added. Response keys: {response.keys()}")
            
            if "outputs" in response:
                out = response["outputs"]
                print(f"[VIDEO] Got outputs with obj_ids: {list(out.keys())}")
            
            self.is_tracking_ready = True
            
        except Exception as e:
            print(f"[ERROR] Failed to add point prompt: {e}")
            import traceback
            traceback.print_exc()
        
        return pix, final_mask, inst


    def video_add_text_prompt(self, text: str):
        """공식 API로 text 프롬프트 추가"""
        if self.session_id is None or not text:
            return None, None, None
        
        # 1) 프리뷰
        pix, final_mask, inst = self.run_text_prompt(text)
        
        # 2) 공식 API로 text 프롬프트
        try:
            response = self.video_model.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.session_id,
                    frame_index=0,
                    text=text,
                )
            )
            
            print(f"[VIDEO] Text prompt added. Response keys: {response.keys()}")
            
            if "outputs" in response:
                out = response["outputs"]
                print(f"[VIDEO] Got outputs with obj_ids: {list(out.keys())}")
            
            self.is_tracking_ready = True
            
        except Exception as e:
            print(f"[ERROR] Failed to add text prompt: {e}")
            import traceback
            traceback.print_exc()
        
        return pix, final_mask, inst

    def video_start_tracking(self, start_frame_idx=0):
        # session 기반 스트림 추적 사용
        if self.session_id is None or self.video_model is None:
            return False
        if not self.is_tracking_ready:
            return False

        # stream(generator)
        self.track_gen = self.video_model.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=self.session_id,
                propagation_direction="forward",
                start_frame_index=start_frame_idx,
                max_frame_num_to_track=None,
            )
        )
        return True


    def video_next_tracked_frame(self):
        """
        next: (frame_idx, obj_ids, masks_logits_or_probs...)
        실제 반환 포맷은 predictor 버전에 따라 달라서, 여기서는 가장 흔한 형태를 try로 핸들링.
        """
        if self.track_gen is None:
            return None

        try:
            out = next(self.track_gen)
        except StopIteration:
            return None

        return out


# -----------------------------
# Canvas (이미지/비디오 공용)
# -----------------------------
class ImageCanvas(QLabel):
    box_drawn = pyqtSignal(int, int, int, int)
    points_updated = pyqtSignal(list, list)
    mask_remove_requested = pyqtSignal(int, int)

    def __init__(self, parent=None, is_interactive=True):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.is_interactive = is_interactive
        self.current_pixmap = None

        self.current_mask_data = None       # (H,W) bool union
        self.current_instance_masks = None  # (N,H,W) bool

        self.mode = "box"
        self.start_point = None
        self.end_point = None
        self.is_drawing = False

        self.points = []
        self.labels = []

    def set_image(self, pixmap, mask_data=None, instance_masks=None):
        self.current_pixmap = pixmap
        self.current_mask_data = mask_data
        self.current_instance_masks = instance_masks
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_mode(self, mode):
        self.mode = mode
        self.clear_interaction()

    def clear_interaction(self):
        self.start_point = None
        self.end_point = None
        self.points = []
        self.labels = []
        self.update()

    def resizeEvent(self, event):
        if self.current_pixmap:
            self.setPixmap(self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)

    def get_real_coordinates(self, pos):
        if not self.current_pixmap:
            return None

        lbl_w, lbl_h = self.width(), self.height()
        pix = self.pixmap()
        if not pix:
            return None
        pix_w, pix_h = pix.width(), pix.height()

        off_x = (lbl_w - pix_w) / 2
        off_y = (lbl_h - pix_h) / 2

        rel_x = pos.x() - off_x
        rel_y = pos.y() - off_y

        if rel_x < 0 or rel_x >= pix_w or rel_y < 0 or rel_y >= pix_h:
            return None

        orig_w = self.current_pixmap.width()
        orig_h = self.current_pixmap.height()

        scale_x = orig_w / pix_w
        scale_y = orig_h / pix_h

        real_x = int(rel_x * scale_x)
        real_y = int(rel_y * scale_y)
        return real_x, real_y

    def mousePressEvent(self, event):
        if not self.is_interactive or not self.current_pixmap:
            return

        real_coords = self.get_real_coordinates(event.pos())
        if not real_coords:
            return
        rx, ry = real_coords

        # 우클릭: "클릭한 위치 포함하는 인스턴스 1개만 제거"
        if event.button() == Qt.RightButton:
            if self.current_instance_masks is not None:
                h, w = self.current_instance_masks.shape[-2], self.current_instance_masks.shape[-1]
                if 0 <= ry < h and 0 <= rx < w:
                    hit = self.current_instance_masks[:, ry, rx]
                    if hit.any():
                        self.clear_interaction()
                        self.mask_remove_requested.emit(rx, ry)
            return

        # 좌클릭: box/point
        if self.mode == "box":
            if event.button() == Qt.LeftButton:
                self.start_point = event.pos()
                self.end_point = event.pos()
                self.is_drawing = True
                self.update()

        elif self.mode == "point":
            if event.button() == Qt.LeftButton:
                label = 1
                self.points.append((rx, ry))
                self.labels.append(label)
                self.update()
                self.points_updated.emit(self.points, self.labels)

    def mouseMoveEvent(self, event):
        if self.mode == "box" and self.is_drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.mode == "box" and self.is_drawing and event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.end_point = event.pos()
            self.update()

            p1 = self.get_real_coordinates(self.start_point)
            p2 = self.get_real_coordinates(self.end_point)
            if p1 and p2:
                x = min(p1[0], p2[0])
                y = min(p1[1], p2[1])
                w = abs(p2[0] - p1[0])
                h = abs(p2[1] - p1[1])
                if w > 5 and h > 5:
                    self.box_drawn.emit(x, y, w, h)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        if self.mode == "box" and self.is_drawing and self.start_point and self.end_point:
            painter.setPen(QPen(Qt.green, 3, Qt.SolidLine))
            painter.drawRect(QRect(self.start_point, self.end_point).normalized())

        if self.mode == "point" and self.points:
            lbl_w, lbl_h = self.width(), self.height()
            pix = self.pixmap()
            if not pix:
                return

            off_x = (lbl_w - pix.width()) / 2
            off_y = (lbl_h - pix.height()) / 2

            orig_w = self.current_pixmap.width()
            orig_h = self.current_pixmap.height()
            scale_x = pix.width() / orig_w
            scale_y = pix.height() / orig_h

            for (px, py), lbl in zip(self.points, self.labels):
                screen_x = int(px * scale_x + off_x)
                screen_y = int(py * scale_y + off_y)
                painter.setBrush(QBrush(Qt.green))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPoint(screen_x, screen_y), 5, 5)


# -----------------------------
# Main App
# -----------------------------
class SAM3App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 Image/Video Interactive (Instance Remove + Video Tracking)")
        self.resize(1600, 1000)

        self.sam_wrapper = SAM3Wrapper()

        self.mode_source = None  # "image" or "video"
        self.image_path = None
        self.video_path = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_video_tick)
        self.fps = 30

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Controls
        control_layout = QHBoxLayout()

        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.clicked.connect(self.load_image)

        self.btn_load_video = QPushButton("Load Video")
        self.btn_load_video.clicked.connect(self.load_video)

        self.group_mode = QButtonGroup(self)
        self.radio_box = QRadioButton("Box Mode")
        self.radio_point = QRadioButton("Point Mode")
        self.radio_box.setChecked(True)
        self.group_mode.addButton(self.radio_box)
        self.group_mode.addButton(self.radio_point)
        self.radio_box.toggled.connect(self.change_prompt_mode)
        self.radio_point.toggled.connect(self.change_prompt_mode)

        self.btn_start = QPushButton("Start (Video Tracking)")
        self.btn_start.clicked.connect(self.start_video_tracking)

        self.btn_reset = QPushButton("Reset All")
        self.btn_reset.clicked.connect(self.reset_all)

        control_layout.addWidget(self.btn_load_image)
        control_layout.addWidget(self.btn_load_video)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.radio_box)
        control_layout.addWidget(self.radio_point)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_reset)

        main_layout.addLayout(control_layout)

        # Canvas
        self.canvas = ImageCanvas(is_interactive=True)
        self.canvas.setStyleSheet("border: 2px solid gray;")
        self.canvas.box_drawn.connect(self.on_box_prompt)
        self.canvas.points_updated.connect(self.on_point_prompt)
        self.canvas.mask_remove_requested.connect(self.remove_one_mask_at)
        main_layout.addWidget(self.canvas)

        # Text prompt
        text_layout = QHBoxLayout()
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("Enter text prompt (e.g. 'wheel')...")
        self.input_text.returnPressed.connect(self.on_text_prompt)

        self.btn_text_run = QPushButton("Run Text")
        self.btn_text_run.clicked.connect(self.on_text_prompt)

        text_layout.addWidget(QLabel("Text:"))
        text_layout.addWidget(self.input_text)
        text_layout.addWidget(self.btn_text_run)
        main_layout.addLayout(text_layout)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(180)
        main_layout.addWidget(self.log)

    def logi(self, s: str):
        self.log.append(s)
        print(s)

    def change_prompt_mode(self):
        if self.radio_box.isChecked():
            self.canvas.set_mode("box")
        else:
            self.canvas.set_mode("point")

    # -------------------------
    # Load Image / Video
    # -------------------------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return

        self.timer.stop()
        self.mode_source = "image"
        self.image_path = path
        self.video_path = None

        self.sam_wrapper.set_image(path)
        self.canvas.set_image(QPixmap(path), mask_data=None, instance_masks=None)
        self.reset_all()
        self.logi("[MODE] Image loaded. Use prompts.")

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return

        self.timer.stop()
        self.mode_source = "video"
        self.video_path = path
        self.image_path = None

        # max_frames / stride 입력
        max_frames, ok = QInputDialog.getInt(self, "Video Load", "Max frames to load (memory-safe):", 300, 10, 2000, 10)
        if not ok:
            return
        stride, ok = QInputDialog.getInt(self, "Video Load", "Frame stride (1=all, 2=every 2nd ...):", 1, 1, 30, 1)
        if not ok:
            return

        frame0 = self.sam_wrapper.load_video(path, max_frames=max_frames, stride=stride)
        self.canvas.set_image(qpix_from_bgr(frame0), mask_data=None, instance_masks=None)

        self.logi("[MODE] Video loaded. Prompt on FIRST frame (box/point/text), then press Start.")
        self.logi(f"[VIDEO] frames loaded into state = {self.sam_wrapper.video_num_frames_loaded}")

    # -------------------------
    # Reset / Remove mask instance
    # -------------------------
    def reset_all(self):
        self.canvas.clear_interaction()

        if self.mode_source == "image" and self.image_path:
            self.canvas.set_image(QPixmap(self.image_path), mask_data=None, instance_masks=None)
            if self.sam_wrapper.inference_state:
                self.sam_wrapper.image_processor.reset_all_prompts(self.sam_wrapper.inference_state)
            self.logi("[RESET] Image prompts reset.")

        elif self.mode_source == "video" and self.video_path:
            self.timer.stop()
            # 첫 프레임 다시 보여주기
            if self.sam_wrapper.video_frame0_bgr is not None:
                self.sam_wrapper.cv_image = self.sam_wrapper.video_frame0_bgr.copy()
                self.sam_wrapper.height, self.sam_wrapper.width = self.sam_wrapper.cv_image.shape[:2]
                self.canvas.set_image(qpix_from_bgr(self.sam_wrapper.cv_image), mask_data=None, instance_masks=None)

            # tracker_state는 프레임/세션 유지하되 prompt 관련 상태만 초기화하고 싶으면
            # 가장 안전한 방법: 비디오를 다시 로드하는 것.
            # 여기서는 간단히 "다시 로드하라고 안내"
            self.sam_wrapper.is_tracking_ready = False
            self.sam_wrapper.track_gen = None
            self.logi("[RESET] Video prompt cleared (tracking generator reset). Re-prompt on frame0.")

        else:
            self.logi("[RESET] Nothing loaded.")

    def remove_one_mask_at(self, rx, ry):
        inst = self.canvas.current_instance_masks
        if inst is None or self.sam_wrapper.cv_image is None:
            return

        h, w = inst.shape[-2], inst.shape[-1]
        if not (0 <= ry < h and 0 <= rx < w):
            return

        hit = inst[:, ry, rx]
        if not hit.any():
            return

        remove_idx = int(np.where(hit)[0][0])
        new_inst = np.delete(inst, remove_idx, axis=0)

        if new_inst.shape[0] == 0:
            clean_pixmap = self.sam_wrapper.reset_prompts()
            if clean_pixmap:
                self.canvas.set_image(clean_pixmap, mask_data=None, instance_masks=None)
                self.canvas.clear_interaction()
            return

        final_mask = np.any(new_inst, axis=0)
        vis_img = overlay_mask_bgr(self.sam_wrapper.cv_image, final_mask)
        pix = qpix_from_bgr(vis_img)
        self.canvas.set_image(pix, mask_data=final_mask, instance_masks=new_inst)

    # -------------------------
    # Prompt handlers
    # -------------------------
    def on_box_prompt(self, x, y, w, h):
        if self.mode_source == "image":
            QApplication.setOverrideCursor(Qt.WaitCursor)
            pix, m, inst = self.sam_wrapper.run_box_prompt(x, y, w, h)
            if pix:
                self.canvas.set_image(pix, m, inst)
            QApplication.restoreOverrideCursor()
            return

        if self.mode_source == "video":
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                # 여기서 죽지 않게 보호
                pix, m, inst = self.sam_wrapper.video_add_box_prompt(x, y, w, h)
                if pix:
                    self.canvas.set_image(pix, m, inst)
                self.logi("[VIDEO] Box prompt added on frame0. Now press Start.")
            except Exception as e:
                self.logi(f"[VIDEO] Box prompt failed: {e}")
                import traceback; traceback.print_exc()
            finally:
                QApplication.restoreOverrideCursor()
            return


    def on_point_prompt(self, points, labels):
        if self.mode_source == "image":
            pix, m, inst = self.sam_wrapper.run_point_prompt(points, labels)
            if pix:
                self.canvas.set_image(pix, m, inst)
            return

        if self.mode_source == "video":
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                pix, m, inst = self.sam_wrapper.video_add_point_prompt(points, labels)
                if pix:
                    self.canvas.set_image(pix, m, inst)
                self.logi("[VIDEO] Point prompt added on frame0. Now press Start.")
            except Exception as e:
                self.logi(f"[VIDEO] Point prompt failed: {e}")
                import traceback; traceback.print_exc()
            finally:
                QApplication.restoreOverrideCursor()
            return


    def on_text_prompt(self):
        text = self.input_text.text().strip()
        if not text:
            return

        if self.mode_source == "image":
            QApplication.setOverrideCursor(Qt.WaitCursor)
            pix, m, inst = self.sam_wrapper.run_text_prompt(text)
            if pix:
                self.canvas.set_image(pix, m, inst)
            QApplication.restoreOverrideCursor()
            return

        if self.mode_source == "video":
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                text = self.input_text.text().strip()
                pix, m, inst = self.sam_wrapper.video_add_text_prompt(text)
                if pix:
                    self.canvas.set_image(pix, m, inst)
                self.logi("[VIDEO] Text prompt applied on frame0. Now press Start.")
            except Exception as e:
                self.logi(f"[VIDEO] Text prompt failed: {e}")
                import traceback; traceback.print_exc()
            finally:
                QApplication.restoreOverrideCursor()
            return


    # -------------------------
    # Video tracking loop
    # -------------------------
    def start_video_tracking(self):
        if self.mode_source != "video":
            QMessageBox.information(self, "Info", "Video mode에서만 Start가 동작합니다.")
            return

        if not self.sam_wrapper.is_tracking_ready:
            QMessageBox.critical(self, "Start Error", "프롬프트(박스/포인트/텍스트)를 먼저 입력해서 초기 객체를 지정해주세요.")
            return

        ok = self.sam_wrapper.video_start_tracking(start_frame_idx=0)
        if not ok:
            QMessageBox.critical(self, "Start Error", "tracking generator를 시작할 수 없습니다.")
            return

        self.fps = 30 
        self.timer.start(int(1000 / max(1, self.fps)))
        self.logi("[VIDEO] Tracking started.")

    def on_video_tick(self):
        out = self.sam_wrapper.video_next_tracked_frame()
        if out is None:
            self.timer.stop()
            self.logi("[VIDEO] Tracking finished.")
            return

        # predictor 반환 포맷이 버전에 따라 다르므로 try로 여러 형태 처리
        # 일반적으로:
        #   (frame_idx, obj_ids, masks) 또는 dict 형태 가능
        frame_idx = None
        masks = None

        if isinstance(out, tuple) and len(out) >= 3:
            frame_idx = out[0]
            obj_ids = out[1]
            masks = out[2]
        elif isinstance(out, dict):
            frame_idx = out.get("frame_idx", None) or out.get("frame_index", None)
            masks = out.get("masks", None) or out.get("pred_masks", None)
        else:
            # 알 수 없는 포맷
            return

        # 프레임 표시용: 우리는 state에 “리사이즈된 정사각(image_size)” 프레임을 넣었으므로,
        # overlay도 그 기준으로 보여준다.
        # 원본 프레임 표시를 원하면 별도 cv2 cap로 원본을 계속 읽어야 하는데,
        # 지금은 메모리 안전/동작 안정성이 우선이라 state 기반 프레임으로 보여줌.
        try:
            # session 기반이면 temp_dir에 저장된 jpg로 프레임 로드
            if getattr(self.sam_wrapper, "temp_dir", None) is not None:
                fp = os.path.join(self.sam_wrapper.temp_dir, f"{frame_idx:05d}.jpg")
                fr_bgr = cv2.imread(fp)
                if fr_bgr is None:
                    return
            else:
                # (혹시 tracker_state가 있으면 기존 방식 유지)
                img_t = self.sam_wrapper.tracker_state["images"][frame_idx]
                fr = img_t.permute(1, 2, 0).cpu().numpy()
                fr_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        except Exception:
            return

        # masks 처리
        final_mask = None
        if masks is not None:
            if torch.is_tensor(masks):
                mm = masks.float().detach().cpu().numpy()
                # 보통 (num_obj,1,H,W) 또는 (num_obj,H,W)
                if mm.ndim == 4:
                    mm = mm.squeeze(1)
                if mm.ndim == 3:
                    final_mask = (mm > 0.0).any(axis=0)
                elif mm.ndim == 2:
                    final_mask = (mm > 0.0)
        if final_mask is not None:
            vis = overlay_mask_bgr(fr_bgr, final_mask)
        else:
            vis = fr_bgr

        self.sam_wrapper.cv_image = vis.copy()  # 우클릭 삭제 등에 사용
        self.canvas.set_image(qpix_from_bgr(vis), mask_data=final_mask, instance_masks=None)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    
    print("1. Starting QApplication...")
    app = QApplication(sys.argv)
    
    print("2. Creating SAM3App...")
    try:
        win = SAM3App()
        print("3. Showing window...")
        win.show()
        print("4. Starting event loop...")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()