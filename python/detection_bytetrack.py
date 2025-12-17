import sys
import os
import time
# [중요] Linux 환경 충돌 방지
# os.environ["QT_QPA_PLATFORM"] = "xcb"
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import cv2
import numpy as np
import torch
from PIL import Image

# === SAM3 GitHub 코드 임포트 (경로 확인 필요) ===
import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3Wrapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

        # 모델 경로 설정 (사용자 환경에 맞게 수정 필요)
        checkpoint_dir = "/home/park/work/SAM3_DINOv3-Image-Matching/sam3/sam3_models"
        checkpoint_name = "sam3.pt"
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        self.bpe_path = f"{self.sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

        print(f"Loading SAM3 model from: {self.checkpoint_path}...")

        # 모델 로드
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.model = build_sam3_image_model(
                checkpoint_path=self.checkpoint_path,
                bpe_path=self.bpe_path
            )

        self.processor = Sam3Processor(self.model, confidence_threshold=0.4) # Threshold 조정 가능
        self.inference_state = None
        print("SAM3 Model loaded successfully.")

    def predict_frame(self, frame_bgr, text_prompt):
        """
        비디오의 한 프레임(BGR)을 받아 텍스트 프롬프트로 세그멘테이션 후
        마스크가 적용된 QImage를 반환
        """
        # 1. 이미지 변환 (BGR -> RGB -> PIL)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        height, width = frame_bgr.shape[:2]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # 2. 이미지 인코딩 (매 프레임 수행 - 무거운 작업)
            # SAM3 Processor는 set_image 시 이미지를 인코딩함
            self.inference_state = self.processor.set_image(pil_image)

            # 3. 텍스트 프롬프트 추론
            # text_prompt가 없으면 원본 반환
            if not text_prompt:
                return self._to_qimage(frame_rgb)

            # 프롬프트 초기화 후 텍스트 적용
            self.processor.reset_all_prompts(self.inference_state)
            final_result = self.processor.set_text_prompt(
                state=self.inference_state,
                prompt=text_prompt
            )

            # 4. 결과 추출
            masks = final_result.get('pred_masks', None) or final_result.get('masks', None)
            
            if masks is None:
                return self._to_qimage(frame_rgb)

            # 5. 시각화 (OpenCV 오버레이)
            vis_img = self._apply_mask_opencv(frame_bgr, masks, width, height)
            
            # BGR -> RGB 변환 후 QImage 리턴
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            return self._to_qimage(vis_img_rgb)

    def _apply_mask_opencv(self, img_bgr, masks_tensor, w, h):
        """마스크 텐서를 이미지에 오버레이"""
        masks = masks_tensor.float().cpu().numpy()

        # 차원 정리: (N, 1, H, W) -> (N, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks.ndim == 2:
            masks = masks[None, ...]

        # 마스크 합치기 (Union)
        final_mask = np.zeros((h, w), dtype=bool)
        
        for m in masks:
            # 해상도 맞추기 (필요시)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            final_mask = np.logical_or(final_mask, m > 0)

        # 빨간색 오버레이
        vis_img = img_bgr.copy()
        color = np.array([0, 0, 255]) # BGR: Red
        
        # 마스크 영역 블렌딩
        vis_img[final_mask] = vis_img[final_mask] * 0.6 + color * 0.4
        return vis_img

    def _to_qimage(self, img_rgb):
        h, w, ch = img_rgb.shape
        return QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)


class VideoCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background-color: black;")
        self.current_pixmap = None

    def update_frame(self, qimg):
        self.current_pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        if self.current_pixmap:
            self.setPixmap(self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)


class SAM3VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 Video Detection (Frame-by-Frame)")
        self.resize(1280, 800)

        # SAM3 Wrapper 초기화
        self.sam_wrapper = SAM3Wrapper()

        # 비디오 관련 변수
        self.video_cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        self.video_fps = 30
        self.total_frames = 0
        
        self.current_text_prompt = ""

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. 캔버스 (영상 출력)
        self.canvas = VideoCanvas()
        main_layout.addWidget(self.canvas, stretch=1)

        # 2. 컨트롤 패널
        control_layout = QHBoxLayout()

        # 비디오 로드 버튼
        self.btn_load = QPushButton("Load Video")
        self.btn_load.clicked.connect(self.load_video)
        control_layout.addWidget(self.btn_load)

        # 재생/일시정지
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        control_layout.addWidget(self.btn_play)

        # 슬라이더
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.valueChanged.connect(self.slider_moved)
        self.slider.setEnabled(False)
        control_layout.addWidget(self.slider)

        main_layout.addLayout(control_layout)

        # 3. 텍스트 프롬프트 입력
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Text Prompt:"))
        
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("Enter object name (e.g. cat, car)...")
        self.input_text.returnPressed.connect(self.apply_prompt)
        prompt_layout.addWidget(self.input_text)

        self.btn_apply = QPushButton("Apply Prompt")
        self.btn_apply.clicked.connect(self.apply_prompt)
        prompt_layout.addWidget(self.btn_apply)

        # 프롬프트 리셋 버튼
        self.btn_reset_prompt = QPushButton("Clear Prompt")
        self.btn_reset_prompt.clicked.connect(self.clear_prompt)
        prompt_layout.addWidget(self.btn_reset_prompt)

        main_layout.addLayout(prompt_layout)

        # 상태 표시줄
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return

        if self.video_cap:
            self.video_cap.release()

        self.video_cap = cv2.VideoCapture(path)
        if not self.video_cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video.")
            return

        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.video_fps <= 0: self.video_fps = 30
        
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_play.setText("Play")
        self.is_playing = False
        
        # 첫 프레임 보여주기
        self.next_frame()
        self.status_label.setText(f"Loaded: {os.path.basename(path)} | FPS: {self.video_fps:.2f}")

    def toggle_play(self):
        if not self.video_cap:
            return

        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("Play")
            self.is_playing = False
        else:
            # SAM3 추론 속도가 느리므로 타이머 간격을 여유있게 설정하거나, 
            # 가능한 빠르게(0ms) 설정하여 추론이 끝나는 대로 다음 프레임 진행
            # 여기서는 추론 시간을 고려해 1ms로 설정 (Blocking 방식이라 추론 끝나야 다음 틱)
            self.timer.start(1) 
            self.btn_play.setText("Pause")
            self.is_playing = True

    def next_frame(self):
        if not self.video_cap:
            return

        ret, frame = self.video_cap.read()
        if not ret:
            self.timer.stop()
            self.btn_play.setText("Play")
            self.is_playing = False
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop or Stop
            return

        current_frame_idx = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # 슬라이더 업데이트 (시그널 블락하여 루프 방지)
        self.slider.blockSignals(True)
        self.slider.setValue(current_frame_idx)
        self.slider.blockSignals(False)

        # === SAM3 Inference & Visualization ===
        t_start = time.time()
        
        # 여기서 SAM3 추론 실행 (텍스트가 있으면)
        qimg = self.sam_wrapper.predict_frame(frame, self.current_text_prompt)
        
        t_end = time.time()
        inference_time = (t_end - t_start) * 1000 # ms

        # 화면 업데이트
        self.canvas.update_frame(qimg)
        
        # 상태 업데이트 (FPS 표시)
        if self.current_text_prompt:
            self.status_label.setText(f"Processing '{self.current_text_prompt}'... ({inference_time:.1f}ms/frame)")
        else:
            self.status_label.setText("Playing (No Prompt)")

    def apply_prompt(self):
        text = self.input_text.text().strip()
        if text:
            self.current_text_prompt = text
            self.status_label.setText(f"Prompt set to: {text}")
            # 현재 정지 상태면 한 프레임 갱신해서 결과 보여주기
            if not self.is_playing and self.video_cap:
                # 현재 프레임을 다시 읽으려면 뒤로 한칸 가야 함
                cur_pos = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)
                if cur_pos > 0:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos - 1)
                self.next_frame()

    def clear_prompt(self):
        self.current_text_prompt = ""
        self.input_text.clear()
        self.status_label.setText("Prompt cleared.")
        # 현재 화면 갱신 (마스크 제거)
        if not self.is_playing and self.video_cap:
            cur_pos = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)
            if cur_pos > 0:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos - 1)
            self.next_frame()

    # 슬라이더 조작
    def slider_pressed(self):
        self.was_playing = self.is_playing
        if self.is_playing:
            self.toggle_play()

    def slider_released(self):
        if self.was_playing:
            self.toggle_play()

    def slider_moved(self, value):
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            # 드래그 중에는 추론 없이 원본만 보여주거나, 끊김을 감수하고 추론
            # 여기서는 부드러운 탐색을 위해 텍스트 프롬프트 없이 보여줌 (옵션)
            ret, frame = self.video_cap.read()
            if ret:
                # 슬라이더 이동 시에는 빠른 반응을 위해 추론 생략하고 이미지만 표시
                # 만약 추론을 보고 싶다면 self.sam_wrapper.predict_frame(frame, self.current_text_prompt) 사용
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.canvas.update_frame(qimg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 스타일 (선택사항)
    app.setStyle("Fusion")
    
    win = SAM3VideoApp()
    win.show()
    sys.exit(app.exec_())