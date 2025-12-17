import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from transformers import AutoImageProcessor, AutoModel

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# SAM3 모듈 Import (없을 경우 더미 처리)
try:
    from core.sam3_model_inference import SAM3TrackerPointPredictor, SAM3Predictor
except ImportError:
    print("Warning: SAM3 modules not found. Using dummy functions for UI testing.")
    class SAM3TrackerPointPredictor:
        def __init__(self, *args, **kwargs): pass
        def predict_by_point(self, img, x, y): return None
    class SAM3Predictor:
        def __init__(self, *args, **kwargs): pass
        def predict(self, img, text, conf=0.4): return [], [], []

class DinoV3Matcher:
    def __init__(self, model_name='facebook/dinov3-vitl16-pretrain-lvd1689m', device='cuda'):
        self.device = device
        print(f"Loading DINOv3 model: {model_name}...")
        
        # [중요] 본인의 HuggingFace 토큰을 입력하세요        
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, token=my_token)
            self.model = AutoModel.from_pretrained(model_name, token=my_token).to(self.device)
            self.model.eval()
            print("DINOv3 Loaded successfully.")
        except OSError as e:
            raise e 

    def extract_features(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 5:, :] 
            features = F.normalize(features, dim=-1)
            
        h, w = inputs['pixel_values'].shape[-2:]
        patch_size = 16 
        grid_h, grid_w = h // patch_size, w // patch_size
        
        return features, (grid_h, grid_w), (h, w)

    def get_embedding(self, img_bgr):
        features, _, _ = self.extract_features(img_bgr)
        embedding = features.mean(dim=1) 
        return embedding

    def find_matches_with_vector(self, query_vectors, target_img, threshold=0.6):
        """
        N개의 Query Vector와 Target Image를 비교하여 매칭되는 좌표들을 반환
        """
        t_feat, (th, tw), (orig_h, orig_w) = self.extract_features(target_img)
        
        # Batch Matrix Multiplication
        # query: (N, Dim), target: (Batch, Patches, Dim)
        sim_batch = torch.einsum('nd,bkd->bnk', query_vectors, t_feat)
        
        # Max Pooling: N개의 쿼리 중 가장 유사한 값을 선택
        sim_max, _ = sim_batch.max(dim=1) 
        sim_map = sim_max.reshape(th, tw).cpu().numpy()
        
        # Normalize
        min_val, max_val = sim_map.min(), sim_map.max()
        sim_map_norm = (sim_map - min_val) / (max_val - min_val + 1e-8)
        
        # Local Maxima Finding
        import scipy.ndimage as ndimage
        data_max = ndimage.maximum_filter(sim_map_norm, size=3)
        maxima = (sim_map_norm == data_max)
        
        data_min = sim_map_norm > threshold
        maxima[data_min == 0] = 0
        
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        
        matched_points = []
        patch_size = 16
        
        for dy, dx in slices:
            grid_y = (dy.start + dy.stop - 1) // 2
            grid_x = (dx.start + dx.stop - 1) // 2
            
            center_x = int((grid_x + 0.5) * patch_size)
            center_y = int((grid_y + 0.5) * patch_size)
            
            final_x = int(center_x * (target_img.shape[1] / orig_w))
            final_y = int(center_y * (target_img.shape[0] / orig_h))
            
            matched_points.append((final_x, final_y))
            
        print(f"[DINOv3] Found {len(matched_points)} matches (Threshold: {threshold})")
        return matched_points

class ImageLabel(QLabel):
    clicked_pos = pyqtSignal(int, int)
    right_clicked_pos = pyqtSignal(int, int)
    box_selected = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.polygons = [] 
        self.draw_color = (0, 255, 0)
        self.cv_image = None 
        
        self.mode = "Point" 
        self.is_drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()

    def set_mode(self, mode):
        self.mode = mode
        self.setCursor(Qt.ArrowCursor if mode == "Point" else Qt.CrossCursor)

    def set_image(self, cv_img):
        self.cv_image = cv_img
        self.polygons = [] 
        self.update_view()

    def update_view(self):
        if self.cv_image is None: return
        h, w, c = self.cv_image.shape
        qimg = QImage(cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB), w, h, w*c, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        
        if self.width() > 0 and self.height() > 0:
            scaled_pix = pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pix)

    def resizeEvent(self, event):
        self.update_view()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.right_clicked_pos.emit(event.x(), event.y())
            return

        if event.button() == Qt.LeftButton:
            if self.mode == "Point":
                self.clicked_pos.emit(event.x(), event.y())
            elif self.mode == "Box": 
                self.is_drawing = True
                self.start_point = event.pos()
                self.end_point = event.pos()
                self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing and self.mode == "Box":
            self.end_point = event.pos()
            self.update() 

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing and self.mode == "Box":
            self.is_drawing = False
            self.end_point = event.pos()
            self.update()
            
            rect = QRect(self.start_point, self.end_point).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self.box_selected.emit(rect.x(), rect.y(), rect.width(), rect.height())

    def draw_polygons(self, polygons_list):
        self.polygons = polygons_list 
        self.update() 

    def set_color(self, color):
        self.draw_color = color

    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Draw Polygons
        if self.cv_image is not None and self.polygons:
            pen = QPen(QColor(*self.draw_color), 3)
            painter.setPen(pen)
            
            # [수정 완료] QBrush 오류 해결
            c = QColor(*self.draw_color)
            c.setAlpha(50) 
            brush = QBrush(c) 
            painter.setBrush(brush)

            pm = self.pixmap()
            if pm:
                ww, wh = self.width(), self.height()
                pw, ph = pm.width(), pm.height()
                dx, dy = (ww - pw) / 2, (wh - ph) / 2
                
                orig_h, orig_w = self.cv_image.shape[:2]
                scale_x, scale_y = pw / orig_w, ph / orig_h

                for poly in self.polygons:
                    if not poly: continue
                    qt_points = []
                    for p in poly:
                        sx = p[0] * scale_x + dx
                        sy = p[1] * scale_y + dy
                        qt_points.append(QPoint(int(sx), int(sy)))
                    painter.drawPolygon(QPolygon(qt_points))
        
        # 2. Draw Drag Box
        if self.is_drawing and self.mode == "Box":
            pen = QPen(Qt.yellow, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(rect)

class ClassBadge(QLabel):
    def __init__(self, text, color="#35A7B6"):
        super().__init__(text)
        self.setStyleSheet(f"""
            background-color: {color}; 
            color: white; 
            border-radius: 10px; 
            padding: 5px 12px; 
            font-weight: bold;
            font-size: 12px;
            margin: 2px;
        """)
        self.adjustSize()

class VisualPromptTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Prompting (Point / Exemplar / Text)")
        self.resize(1600, 950)
        
        sam_model_path = "/home/park/work/sam3/sam3_models"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 모델 초기화
        self.sam_point = SAM3TrackerPointPredictor(sam_model_path, device=self.device)
        self.sam_text = SAM3Predictor(sam_model_path, device=self.device)
        self.matcher = DinoV3Matcher(device=self.device)

        self.class_memory = {} 
        self.query_examples = []
        self.test_polygons = [] 
        
        self.current_mode = "Point" 
        
        self.init_ui()
        self.query_img = None
        self.test_img = None

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # === Top Bar ===
        self.top_bar = QWidget()
        self.top_layout = QHBoxLayout(self.top_bar)
        self.top_layout.setAlignment(Qt.AlignLeft)
        self.top_bar.setFixedHeight(50)
        
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("font-weight: bold; margin-left: 10px;")
        self.top_layout.addWidget(mode_label)

        self.btn_group = QButtonGroup(self)
        
        self.btn_point = QPushButton("Point")
        self.btn_point.setCheckable(True)
        self.btn_point.setChecked(True)
        self.btn_point.clicked.connect(lambda: self.change_mode("Point"))
        
        self.btn_text = QPushButton("Text")
        self.btn_text.setCheckable(True)
        self.btn_text.clicked.connect(lambda: self.change_mode("Text"))
        
        self.btn_exemplar = QPushButton("Exemplar (Box)") 
        self.btn_exemplar.setCheckable(True)
        self.btn_exemplar.clicked.connect(lambda: self.change_mode("Box"))

        self.btn_group.addButton(self.btn_point)
        self.btn_group.addButton(self.btn_text)
        self.btn_group.addButton(self.btn_exemplar)

        btn_style = "QPushButton { padding: 5px 10px; } QPushButton:checked { background-color: #3F51B5; color: white; font-weight: bold; }"
        self.btn_point.setStyleSheet(btn_style)
        self.btn_text.setStyleSheet(btn_style)
        self.btn_exemplar.setStyleSheet(btn_style)

        self.top_layout.addWidget(self.btn_point)
        self.top_layout.addWidget(self.btn_text)
        self.top_layout.addWidget(self.btn_exemplar)
        
        # Text Input Area
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Enter text prompt...")
        self.txt_input.setFixedWidth(200)
        self.txt_input.setVisible(False)
        self.txt_input.returnPressed.connect(self.run_text_query)
        
        self.btn_run_text = QPushButton("Go")
        self.btn_run_text.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_run_text.setVisible(False)
        self.btn_run_text.clicked.connect(self.run_text_query)

        self.top_layout.addWidget(self.txt_input)
        self.top_layout.addWidget(self.btn_run_text)

        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        self.top_layout.addWidget(line)

        title_lbl = QLabel("Registered Classes:")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 14px; margin-left: 10px; margin-right: 10px;")
        self.top_layout.addWidget(title_lbl)
        main_layout.addWidget(self.top_bar)

        # === Splitter ===
        splitter = QSplitter(Qt.Horizontal)

        # Left: Query
        l_widget = QWidget()
        l_layout = QVBoxLayout(l_widget)
        btn_layout = QHBoxLayout()
        btn_q = QPushButton("1. Load Query Img")
        btn_q.clicked.connect(self.load_query)
        btn_clear = QPushButton("Clear Selection")
        btn_clear.clicked.connect(self.clear_query_selection)
        btn_clear.setStyleSheet("background-color: #FFCDD2; color: black;")
        btn_infer = QPushButton("2. Run Inference (Match)")
        btn_infer.clicked.connect(self.run_inference)
        btn_infer.setStyleSheet("background-color: #C8E6C9; color: black; font-weight: bold;")
        btn_layout.addWidget(btn_q)
        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(btn_infer)
        l_layout.addLayout(btn_layout)
        
        self.lbl_q = ImageLabel()
        self.lbl_q.setStyleSheet("border: 2px solid blue")
        self.lbl_q.set_color((0, 0, 255))
        self.lbl_q.clicked_pos.connect(self.on_query_click)
        self.lbl_q.box_selected.connect(self.on_query_box_selected) 
        self.lbl_q.right_clicked_pos.connect(self.on_query_right_click)
        l_layout.addWidget(self.lbl_q)
        splitter.addWidget(l_widget)

        # Right: Test
        r_widget = QWidget()
        r_layout = QVBoxLayout(r_widget)
        btn_t = QPushButton("Load Test Image")
        btn_t.clicked.connect(self.load_test)
        self.lbl_t = ImageLabel()
        self.lbl_t.setStyleSheet("border: 2px solid red")
        self.lbl_t.set_color((255, 0, 0))
        self.lbl_t.right_clicked_pos.connect(self.on_test_right_click)
        r_layout.addWidget(btn_t)
        r_layout.addWidget(self.lbl_t)
        splitter.addWidget(r_widget)
        
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

    def change_mode(self, mode):
        self.current_mode = mode
        label_mode = "Box" if mode == "Box" else "Point"
        self.lbl_q.set_mode(label_mode)
        
        print(f"Mode changed to: {mode} (Label Mode: {label_mode})")
        
        is_text = (mode == "Text")
        self.txt_input.setVisible(is_text)
        self.btn_run_text.setVisible(is_text)
        
        self.clear_query_selection()

    def update_class_bar(self):
        idx_start = -1
        for i in range(self.top_layout.count()):
            item = self.top_layout.itemAt(i)
            if item.widget() and isinstance(item.widget(), QLabel) and item.widget().text() == "Registered Classes:":
                idx_start = i
                break
        
        if idx_start != -1:
            while self.top_layout.count() > idx_start + 1:
                item = self.top_layout.takeAt(idx_start + 1)
                if item.widget(): item.widget().deleteLater()

        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#009688"]
        for i, cls_name in enumerate(self.class_memory.keys()):
            color = colors[i % len(colors)]
            badge = ClassBadge(cls_name, color)
            self.top_layout.addWidget(badge)

    def load_query(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Query Image")
        if path:
            self.query_img = cv2.imread(path)
            self.clear_query_selection()
            self.lbl_q.set_image(self.query_img)

    def load_test(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Test Image")
        if path:
            self.test_img = cv2.imread(path)
            self.test_polygons = [] 
            self.lbl_t.set_image(self.test_img)

    def clear_query_selection(self):
        self.query_examples = []
        self.lbl_q.draw_polygons([]) 
        print("Selection cleared.")

    def get_img_coord(self, lbl, img, wx, wy):
        if img is None or lbl.pixmap() is None: return None
        pm = lbl.pixmap()
        ww, wh = lbl.width(), lbl.height()
        pw, ph = pm.width(), pm.height()
        dx, dy = (ww - pw) / 2, (wh - ph) / 2
        if not (dx <= wx < dx+pw and dy <= wy < dy+ph): return None
        scale_x = pw / img.shape[1]
        scale_y = ph / img.shape[0]
        ix = int((wx - dx) / scale_x)
        iy = int((wy - dy) / scale_y)
        return ix, iy

    # === [Point Mode] 클릭 -> 특징 추출 -> 유사 객체 전체 검색 ===
    def on_query_click(self, x, y):
        if self.query_img is None: return        
        pt = self.get_img_coord(self.lbl_q, self.query_img, x, y)
        if not pt: return
        ix, iy = pt
        print(f"[Point Mode] Clicking at ({ix}, {iy})...")

        # 1. 클릭한 객체 Segmentation
        poly_np = self.sam_point.predict_by_point(self.query_img, ix, iy)
        if poly_np is None: return  
        poly = poly_np.tolist()
        
        # 2. 해당 객체와 유사한 모든 객체를 화면에서 찾아 등록 (DINO 사용)
        self.segment_all_similar_in_query(poly)

    # === [Box Mode] 드래그 -> 특징 추출 -> 유사 객체 전체 검색 ===
    def on_query_box_selected(self, x, y, w, h):
        if self.query_img is None: return
        pt1 = self.get_img_coord(self.lbl_q, self.query_img, x, y)
        pt2 = self.get_img_coord(self.lbl_q, self.query_img, x+w, y+h)
        if not pt1 or not pt2: return
        
        ix1, iy1 = pt1
        ix2, iy2 = pt2
        
        # 박스 중심점 계산 (SAM 추론용)
        cx = (ix1 + ix2) / 2
        cy = (iy1 + iy2) / 2
        
        print(f"[Exemplar Mode] Box Center: ({cx},{cy})...")
        
        # 1. 박스 안의 객체 Segment (Point Prompt 사용)
        poly_np = self.sam_point.predict_by_point(self.query_img, cx, cy)
        if poly_np is None: return
        poly = poly_np.tolist()
        
        # 2. 해당 객체 특징으로 화면 전체 검색
        self.segment_all_similar_in_query(poly)

    def run_text_query(self):
        if self.query_img is None: return
        text = self.txt_input.text().strip()
        if not text: return

        print(f"[Text Mode] Prompt: '{text}'")
        self.clear_query_selection() 

        polygons, labels, _ = self.sam_text.predict(self.query_img, text, conf=0.2)
        if not polygons:
            print("No objects found by text.")
            return

        print(f"Found {len(polygons)} objects by text.")
        for poly_np in polygons:
            self.add_example_directly(poly_np.tolist())
            
    def segment_all_similar_in_query(self, template_poly):
        # 1. Template의 Feature 추출
        pts = np.array(template_poly)
        x_min = max(0, int(pts[:,0].min()))
        x_max = min(self.query_img.shape[1], int(pts[:,0].max()))
        y_min = max(0, int(pts[:,1].min()))
        y_max = min(self.query_img.shape[0], int(pts[:,1].max()))
        
        crop_img = self.query_img[y_min:y_max, x_min:x_max]
        if crop_img.size == 0: return

        template_feature = self.matcher.get_embedding(crop_img) 
        
        print("Searching for similar objects in Query Image...")
        
        matched_points = self.matcher.find_matches_with_vector(template_feature, self.query_img, threshold=0.7)
        
        if not matched_points:
            print("No similar objects found. Adding self only.")
            self.add_example_directly(template_poly)
            return

        print(f"-> Found {len(matched_points)} similar objects in Query Image.")

        # 3. 자기 자신 및 찾은 객체들 등록
        self.add_example_directly(template_poly)

        for (mx, my) in matched_points:
            target_poly_np = self.sam_point.predict_by_point(self.query_img, mx, my)
            if target_poly_np is not None:
                new_poly = target_poly_np.tolist()
                
                # 중복 체크
                existing_polys = [ex['poly'] for ex in self.query_examples]
                if self.check_duplicate(new_poly, existing_polys, iou_threshold=0.7):
                    continue
                
                self.add_example_directly(new_poly)

    def add_example_directly(self, poly):
        """리스트 추가 및 화면 갱신"""
        pts = np.array(poly)
        x_min = max(0, int(pts[:,0].min()))
        x_max = min(self.query_img.shape[1], int(pts[:,0].max()))
        y_min = max(0, int(pts[:,1].min()))
        y_max = min(self.query_img.shape[0], int(pts[:,1].max()))
        
        crop_img = self.query_img[y_min:y_max, x_min:x_max]
        if crop_img.size == 0: return

        feature = self.matcher.get_embedding(crop_img)
        
        self.query_examples.append({
            'feature': feature,
            'poly': poly
        })
        
        all_polys = [ex['poly'] for ex in self.query_examples]
        self.lbl_q.draw_polygons(all_polys)
        QApplication.processEvents()

    def on_query_right_click(self, x, y):
        if self.query_img is None or not self.query_examples: return
        pt = self.get_img_coord(self.lbl_q, self.query_img, x, y)
        if not pt: return
        ix, iy = pt

        deleted = False
        for i in range(len(self.query_examples) - 1, -1, -1):
            poly = self.query_examples[i]['poly']
            pts = np.array(poly, dtype=np.int32)
            dist = cv2.pointPolygonTest(pts, (ix, iy), False)
            if dist >= 0:
                print(f"Deleting query example at index {i}")
                self.query_examples.pop(i)
                deleted = True
                break
        
        if deleted:
            all_polys = [ex['poly'] for ex in self.query_examples]
            self.lbl_q.draw_polygons(all_polys)

    def on_test_right_click(self, x, y):
        if self.test_img is None or not self.test_polygons: return
        pt = self.get_img_coord(self.lbl_t, self.test_img, x, y)
        if not pt: return
        ix, iy = pt

        deleted = False
        for i in range(len(self.test_polygons) - 1, -1, -1):
            poly = self.test_polygons[i]
            pts = np.array(poly, dtype=np.int32)
            dist = cv2.pointPolygonTest(pts, (ix, iy), False)
            if dist >= 0:
                print(f"Deleting test polygon at index {i}")
                self.test_polygons.pop(i)
                deleted = True
                break
        
        if deleted:
            self.lbl_t.draw_polygons(self.test_polygons)
    
    def check_duplicate(self, new_poly, existing_polys, iou_threshold=0.8):
        if not existing_polys: return False
        pts = np.array(new_poly)
        nx1, ny1, nx2, ny2 = pts[:,0].min(), pts[:,1].min(), pts[:,0].max(), pts[:,1].max()
        n_area = (nx2 - nx1) * (ny2 - ny1)
        for ex_poly in existing_polys:
            e_pts = np.array(ex_poly)
            ex1, ey1, ex2, ey2 = e_pts[:,0].min(), e_pts[:,1].min(), e_pts[:,0].max(), e_pts[:,1].max()
            ix1, iy1, ix2, iy2 = max(nx1, ex1), max(ny1, ey1), min(nx2, ex2), min(ny2, ey2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = n_area + (ex2-ex1)*(ey2-ey1) - inter
                if inter / union > iou_threshold: return True
        return False

    def run_inference(self):
        """Test Image에 대한 매칭 수행"""
        if not self.query_examples:
            QMessageBox.warning(self, "Warning", "Please select at least one object.")
            return
        if self.test_img is None:
            QMessageBox.warning(self, "Warning", "Please load a Test Image.")
            return

        print(f"Computing Features from {len(self.query_examples)} examples...")
        
        all_feats = [ex['feature'] for ex in self.query_examples]
        batch_features = torch.cat(all_feats, dim=0) 
        
        print("Matching with Batch Features (Max Pooling)...")
        matched_points = self.matcher.find_matches_with_vector(batch_features, self.test_img, threshold=0.65)
        
        if not matched_points:
            print("No matching objects found.")
            return

        self.test_polygons = [] 
        
        for (mx, my) in matched_points:
            target_poly_np = self.sam_point.predict_by_point(self.test_img, mx, my)
            if target_poly_np is not None:
                new_poly = target_poly_np.tolist()
                if self.check_duplicate(new_poly, self.test_polygons, iou_threshold=0.6):
                    continue
                self.test_polygons.append(new_poly)
            else:
                tx1, ty1 = max(0, mx-20), max(0, my-20)
                tx2, ty2 = min(self.test_img.shape[1], mx+20), min(self.test_img.shape[0], my+20)
                box_poly = [[tx1,ty1], [tx2,ty1], [tx2,ty2], [tx1,ty2]]
                if not self.check_duplicate(box_poly, self.test_polygons):
                    self.test_polygons.append(box_poly)

        self.lbl_t.draw_polygons(self.test_polygons)
        
        mean_feature_for_save = batch_features.mean(dim=0)
        
        detected_class = None
        max_sim = -1.0
        
        for cls_name, saved_feat in self.class_memory.items():
            sim = torch.nn.functional.cosine_similarity(mean_feature_for_save, saved_feat, dim=0).item()
            if sim > 0.85 and sim > max_sim:
                max_sim = sim
                detected_class = cls_name
        
        if detected_class:
            print(f"Recognized as: {detected_class}")
        else:
            text, ok = QInputDialog.getText(self, "New Class", 
                                          f"Found {len(self.test_polygons)} objects.\nEnter Class Name:")
            if ok and text:
                self.class_memory[text] = mean_feature_for_save
                self.update_class_bar()
                print(f"Registered new class: {text}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = VisualPromptTool()
    ex.show()
    sys.exit(app.exec_())