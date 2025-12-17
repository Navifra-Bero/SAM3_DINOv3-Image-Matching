import os, cv2, tempfile
import numpy as np

from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import prepare_masks_for_visualization

def extract_video_to_frames(video_path, max_frames=None, stride=1, jpg_quality=95):
    temp_dir = tempfile.mkdtemp(prefix="sam3_frames_")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    saved, idx = 0, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            out = os.path.join(temp_dir, f"{saved:05d}.jpg")
            cv2.imwrite(out, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        idx += 1
    cap.release()
    return temp_dir, saved, fps / max(1, stride), (W, H)

def propagate_in_video(predictor, session_id):
    for resp in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session_id)):
        yield resp["frame_index"], resp["outputs"]

def _color_for_id(obj_id: int):
    rng = np.random.default_rng(obj_id * 9973)
    c = rng.integers(40, 255, size=(3,), dtype=np.uint8)
    return (int(c[0]), int(c[1]), int(c[2]))

def overlay_formatted(frame_bgr, formatted_frame, id_to_label, alpha=0.45):
    """
    formatted_frame: prepare_masks_for_visualization({frame_idx: outputs})[frame_idx]
    버전마다 구조가 달라서, 가능한 형태를 최대한 폭넓게 처리.
    """
    H, W = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    # 케이스1) {obj_id: {...}} 형태 (가장 흔함)
    if isinstance(formatted_frame, dict):
        # obj_id 후보 추출
        obj_ids = []
        for k in formatted_frame.keys():
            if isinstance(k, int):
                obj_ids.append(k)
            elif isinstance(k, str) and k.isdigit():
                obj_ids.append(int(k))

        if obj_ids:
            obj_ids = sorted(set(obj_ids))
            for obj_id in obj_ids:
                info = formatted_frame.get(obj_id, formatted_frame.get(str(obj_id)))
                if not isinstance(info, dict):
                    continue

                # 마스크 후보 키들
                mask = None
                for mk in ["mask", "masks", "segmentation", "binary_mask", "pred_mask"]:
                    if mk in info:
                        mask = info[mk]
                        break
                if mask is None:
                    continue

                m = np.asarray(mask)
                if m.ndim == 3:
                    m = np.any(m.astype(bool), axis=0)
                else:
                    m = m.astype(bool)

                if m.shape[:2] != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

                color = _color_for_id(obj_id)

                # color overlay
                colored = np.zeros_like(out, dtype=np.uint8)
                colored[m] = color
                out = cv2.addWeighted(out, 1.0, colored, alpha, 0)

                # bbox + label
                ys, xs = np.where(m)
                if len(xs) > 0:
                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                    label = id_to_label.get(obj_id, f"id:{obj_id}")
                    txt = f"{label} (id:{obj_id})"
                    cv2.putText(out, txt, (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            return out

    # 못 파싱하면 원본 반환
    return out

def run_multi_class_text_tracking(
    video_path,
    prompts,                 # 예: ["wheel", "door handle", ...]
    gpus_to_use=(0,),
    stride=1,
    max_frames=None,
    out_mp4_path="sam3_multi_class_vis.mp4",
    show_window=False,       # True면 OpenCV 창(환경에 따라 segfault 가능)
):
    frame_dir, n_frames, fps, (W, H) = extract_video_to_frames(video_path, max_frames=max_frames, stride=stride)
    print("frames:", n_frames, "fps:", fps, "dir:", frame_dir)

    predictor = build_sam3_video_predictor(gpus_to_use=list(gpus_to_use))
    session_id = predictor.handle_request(dict(type="start_session", resource_path=frame_dir))["session_id"]
    print("session:", session_id)

    # ✅ 멀티 클래스: prompt마다 obj_id 고정
    id_to_label = {}
    for obj_id, text in enumerate(prompts, start=1):
        predictor.handle_request(dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=text,
            obj_id=obj_id,
        ))
        id_to_label[obj_id] = text
        print("add:", obj_id, text)

    # writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4_path, fourcc, fps, (W, H))

    for frame_idx, outputs in propagate_in_video(predictor, session_id):
        fp = os.path.join(frame_dir, f"{frame_idx:05d}.jpg")
        frame_bgr = cv2.imread(fp)
        if frame_bgr is None:
            continue

        formatted_map = prepare_masks_for_visualization({frame_idx: outputs})
        formatted_frame = formatted_map.get(frame_idx, {})

        vis = overlay_formatted(frame_bgr, formatted_frame, id_to_label, alpha=0.45)

        # 상단 상태 텍스트
        cv2.putText(vis, f"frame {frame_idx}/{n_frames-1}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        vw.write(vis)

        if show_window:
            cv2.imshow("SAM3 multi-class text tracking", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        if frame_idx % 50 == 0:
            print("processing", frame_idx)

    vw.release()
    if show_window:
        cv2.destroyAllWindows()
    print("saved:", out_mp4_path)

run_multi_class_text_tracking(
    video_path="test2.mp4",
    prompts=["wheel", "truck", "door handle"],
    out_mp4_path="ai_truck.mp4",
    show_window=False,   # 창 띄우면 환경에 따라 segfault 날 수 있어서 기본 False 추천
)
