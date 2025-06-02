import cv2
import numpy as np

from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
from utils import frame_to_bgr_image  # 컬러/깊이 프레임을 OpenCV 이미지로 변환해 주는 유틸
from ultralytics import YOLO

ESC_KEY     = 27
MIN_DEPTH   = 20      # mm
MAX_DEPTH   = 10000   # mm
WINDOW_W, WINDOW_H = 1280, 720

model = YOLO("yolov8n.pt")

def main():
    pipeline = Pipeline()
    config   = Config()

    # ——— StreamProfile 객체를 직접 선택해서 넘겨주기 ———
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    # 원하는 컬러 프로파일 찾기 (1280×960, RGB)
    color_profile = None
    for i in range(len(color_profiles)):
        p = color_profiles[i]
        if p.get_width() == 1280 and p.get_height() == 960 and p.get_format() == OBFormat.RGB:
            color_profile = p
            print("find color frame")
            break
    if color_profile is None:
        color_profile = color_profiles.get_default_video_stream_profile()

    print(f"Selected COLOR: {color_profile.get_width()}x{color_profile.get_height()} {color_profile.get_format()} {color_profile.get_fps()}fps")

    # 원하는 깊이 프로파일 찾기 (640×576, Y16)
    depth_profile = None
    for i in range(len(depth_profiles)):
        p = depth_profiles[i]
        if p.get_width() == 640 and p.get_height() == 576 and p.get_format() == OBFormat.Y16:
            depth_profile = p
            print("find depth frame")
            break
    if depth_profile is None:
        depth_profile = depth_profiles.get_default_video_stream_profile()

    print(f"Selected DEPTH: {depth_profile.get_width()}x{depth_profile.get_height()} {depth_profile.get_format()} {depth_profile.get_fps()}fps")

    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

    pipeline.start(config)
    print("Pipeline started. Press 'q' or ESC to exit.")

    cv2.namedWindow("Simple Aligned Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Simple Aligned Viewer", WINDOW_W, WINDOW_H)

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if not frames:
                continue

            # 컬러 프레임 → BGR
            c_frame = frames.get_color_frame()
            if not c_frame:
                continue
            color_img_original = frame_to_bgr_image(c_frame)
            
            # 깊이 프레임 → numpy + 거리(mm)
            d_frame = frames.get_depth_frame()
            if not d_frame or d_frame.get_format() != OBFormat.Y16:
                continue

            w_d, h_d = d_frame.get_width(), d_frame.get_height()  # 640x576
            scale = d_frame.get_depth_scale()
            raw = np.frombuffer(d_frame.get_data(), dtype=np.uint16)
            depth_m = raw.reshape((h_d, w_d)).astype(np.float32) * scale
            depth_m = np.where((depth_m >= MIN_DEPTH/1000.0) & (depth_m <= MAX_DEPTH/1000.0), depth_m, 0)

            # 컬러 이미지를 깊이 센서 해상도로 직접 리사이즈 (간단한 방법)
            color_img_resized = cv2.resize(color_img_original, (w_d, h_d))

            # YOLO 추론을 리사이즈된 이미지에서 수행
            results = model(color_img_resized)[0]

            # 시각화용 깊이 컬러맵
            depth_vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # 결과 처리
            for box in results.boxes:
                # 바운딩 박스 좌표 (이미 깊이 센서 해상도에 맞춰져 있음)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = box.conf[0].item()

                # 바운딩 박스 중심점
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 깊이 측정 (1:1 매핑)
                if 0 <= cx < w_d and 0 <= cy < h_d:
                    dist_m = depth_m[cy, cx]
                    dist_mm = dist_m * 1000.0
                else:
                    dist_mm = 0.0

                # 리사이즈된 컬러 이미지에 표시
                text = f"{label} {conf:.2f}"
                cv2.rectangle(color_img_resized, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(color_img_resized, text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                cv2.putText(color_img_resized, text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
                cv2.circle(color_img_resized, (cx, cy), 3, (0,0,255), -1)

                # 거리 정보 표시
                dist_text = f"{dist_mm:.1f}mm"
                cv2.putText(color_img_resized, dist_text, (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                cv2.putText(color_img_resized, dist_text, (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1)

                # 깊이 이미지에도 동일한 좌표로 표시
                cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(depth_vis, dist_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                cv2.putText(depth_vis, dist_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
                cv2.circle(depth_vis, (cx, cy), 3, (0,0,255), -1)

            # 좌우로 컬러/깊이 창 병합하여 표시
            left = cv2.resize(color_img_resized, (WINDOW_W//2, WINDOW_H))
            right = cv2.resize(depth_vis, (WINDOW_W//2, WINDOW_H))
            combined = np.hstack((left, right))
            
            # 정보 텍스트 추가
            info_text = f"Aligned Resolution: {w_d}x{h_d} (Color resized to match Depth)"
            cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(combined, "1:1 Pixel Mapping - No Offset", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow("Simple Aligned Viewer", combined)

            if cv2.waitKey(1) in [ord('q'), ESC_KEY]:
                break

    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        print("Pipeline stopped and all windows closed.")


if __name__ == "__main__":
    main()

