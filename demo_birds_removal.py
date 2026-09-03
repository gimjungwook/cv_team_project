"""
Birds Removal GIF Generator

birds.jpeg에서 YOLO로 새를 탐지하고 Telea로 제거하는 과정을 GIF로 생성

사용법:
    python demo_birds_removal.py
"""

import heapq
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

from core.detect import load_default_detector

# 픽셀 상태 상수
KNOWN = 0
BAND = 1
UNKNOWN = 2

# 시각화 색상 (BGR)
COLOR_BAND = (0, 255, 255)      # 노란색 - 경계
COLOR_UNKNOWN = (0, 0, 200)     # 빨간색 - 미복원
COLOR_FILLED = (0, 220, 0)      # 초록색 - 복원중


def generate_removal_gif(
    image: np.ndarray,
    mask: np.ndarray,
    detections: list,
    output_path: str = "birds_removal.gif",
    num_frames: int = 60,
    radius: int = 5
):
    """객체 제거 GIF 생성"""

    print(f"\nGIF 생성 시작...")
    h, w = image.shape[:2]
    print(f"  이미지 크기: {w}x{h}")
    print(f"  마스크 픽셀: {np.sum(mask > 0)}")

    result = image.copy().astype(np.float32)
    binary_mask = (mask > 0).astype(np.uint8)

    # 상태/거리 맵
    flag = np.zeros((h, w), dtype=np.uint8)
    flag[binary_mask > 0] = UNKNOWN

    dist = np.full((h, w), np.inf, dtype=np.float32)
    dist[flag == KNOWN] = 0

    # 경계 초기화
    heap: List[Tuple[float, int, int]] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        for x in range(w):
            if flag[y, x] == UNKNOWN:
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if flag[ny, nx] == KNOWN:
                            flag[y, x] = BAND
                            dist[y, x] = 1.0
                            heapq.heappush(heap, (1.0, y, x))
                            break

    total_pixels = np.sum(binary_mask > 0)
    pixels_per_frame = max(1, total_pixels // num_frames)

    frames = []
    epsilon = 1e-6

    # 디스플레이용 스케일 (원본이 크면 축소)
    max_display = 800
    scale = min(1.0, max_display / max(h, w))
    display_h, display_w = int(h * scale), int(w * scale)

    def make_frame(recently_filled=None, phase="inpainting"):
        """프레임 생성"""
        vis = np.clip(result, 0, 255).astype(np.uint8).copy()

        if phase == "detection":
            # 탐지 결과 표시
            for det in detections:
                x, y, bw, bh = det.bbox
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
                label = f"{det.label} {det.confidence*100:.0f}%"
                cv2.putText(vis, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif phase == "mask":
            # 마스크 오버레이 (빨간 테두리로 표시)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
            # 반투명 빨간색으로 마스크 영역 표시
            overlay = vis.copy()
            overlay[binary_mask > 0] = [0, 0, 200]
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        elif phase == "inpainting":
            # 인페인팅 진행: 실제 결과 보여주고, 미복원 영역만 테두리로 표시
            # 아직 안 채워진 영역 (UNKNOWN + BAND)의 테두리만 표시
            remaining_mask = np.zeros((h, w), dtype=np.uint8)
            remaining_mask[flag == UNKNOWN] = 255
            remaining_mask[flag == BAND] = 255

            if np.any(remaining_mask):
                contours, _ = cv2.findContours(remaining_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 200, 255), 2)  # 주황색 테두리

        # 리사이즈
        vis_resized = cv2.resize(vis, (display_w, display_h))

        # 정보 바
        info_h = 50
        frame = np.zeros((display_h + info_h, display_w, 3), dtype=np.uint8)
        frame[info_h:, :] = vis_resized

        # 상단 정보
        remaining = np.sum(flag == UNKNOWN) + np.sum(flag == BAND)
        progress = int(100 * (1 - remaining / max(total_pixels, 1)))

        if phase == "detection":
            text = f"Step 1: YOLO Detection - {len(detections)} birds found"
            color = (0, 255, 0)
        elif phase == "mask":
            text = f"Step 2: GrabCut Mask Generated"
            color = (0, 100, 255)
        elif phase == "complete":
            text = "Step 3: Telea Inpainting - Complete!"
            color = (0, 255, 0)
        else:
            text = f"Step 3: Telea Inpainting - {progress}%"
            color = (255, 255, 0)

        cv2.putText(frame, text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame

    def inpaint_pixel(y, x):
        """픽셀 복원"""
        total_weight = 0.0
        weighted_sum = np.zeros(3, dtype=np.float32)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if flag[ny, nx] != KNOWN:
                    continue

                pixel_dist = np.sqrt(dy ** 2 + dx ** 2)
                if pixel_dist > radius or pixel_dist < epsilon:
                    continue

                weight = 1.0 / (pixel_dist ** 2 + epsilon)
                total_weight += weight
                weighted_sum += weight * result[ny, nx]

        if total_weight > epsilon:
            result[y, x] = weighted_sum / total_weight

    def compute_distance(y, x):
        """거리 계산"""
        dx_min = np.inf
        if x > 0: dx_min = min(dx_min, dist[y, x - 1])
        if x < w - 1: dx_min = min(dx_min, dist[y, x + 1])

        dy_min = np.inf
        if y > 0: dy_min = min(dy_min, dist[y - 1, x])
        if y < h - 1: dy_min = min(dy_min, dist[y + 1, x])

        if np.isinf(dx_min) and np.isinf(dy_min): return np.inf
        if np.isinf(dx_min): return dy_min + 1.0
        if np.isinf(dy_min): return dx_min + 1.0

        discriminant = 2 - (dx_min - dy_min) ** 2
        if discriminant < 0: return min(dx_min, dy_min) + 1.0
        return (dx_min + dy_min + np.sqrt(discriminant)) / 2

    # Phase 1: Detection 프레임 (여러 번 추가해서 잠시 보여줌)
    print("  Phase 1: Detection 프레임 생성...")
    for _ in range(8):
        frames.append(make_frame(phase="detection"))

    # Phase 2: Mask 프레임
    print("  Phase 2: Mask 프레임 생성...")
    for _ in range(8):
        frames.append(make_frame(phase="mask"))

    # Phase 3: Inpainting
    print("  Phase 3: Inpainting 프레임 생성...")
    frames.append(make_frame(phase="inpainting"))

    pixel_count = 0
    recently_filled = []
    frame_count = 0

    while heap:
        d, y, x = heapq.heappop(heap)

        if flag[y, x] == KNOWN:
            continue

        inpaint_pixel(y, x)
        flag[y, x] = KNOWN
        recently_filled.append((y, x))
        pixel_count += 1

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if flag[ny, nx] == UNKNOWN:
                    flag[ny, nx] = BAND
                    dist[ny, nx] = compute_distance(ny, nx)
                    heapq.heappush(heap, (dist[ny, nx], ny, nx))

        if pixel_count % pixels_per_frame == 0:
            frames.append(make_frame(recently_filled))
            recently_filled = []
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"    프레임 {frame_count} 생성...")

    # 완료 프레임 (더 길게 보여주기)
    print("  Complete 프레임 생성...")
    for _ in range(25):
        frames.append(make_frame(phase="complete"))

    # GIF 저장
    print(f"\nGIF 저장 중... ({len(frames)} 프레임)")

    import imageio
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(output_path, rgb_frames, duration=0.12, loop=0)

    print(f"저장 완료: {output_path}")

    # 최종 결과도 저장
    final = np.clip(result, 0, 255).astype(np.uint8)
    cv2.imwrite("birds_removed.png", final)
    print(f"최종 결과: birds_removed.png")

    return output_path


class ManualDetection:
    """수동 탐지 결과"""
    def __init__(self, label, confidence, bbox):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # (x, y, w, h)


def main():
    print("=" * 60)
    print("  Birds Removal GIF Generator")
    print("  YOLO Detection + Telea Inpainting")
    print("=" * 60)

    # 이미지 로드
    image_path = "birds.jpeg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    print(f"\n원본 이미지: {image_path}")
    print(f"원본 크기: {image.shape[1]}x{image.shape[0]}")

    # 이미지 축소 (처리 속도를 위해)
    max_size = 800
    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"축소 크기: {new_w}x{new_h} (scale: {scale:.2f})")

    # YOLO 탐지 시도
    print("\n[Step 1] YOLO 탐지 중...")
    try:
        detector = load_default_detector(Path("models"))
        # confidence를 낮춰서 시도
        detector.conf_threshold = 0.3
        detections = detector.detect(image)
        bird_detections = [d for d in detections if d.label == "bird"]
    except Exception as e:
        print(f"YOLO 탐지 오류: {e}")
        bird_detections = []
        detections = []

    print(f"탐지된 새: {len(bird_detections)}마리")

    # YOLO가 탐지 못하면 직접 선택
    if not bird_detections:
        print("YOLO 탐지 실패 - 직접 선택 모드")
        print("\n[안내] 새를 드래그로 선택하세요")
        print("  - Enter/Space: 선택 확정")
        print("  - c: 취소 (선택 종료)")
        print("  - 여러 마리 선택 가능\n")

        bird_detections = []
        select_img = image.copy()

        while True:
            # 이미 선택된 영역 표시
            display = select_img.copy()
            for i, det in enumerate(bird_detections):
                x, y, bw, bh = det.bbox
                cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(display, f"Bird {i+1}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            roi = cv2.selectROI(f"Select Bird (selected: {len(bird_detections)})",
                               display, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(f"Select Bird (selected: {len(bird_detections)})")

            x, y, w, h = roi
            if w > 0 and h > 0:
                bird_detections.append(ManualDetection("bird", 0.95, (x, y, w, h)))
                print(f"  Bird {len(bird_detections)} 추가: ({x}, {y}, {w}, {h})")
            else:
                print(f"\n선택 완료: {len(bird_detections)}마리")
                break

        if not bird_detections:
            print("선택된 새가 없습니다. 종료합니다.")
            return

    # GrabCut으로 마스크 생성
    print("\n[Step 2] GrabCut 마스크 생성 중...")
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for det in bird_detections:
        bx, by, bw, bh = det.bbox

        # 경계 보정
        x1 = max(bx, 0)
        y1 = max(by, 0)
        x2 = min(bx + bw, w - 1)
        y2 = min(by + bh, h - 1)

        if x2 - x1 < 10 or y2 - y1 < 10:
            mask[y1:y2, x1:x2] = 255
            continue

        try:
            gc_mask = np.zeros((h, w), dtype=np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(image, gc_mask, (x1, y1, x2 - x1, y2 - y1),
                        bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            if np.any(fg):
                mask = cv2.bitwise_or(mask, fg)
            else:
                mask[y1:y2, x1:x2] = 255
        except cv2.error:
            mask[y1:y2, x1:x2] = 255

    # 마스크 팽창
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    print(f"마스크 픽셀 수: {np.sum(mask > 0)}")

    # GIF 생성
    print("\n[Step 3] Telea Inpainting GIF 생성...")
    generate_removal_gif(
        image, mask, bird_detections,
        output_path="birds_removal.gif",
        num_frames=50,
        radius=5
    )

    print("\n" + "=" * 60)
    print("  완료!")
    print("  - birds_removal.gif : 전체 과정 애니메이션")
    print("  - birds_removed.png : 최종 결과")
    print("=" * 60)


if __name__ == "__main__":
    main()
