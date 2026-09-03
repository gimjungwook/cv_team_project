"""
Telea Inpainting GIF 자동 생성

발표용: 알고리즘 진행 과정을 GIF로 저장

사용법:
    python demo_telea_gif.py
"""

import heapq
from typing import List, Tuple
import cv2
import numpy as np

# 픽셀 상태 상수
KNOWN = 0
BAND = 1
UNKNOWN = 2

# 시각화 색상 (BGR)
COLOR_KNOWN = (80, 80, 80)
COLOR_BAND = (0, 255, 255)      # 노란색
COLOR_UNKNOWN = (50, 50, 200)   # 빨간색
COLOR_FILLED = (0, 220, 0)      # 초록색


def create_test_image(size: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """테스트 이미지 생성"""
    image = np.zeros((size, size, 3), dtype=np.uint8)

    # 그라데이션 배경
    for i in range(size):
        for j in range(size):
            image[i, j] = [
                int(180 + 50 * np.sin(i * 0.08)),
                int(150 + 50 * np.sin(j * 0.08)),
                int(120 + 50 * np.cos((i + j) * 0.05))
            ]

    # 격자 패턴
    for i in range(0, size, 25):
        cv2.line(image, (i, 0), (i, size), (255, 255, 255), 1)
        cv2.line(image, (0, i), (size, i), (255, 255, 255), 1)

    # 원형 마스크 (중앙)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 5, 255, -1)

    return image, mask


def generate_telea_gif(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str = "telea_animation.gif",
    num_frames: int = 50,
    radius: int = 5
):
    """Telea 알고리즘 GIF 생성"""

    print(f"GIF 생성 시작...")
    print(f"  이미지 크기: {image.shape[1]}x{image.shape[0]}")
    print(f"  마스크 픽셀: {np.sum(mask > 0)}")
    print(f"  프레임 수: {num_frames}")

    h, w = image.shape[:2]
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

    # 전체 픽셀 수 계산
    total_pixels = np.sum(binary_mask > 0)
    pixels_per_frame = max(1, total_pixels // num_frames)

    frames = []
    epsilon = 1e-6

    def make_frame(recently_filled=None):
        """프레임 생성"""
        vis = np.clip(result, 0, 255).astype(np.uint8).copy()
        overlay = np.zeros_like(vis)

        # 상태별 색상
        overlay[flag == UNKNOWN] = COLOR_UNKNOWN
        overlay[flag == BAND] = COLOR_BAND

        # 방금 채워진 픽셀
        if recently_filled:
            for fy, fx in recently_filled:
                overlay[fy, fx] = COLOR_FILLED

        # 마스크 영역만 오버레이 적용
        mask_area = binary_mask > 0
        vis[mask_area] = cv2.addWeighted(
            vis[mask_area], 0.4,
            overlay[mask_area], 0.6, 0
        )

        # 정보 텍스트
        remaining = np.sum(flag == UNKNOWN) + np.sum(flag == BAND)
        progress = int(100 * (1 - remaining / total_pixels))

        # 상단 바
        cv2.rectangle(vis, (0, 0), (w, 35), (0, 0, 0), -1)
        cv2.putText(vis, f"Telea Inpainting: {progress}%", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 범례
        cv2.rectangle(vis, (0, h - 25), (w, h), (0, 0, 0), -1)
        cv2.rectangle(vis, (5, h - 20), (15, h - 10), COLOR_BAND, -1)
        cv2.putText(vis, "Boundary", (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.rectangle(vis, (90, h - 20), (100, h - 10), COLOR_UNKNOWN, -1)
        cv2.putText(vis, "Unknown", (105, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.rectangle(vis, (170, h - 20), (180, h - 10), COLOR_FILLED, -1)
        cv2.putText(vis, "Filling", (185, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        return vis

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

    # 초기 프레임
    frames.append(make_frame())

    # Fast Marching
    pixel_count = 0
    recently_filled = []

    while heap:
        d, y, x = heapq.heappop(heap)

        if flag[y, x] == KNOWN:
            continue

        inpaint_pixel(y, x)
        flag[y, x] = KNOWN
        recently_filled.append((y, x))
        pixel_count += 1

        # 이웃 업데이트
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if flag[ny, nx] == UNKNOWN:
                    flag[ny, nx] = BAND
                    dist[ny, nx] = compute_distance(ny, nx)
                    heapq.heappush(heap, (dist[ny, nx], ny, nx))

        # 프레임 저장
        if pixel_count % pixels_per_frame == 0:
            frames.append(make_frame(recently_filled))
            recently_filled = []
            print(f"  프레임 {len(frames)}/{num_frames} 생성 중...")

    # 최종 프레임 (완료 상태)
    final = np.clip(result, 0, 255).astype(np.uint8)
    cv2.rectangle(final, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.putText(final, "Complete!", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 마지막 프레임 여러 번 추가 (잠시 멈춤 효과)
    for _ in range(5):
        frames.append(final)

    # GIF 저장
    print(f"\nGIF 저장 중... ({len(frames)} 프레임)")

    import imageio
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(output_path, rgb_frames, duration=0.15, loop=0)

    print(f"저장 완료: {output_path}")

    # 최종 결과도 저장
    cv2.imwrite("telea_result.png", final)
    print(f"최종 결과: telea_result.png")

    return output_path


def main():
    print("=" * 50)
    print("  Telea Inpainting GIF Generator")
    print("=" * 50)

    # 테스트 이미지 생성
    image, mask = create_test_image(250)

    # GIF 생성
    generate_telea_gif(image, mask, "telea_animation.gif", num_frames=40)

    print("\n" + "=" * 50)
    print("  완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
