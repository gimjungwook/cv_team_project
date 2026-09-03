"""
Telea Inpainting 알고리즘 시각화 데모

발표용: 알고리즘이 경계에서 내부로 진행하는 과정을 시각적으로 보여줌

사용법:
    python demo_telea_visualization.py [이미지경로]
    python demo_telea_visualization.py  # 기본 테스트 이미지 생성

키보드:
    Space: 다음 단계 (수동 모드)
    Enter: 자동 재생 토글
    r: 리셋
    s: GIF 저장
    q: 종료
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
import heapq

import cv2
import numpy as np

# 픽셀 상태 상수
KNOWN = 0      # 알려진 픽셀 (원본)
BAND = 1       # 경계 픽셀 (처리 대기)
UNKNOWN = 2    # 마스크 내부 (미복원)

# 상태별 시각화 색상 (BGR)
COLOR_KNOWN = (50, 50, 50)       # 어두운 회색 - 원본
COLOR_BAND = (0, 255, 255)       # 노란색 - 현재 경계
COLOR_UNKNOWN = (0, 0, 200)      # 빨간색 - 미복원
COLOR_JUST_FILLED = (0, 255, 0)  # 초록색 - 방금 복원됨


class TeleaVisualizer:
    """Telea 알고리즘 시각화 클래스"""

    def __init__(self, image: np.ndarray, mask: np.ndarray, radius: int = 5):
        self.original = image.copy()
        self.result = image.copy().astype(np.float32)
        self.mask = (mask > 0).astype(np.uint8)
        self.radius = radius
        self.epsilon = 1e-6

        h, w = image.shape[:2]
        self.h, self.w = h, w

        # 상태 맵
        self.flag = np.zeros((h, w), dtype=np.uint8)
        self.flag[self.mask > 0] = UNKNOWN

        # 거리 맵
        self.dist = np.full((h, w), np.inf, dtype=np.float32)
        self.dist[self.flag == KNOWN] = 0

        # Priority Queue
        self.heap: List[Tuple[float, int, int]] = []

        # 시각화용
        self.step = 0
        self.frames: List[np.ndarray] = []
        self.recently_filled: List[Tuple[int, int]] = []

        # 경계 초기화
        self._init_boundary()

    def _init_boundary(self) -> None:
        """경계 픽셀 초기화"""
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y in range(self.h):
            for x in range(self.w):
                if self.flag[y, x] == UNKNOWN:
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.h and 0 <= nx < self.w:
                            if self.flag[ny, nx] == KNOWN:
                                self.flag[y, x] = BAND
                                self.dist[y, x] = 1.0
                                heapq.heappush(self.heap, (1.0, y, x))
                                break

    def step_forward(self, num_pixels: int = 1) -> bool:
        """지정된 픽셀 수만큼 진행"""
        self.recently_filled = []

        for _ in range(num_pixels):
            if not self.heap:
                return False

            d, y, x = heapq.heappop(self.heap)

            if self.flag[y, x] == KNOWN:
                continue

            # 픽셀 복원
            self._inpaint_pixel(y, x)
            self.flag[y, x] = KNOWN
            self.recently_filled.append((y, x))
            self.step += 1

            # 이웃 업데이트
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.h and 0 <= nx < self.w:
                    if self.flag[ny, nx] == UNKNOWN:
                        self.flag[ny, nx] = BAND
                        new_dist = self._compute_distance(ny, nx)
                        self.dist[ny, nx] = new_dist
                        heapq.heappush(self.heap, (new_dist, ny, nx))

        return True

    def _compute_distance(self, y: int, x: int) -> float:
        """Eikonal 방정식으로 거리 계산"""
        dx_min = np.inf
        if x > 0:
            dx_min = min(dx_min, self.dist[y, x - 1])
        if x < self.w - 1:
            dx_min = min(dx_min, self.dist[y, x + 1])

        dy_min = np.inf
        if y > 0:
            dy_min = min(dy_min, self.dist[y - 1, x])
        if y < self.h - 1:
            dy_min = min(dy_min, self.dist[y + 1, x])

        if np.isinf(dx_min) and np.isinf(dy_min):
            return np.inf
        if np.isinf(dx_min):
            return dy_min + 1.0
        if np.isinf(dy_min):
            return dx_min + 1.0

        a = 2.0
        b = -2.0 * (dx_min + dy_min)
        c = dx_min ** 2 + dy_min ** 2 - 1.0
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return min(dx_min, dy_min) + 1.0
        return (-b + np.sqrt(discriminant)) / (2 * a)

    def _inpaint_pixel(self, y: int, x: int) -> None:
        """단일 픽셀 복원"""
        r = self.radius
        total_weight = 0.0
        weighted_sum = np.zeros(3, dtype=np.float32)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = y + dy, x + dx

                if not (0 <= ny < self.h and 0 <= nx < self.w):
                    continue
                if self.flag[ny, nx] != KNOWN:
                    continue

                pixel_dist = np.sqrt(dy ** 2 + dx ** 2)
                if pixel_dist > r or pixel_dist < self.epsilon:
                    continue

                # 거리 가중치
                w_dist = 1.0 / (pixel_dist ** 2 + self.epsilon)

                # 방향 가중치 (간소화)
                w_dir = 1.0

                # 레벨셋 가중치
                if not np.isinf(self.dist[ny, nx]):
                    level_diff = abs(self.dist[ny, nx] - self.dist[y, x])
                    w_level = 1.0 / (1.0 + level_diff)
                else:
                    w_level = 1.0

                weight = w_dist * w_dir * w_level
                total_weight += weight
                weighted_sum += weight * self.result[ny, nx]

        if total_weight > self.epsilon:
            self.result[y, x] = weighted_sum / total_weight

    def get_visualization(self, show_state: bool = True) -> np.ndarray:
        """현재 상태 시각화 이미지 생성"""
        # 결과 이미지 복사
        vis = np.clip(self.result, 0, 255).astype(np.uint8).copy()

        if show_state:
            # 상태 오버레이 생성
            overlay = vis.copy()

            # UNKNOWN 영역 (빨간색)
            unknown_mask = (self.flag == UNKNOWN)
            overlay[unknown_mask] = COLOR_UNKNOWN

            # BAND 영역 (노란색)
            band_mask = (self.flag == BAND)
            overlay[band_mask] = COLOR_BAND

            # 방금 채워진 픽셀 (초록색)
            for y, x in self.recently_filled:
                overlay[y, x] = COLOR_JUST_FILLED

            # 블렌딩
            alpha = 0.6
            vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

        return vis

    def get_state_map(self) -> np.ndarray:
        """상태 맵 시각화"""
        state_vis = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # KNOWN (회색)
        state_vis[self.flag == KNOWN] = COLOR_KNOWN
        # BAND (노란색)
        state_vis[self.flag == BAND] = COLOR_BAND
        # UNKNOWN (빨간색)
        state_vis[self.flag == UNKNOWN] = COLOR_UNKNOWN
        # 방금 채워진 (초록색)
        for y, x in self.recently_filled:
            state_vis[y, x] = COLOR_JUST_FILLED

        return state_vis

    def is_done(self) -> bool:
        """완료 여부"""
        return len(self.heap) == 0

    def get_final_result(self) -> np.ndarray:
        """최종 결과"""
        return np.clip(self.result, 0, 255).astype(np.uint8)

    def save_frame(self) -> None:
        """현재 프레임 저장"""
        self.frames.append(self.get_visualization().copy())


def create_test_image_and_mask(size: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """테스트용 이미지와 마스크 생성"""
    # 그라데이션 배경
    image = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            image[i, j] = [
                int(255 * i / size),  # B
                int(255 * j / size),  # G
                int(128 + 127 * np.sin(i * 0.1) * np.cos(j * 0.1))  # R
            ]

    # 원형 마스크
    mask = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = size // 6
    cv2.circle(mask, (center, center), radius, 255, -1)

    return image, mask


def main():
    # 이미지 로드 또는 생성
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            sys.exit(1)

        # 마스크 선택
        print("마스크 영역을 선택하세요 (드래그 후 Enter)")
        roi = cv2.selectROI("Select Mask Area", image, fromCenter=False)
        cv2.destroyWindow("Select Mask Area")

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x, y, w, h = roi
        if w > 0 and h > 0:
            mask[y:y+h, x:x+w] = 255
        else:
            # 중앙에 원형 마스크
            center = (image.shape[1] // 2, image.shape[0] // 2)
            radius = min(image.shape[:2]) // 8
            cv2.circle(mask, center, radius, 255, -1)
    else:
        print("테스트 이미지 생성 중...")
        image, mask = create_test_image_and_mask(300)

    # 시각화 클래스 생성
    visualizer = TeleaVisualizer(image, mask, radius=5)

    # 윈도우 설정
    window_name = "Telea Algorithm Visualization"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 상태
    auto_play = False
    pixels_per_step = max(1, np.sum(mask > 0) // 100)  # 전체의 1%씩

    print("\n" + "=" * 50)
    print("Telea Inpainting 알고리즘 시각화")
    print("=" * 50)
    print(f"이미지 크기: {image.shape[1]}x{image.shape[0]}")
    print(f"마스크 픽셀 수: {np.sum(mask > 0)}")
    print(f"단계당 픽셀: {pixels_per_step}")
    print("=" * 50)
    print("\n[조작법]")
    print("  Space : 다음 단계")
    print("  Enter : 자동 재생 토글")
    print("  r     : 리셋")
    print("  s     : GIF 저장")
    print("  q     : 종료")
    print("=" * 50 + "\n")

    # 범례 이미지 생성
    def create_legend() -> np.ndarray:
        legend = np.zeros((120, 250, 3), dtype=np.uint8)
        cv2.putText(legend, "Legend:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # KNOWN
        cv2.rectangle(legend, (10, 40), (30, 60), COLOR_KNOWN, -1)
        cv2.putText(legend, "KNOWN (restored)", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # BAND
        cv2.rectangle(legend, (10, 65), (30, 85), COLOR_BAND, -1)
        cv2.putText(legend, "BAND (boundary)", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # UNKNOWN
        cv2.rectangle(legend, (10, 90), (30, 110), COLOR_UNKNOWN, -1)
        cv2.putText(legend, "UNKNOWN (to fill)", (40, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return legend

    legend = create_legend()

    while True:
        # 시각화 이미지 생성
        vis = visualizer.get_visualization()
        state_map = visualizer.get_state_map()

        # 상태 맵 크기 조정
        h, w = vis.shape[:2]
        state_map_resized = cv2.resize(state_map, (w, h))

        # 원본 이미지
        original_display = visualizer.original.copy()
        cv2.putText(original_display, "Original + Mask", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 마스크 오버레이
        original_display[mask > 0] = [0, 0, 255]

        # 정보 텍스트
        info_text = f"Step: {visualizer.step}"
        remaining = np.sum(visualizer.flag == UNKNOWN) + np.sum(visualizer.flag == BAND)
        info_text += f" | Remaining: {remaining}"
        info_text += f" | Auto: {'ON' if auto_play else 'OFF'}"

        # 결합
        cv2.putText(vis, "Inpainting Progress", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(state_map_resized, "State Map", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 3열 배치
        top_row = np.hstack([original_display, vis, state_map_resized])

        # 범례 및 정보 추가
        info_bar = np.zeros((40, top_row.shape[1], 3), dtype=np.uint8)
        cv2.putText(info_bar, info_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 범례 크기 조정 및 추가
        legend_resized = cv2.resize(legend, (top_row.shape[1], 120))

        combined = np.vstack([top_row, info_bar, legend_resized])

        # 완료 표시
        if visualizer.is_done():
            cv2.putText(combined, "COMPLETE!", (combined.shape[1] // 2 - 100, combined.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            auto_play = False

        cv2.imshow(window_name, combined)

        # 자동 재생
        wait_time = 1 if auto_play else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            if not visualizer.is_done():
                visualizer.step_forward(pixels_per_step)
                visualizer.save_frame()
        elif key == 13:  # Enter
            auto_play = not auto_play
            print(f"자동 재생: {'ON' if auto_play else 'OFF'}")
        elif key == ord('r'):
            visualizer = TeleaVisualizer(image, mask, radius=5)
            auto_play = False
            print("리셋됨")
        elif key == ord('s'):
            # GIF 저장 (프레임들 저장)
            if visualizer.frames:
                output_path = "telea_animation.gif"
                try:
                    import imageio
                    # BGR to RGB 변환
                    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in visualizer.frames]
                    imageio.mimsave(output_path, rgb_frames, duration=0.1)
                    print(f"GIF 저장됨: {output_path}")
                except ImportError:
                    print("imageio 설치 필요: pip install imageio")
                    # 대안: 개별 이미지로 저장
                    for i, frame in enumerate(visualizer.frames):
                        cv2.imwrite(f"frame_{i:04d}.png", frame)
                    print(f"{len(visualizer.frames)}개 프레임 저장됨")

        # 자동 재생 시 진행
        if auto_play and not visualizer.is_done():
            visualizer.step_forward(pixels_per_step)
            visualizer.save_frame()

    cv2.destroyAllWindows()

    # 최종 결과 저장
    final_result = visualizer.get_final_result()
    cv2.imwrite("telea_result.png", final_result)
    print("최종 결과 저장됨: telea_result.png")


if __name__ == "__main__":
    main()
