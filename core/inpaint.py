"""
Telea Inpainting 알고리즘 직접 구현

Fast Marching Method (FMM) 기반의 이미지 인페인팅 알고리즘
cv2.inpaint() 함수를 사용하지 않고 직접 구현

참고 논문:
    A. Telea, "An Image Inpainting Technique Based on the Fast Marching Method"
    Journal of Graphics Tools, 2004
"""

import heapq
from typing import List, Tuple, Set
import numpy as np


# 픽셀 상태 상수
KNOWN = 0      # 알려진 픽셀 (원본 또는 이미 복원됨)
BAND = 1       # 경계 픽셀 (처리 대기 중)
UNKNOWN = 2   # 알려지지 않은 픽셀 (마스크 내부)


class TeleaInpainter:
    """Telea 알고리즘 기반 인페인팅 클래스"""

    def __init__(self, inpaint_radius: int = 5):
        """
        Args:
            inpaint_radius: 인페인팅에 사용할 이웃 픽셀 반경
        """
        self.radius = inpaint_radius
        self.epsilon = 1e-6  # 0으로 나누기 방지

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        이미지에서 마스크 영역을 인페인팅

        Args:
            image: BGR 형식의 원본 이미지 (H, W, 3)
            mask: 제거할 영역의 마스크 (H, W), 0이 아닌 값 = 제거할 영역

        Returns:
            인페인팅된 이미지
        """
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)

        # 마스크 이진화
        binary_mask = (mask > 0).astype(np.uint8)

        # 상태 맵 초기화
        flag = np.zeros((h, w), dtype=np.uint8)
        flag[binary_mask > 0] = UNKNOWN

        # 거리 맵 초기화 (경계까지의 거리)
        dist = np.full((h, w), np.inf, dtype=np.float32)
        dist[flag == KNOWN] = 0

        # 경계 픽셀 찾기 및 우선순위 큐 초기화
        heap: List[Tuple[float, int, int]] = []
        self._init_boundary(flag, dist, heap, h, w)

        # Fast Marching Method
        self._fast_marching(result, flag, dist, heap, h, w)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _init_boundary(
        self,
        flag: np.ndarray,
        dist: np.ndarray,
        heap: List[Tuple[float, int, int]],
        h: int,
        w: int
    ) -> None:
        """경계 픽셀 초기화"""
        # 4방향 이웃
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y in range(h):
            for x in range(w):
                if flag[y, x] == UNKNOWN:
                    # 이웃 중 KNOWN 픽셀이 있으면 BAND로 설정
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if flag[ny, nx] == KNOWN:
                                flag[y, x] = BAND
                                dist[y, x] = 1.0
                                heapq.heappush(heap, (1.0, y, x))
                                break

    def _fast_marching(
        self,
        image: np.ndarray,
        flag: np.ndarray,
        dist: np.ndarray,
        heap: List[Tuple[float, int, int]],
        h: int,
        w: int
    ) -> None:
        """Fast Marching Method로 픽셀 복원"""
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while heap:
            d, y, x = heapq.heappop(heap)

            # 이미 처리된 픽셀은 건너뛰기
            if flag[y, x] == KNOWN:
                continue

            # 현재 픽셀 복원
            self._inpaint_pixel(image, flag, dist, y, x, h, w)

            # 상태 업데이트
            flag[y, x] = KNOWN

            # 이웃 UNKNOWN 픽셀들을 BAND로 추가
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if flag[ny, nx] == UNKNOWN:
                        flag[ny, nx] = BAND
                        # 거리 업데이트 (Eikonal 방정식 근사)
                        new_dist = self._compute_distance(dist, ny, nx, h, w)
                        dist[ny, nx] = new_dist
                        heapq.heappush(heap, (new_dist, ny, nx))

    def _compute_distance(
        self,
        dist: np.ndarray,
        y: int,
        x: int,
        h: int,
        w: int
    ) -> float:
        """Eikonal 방정식을 사용하여 거리 계산"""
        # x 방향 최소 거리
        dx_min = np.inf
        if x > 0:
            dx_min = min(dx_min, dist[y, x - 1])
        if x < w - 1:
            dx_min = min(dx_min, dist[y, x + 1])

        # y 방향 최소 거리
        dy_min = np.inf
        if y > 0:
            dy_min = min(dy_min, dist[y - 1, x])
        if y < h - 1:
            dy_min = min(dy_min, dist[y + 1, x])

        # 둘 다 무한대면 기본값 반환
        if np.isinf(dx_min) and np.isinf(dy_min):
            return np.inf

        # 하나만 유효하면 그 값 + 1
        if np.isinf(dx_min):
            return dy_min + 1.0
        if np.isinf(dy_min):
            return dx_min + 1.0

        # 둘 다 유효하면 Eikonal 방정식 풀기
        # (d - dx)^2 + (d - dy)^2 = 1
        # 이차방정식: 2d^2 - 2(dx+dy)d + (dx^2 + dy^2 - 1) = 0
        a = 2.0
        b = -2.0 * (dx_min + dy_min)
        c = dx_min ** 2 + dy_min ** 2 - 1.0

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return min(dx_min, dy_min) + 1.0

        return (-b + np.sqrt(discriminant)) / (2 * a)

    def _inpaint_pixel(
        self,
        image: np.ndarray,
        flag: np.ndarray,
        dist: np.ndarray,
        y: int,
        x: int,
        h: int,
        w: int
    ) -> None:
        """단일 픽셀 복원 (Telea 가중 평균)"""
        r = self.radius

        # 이웃 KNOWN 픽셀들의 가중 평균 계산
        total_weight = 0.0
        weighted_sum = np.zeros(3, dtype=np.float32)

        # 현재 픽셀에서의 그래디언트 추정 (경계 방향)
        grad_y, grad_x = self._estimate_gradient(dist, y, x, h, w)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = y + dy, x + dx

                # 경계 체크 및 KNOWN 픽셀만
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if flag[ny, nx] != KNOWN:
                    continue

                # 거리 가중치
                pixel_dist = np.sqrt(dy ** 2 + dx ** 2)
                if pixel_dist > r or pixel_dist < self.epsilon:
                    continue

                # Telea 가중치: 거리, 방향, 레벨셋
                # 1. 거리 가중치 (가까울수록 높음)
                w_dist = 1.0 / (pixel_dist ** 2 + self.epsilon)

                # 2. 방향 가중치 (등고선 방향일수록 높음)
                # 경계에서 현재 픽셀로의 방향과 그래디언트가 수직이면 좋음
                if pixel_dist > self.epsilon:
                    dir_y = dy / pixel_dist
                    dir_x = dx / pixel_dist
                    # 그래디언트와의 내적 (그래디언트 방향과 반대일수록 좋음)
                    dot = grad_y * dir_y + grad_x * dir_x
                    w_dir = max(0.0, -dot + 1.0)  # [-1, 1] -> [0, 2]
                else:
                    w_dir = 1.0

                # 3. 레벨셋 가중치 (같은 등고선상일수록 높음)
                level_diff = abs(dist[ny, nx] - dist[y, x])
                w_level = 1.0 / (1.0 + level_diff)

                # 최종 가중치
                weight = w_dist * w_dir * w_level

                total_weight += weight
                weighted_sum += weight * image[ny, nx]

        # 가중 평균 적용
        if total_weight > self.epsilon:
            image[y, x] = weighted_sum / total_weight
        else:
            # 폴백: 단순 평균
            self._fallback_average(image, flag, y, x, h, w, r)

    def _estimate_gradient(
        self,
        dist: np.ndarray,
        y: int,
        x: int,
        h: int,
        w: int
    ) -> Tuple[float, float]:
        """거리 맵에서 그래디언트 추정 (경계 방향)"""
        grad_y = 0.0
        grad_x = 0.0

        # x 방향 그래디언트 (inf 값 처리)
        if x > 0 and x < w - 1:
            d_left = dist[y, x - 1]
            d_right = dist[y, x + 1]
            if not (np.isinf(d_left) or np.isinf(d_right)):
                grad_x = (d_right - d_left) / 2.0
        elif x > 0:
            d_left = dist[y, x - 1]
            d_curr = dist[y, x]
            if not (np.isinf(d_left) or np.isinf(d_curr)):
                grad_x = d_curr - d_left
        elif x < w - 1:
            d_right = dist[y, x + 1]
            d_curr = dist[y, x]
            if not (np.isinf(d_right) or np.isinf(d_curr)):
                grad_x = d_right - d_curr

        # y 방향 그래디언트 (inf 값 처리)
        if y > 0 and y < h - 1:
            d_up = dist[y - 1, x]
            d_down = dist[y + 1, x]
            if not (np.isinf(d_up) or np.isinf(d_down)):
                grad_y = (d_down - d_up) / 2.0
        elif y > 0:
            d_up = dist[y - 1, x]
            d_curr = dist[y, x]
            if not (np.isinf(d_up) or np.isinf(d_curr)):
                grad_y = d_curr - d_up
        elif y < h - 1:
            d_down = dist[y + 1, x]
            d_curr = dist[y, x]
            if not (np.isinf(d_down) or np.isinf(d_curr)):
                grad_y = d_down - d_curr

        # 정규화 (magnitude가 0이면 기본값 반환)
        mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        if mag < self.epsilon:
            return 0.0, 0.0
        return grad_y / mag, grad_x / mag

    def _fallback_average(
        self,
        image: np.ndarray,
        flag: np.ndarray,
        y: int,
        x: int,
        h: int,
        w: int,
        r: int
    ) -> None:
        """폴백: 단순 평균"""
        count = 0
        total = np.zeros(3, dtype=np.float32)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and flag[ny, nx] == KNOWN:
                    total += image[ny, nx]
                    count += 1

        if count > 0:
            image[y, x] = total / count


def telea_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    inpaint_radius: int = 5
) -> np.ndarray:
    """
    Telea 알고리즘으로 이미지 인페인팅 (직접 구현)

    Args:
        image: BGR 형식의 원본 이미지
        mask: 제거할 영역의 마스크 (0이 아닌 값 = 제거할 영역)
        inpaint_radius: 인페인팅에 사용할 이웃 픽셀 반경

    Returns:
        인페인팅된 이미지
    """
    inpainter = TeleaInpainter(inpaint_radius)
    return inpainter.inpaint(image, mask)


def cv_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    inpaint_radius: int = 5
) -> np.ndarray:
    """
    OpenCV 내장 Telea 인페인팅 (cv2.inpaint 래퍼)

    Args:
        image: BGR 형식의 원본 이미지
        mask: 제거할 영역의 마스크 (0이 아닌 값 = 제거할 영역)
        inpaint_radius: 인페인팅 반경

    Returns:
        인페인팅된 이미지
    """
    import cv2
    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)


def class_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 5
) -> np.ndarray:
    """
    수업에서 배운 기법들만 사용한 인페인팅 (반복적 경계 확산)

    알고리즘:
        1. erode()로 마스크 경계를 1픽셀씩 찾음
        2. 경계 픽셀을 주변 알려진 픽셀들의 거리 기반 가중치 평균으로 채움
        3. 마스크가 완전히 채워질 때까지 반복
        4. GaussianBlur()로 경계 스무딩

    사용 기법:
        - erode(): 경계 찾기 (morphological operation)
        - subtract(): 경계 마스크 계산
        - GaussianBlur(): 후처리 스무딩
        - 픽셀 접근 (numpy 인덱싱): 가중치 평균 계산

    Args:
        image: BGR 형식의 원본 이미지
        mask: 제거할 영역의 마스크 (0이 아닌 값 = 제거할 영역)
        radius: 인페인팅에 사용할 이웃 픽셀 반경

    Returns:
        인페인팅된 이미지
    """
    import cv2

    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)

    # 마스크 이진화
    remaining = (mask > 0).astype(np.uint8) * 255

    # erode용 커널 (3x3)
    kernel = np.ones((3, 3), np.uint8)

    # 반복: 경계에서 안쪽으로 채워나감
    iteration = 0
    max_iterations = max(h, w)  # 무한 루프 방지

    while np.any(remaining > 0) and iteration < max_iterations:
        iteration += 1

        # 1. 경계 찾기: erode + subtract
        eroded = cv2.erode(remaining, kernel, iterations=1)
        boundary = cv2.subtract(remaining, eroded)

        if not np.any(boundary > 0):
            break

        # 2. 경계 픽셀 찾기
        boundary_points = np.where(boundary > 0)

        # 3. 경계 픽셀 채우기 (가중치 평균)
        for y, x in zip(boundary_points[0], boundary_points[1]):
            total_weight = 0.0
            weighted_sum = np.zeros(3, dtype=np.float32)

            # 주변 radius x radius 영역 탐색
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx

                    # 경계 체크
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue

                    # 알려진 픽셀만 (remaining이 0인 픽셀)
                    if remaining[ny, nx] == 0:
                        dist = np.sqrt(dy ** 2 + dx ** 2)
                        if dist > 0:
                            # 거리 기반 가중치 (가까울수록 높음)
                            weight = 1.0 / (dist ** 2 + 0.001)
                            total_weight += weight
                            weighted_sum += weight * result[ny, nx]

            # 가중 평균 적용
            if total_weight > 0:
                result[y, x] = weighted_sum / total_weight

        # 4. 마스크 업데이트: 경계가 채워졌으니 eroded로 대체
        remaining = eroded

    # 5. 결과 클리핑
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 6. 후처리: GaussianBlur로 경계 스무딩
    blurred = cv2.GaussianBlur(result, (5, 5), 0)

    # 마스크 경계 영역만 블렌딩 (경계가 부드럽게)
    mask_float = mask.astype(np.float32) / 255.0
    mask_blur = cv2.GaussianBlur(mask_float, (15, 15), 0)

    # 3채널로 확장
    if len(mask_blur.shape) == 2:
        mask_blur = mask_blur[:, :, np.newaxis]

    # 원본과 블러 블렌딩 (마스크 영역에서만 약간의 스무딩)
    blend_factor = 0.3
    result = (result * (1 - mask_blur * blend_factor) +
              blurred * mask_blur * blend_factor).astype(np.uint8)

    return result
