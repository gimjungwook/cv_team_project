"""
Inpainting 방식 비교 데모

Telea 직접구현 vs cv2.inpaint() vs LaMa 결과 비교

사용법:
    python demo_comparison.py [이미지경로]
    python demo_comparison.py  # 기본 테스트 이미지

키보드:
    r: ROI 다시 선택
    s: 결과 저장
    q: 종료
"""

import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

# 프로젝트 모듈
from core.inpaint import telea_inpaint
from core.lama import load_lama_model


def create_test_image(size: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """복잡한 테스트 이미지 생성"""
    image = np.zeros((size, size, 3), dtype=np.uint8)

    # 배경 그라데이션
    for i in range(size):
        for j in range(size):
            image[i, j] = [
                int(100 + 50 * np.sin(i * 0.05)),
                int(100 + 50 * np.sin(j * 0.05)),
                int(150 + 50 * np.sin((i + j) * 0.03))
            ]

    # 패턴 추가
    for i in range(0, size, 40):
        cv2.line(image, (i, 0), (i, size), (200, 200, 200), 2)
        cv2.line(image, (0, i), (size, i), (200, 200, 200), 2)

    # 원형 추가
    cv2.circle(image, (size // 4, size // 4), 30, (255, 100, 100), -1)
    cv2.circle(image, (3 * size // 4, size // 4), 30, (100, 255, 100), -1)
    cv2.circle(image, (size // 2, 3 * size // 4), 30, (100, 100, 255), -1)

    # 마스크 (중앙 영역)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (size // 3, size // 3), (2 * size // 3, 2 * size // 3), 255, -1)

    return image, mask


def run_telea_custom(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Telea 직접 구현 실행"""
    start = time.time()
    result = telea_inpaint(image, mask, inpaint_radius=5)
    elapsed = time.time() - start
    return result, elapsed


def run_cv2_inpaint(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """OpenCV inpaint 실행"""
    start = time.time()
    result = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    elapsed = time.time() - start
    return result, elapsed


def run_lama(image: np.ndarray, mask: np.ndarray, model_dir: Path) -> Tuple[Optional[np.ndarray], float]:
    """LaMa 실행"""
    lama = load_lama_model(model_dir)
    if lama is None:
        return None, 0.0

    start = time.time()
    result = lama.inpaint(image, mask)
    elapsed = time.time() - start
    return result, elapsed


def create_comparison_view(
    original: np.ndarray,
    mask: np.ndarray,
    telea_result: np.ndarray,
    telea_time: float,
    cv2_result: np.ndarray,
    cv2_time: float,
    lama_result: Optional[np.ndarray],
    lama_time: float
) -> np.ndarray:
    """비교 뷰 생성"""
    h, w = original.shape[:2]

    # 모든 이미지 크기 통일
    target_h, target_w = 400, 400

    def resize_with_padding(img: np.ndarray) -> np.ndarray:
        """비율 유지하며 리사이즈"""
        scale = min(target_w / img.shape[1], target_h / img.shape[0])
        new_w = int(img.shape[1] * scale)
        new_h = int(img.shape[0] * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # 패딩
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return result

    # 원본 + 마스크 오버레이
    original_masked = original.copy()
    original_masked[mask > 0] = [0, 0, 255]

    # 리사이즈
    img_original = resize_with_padding(original_masked)
    img_telea = resize_with_padding(telea_result)
    img_cv2 = resize_with_padding(cv2_result)

    if lama_result is not None:
        img_lama = resize_with_padding(lama_result)
    else:
        img_lama = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        cv2.putText(img_lama, "LaMa model", (50, target_h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        cv2.putText(img_lama, "not found", (70, target_h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    # 텍스트 추가 함수
    def add_label(img: np.ndarray, title: str, time_sec: float, color: Tuple[int, int, int]) -> np.ndarray:
        result = img.copy()
        # 배경 박스
        cv2.rectangle(result, (0, 0), (target_w, 70), (0, 0, 0), -1)
        cv2.rectangle(result, (0, 0), (target_w, 70), color, 3)

        # 제목
        cv2.putText(result, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 시간
        if time_sec > 0:
            time_str = f"Time: {time_sec:.3f}s"
            cv2.putText(result, time_str, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return result

    # 라벨 추가
    img_original = add_label(img_original, "Original + Mask", 0, (0, 255, 255))
    img_telea = add_label(img_telea, "Telea (Custom)", telea_time, (0, 200, 255))
    img_cv2 = add_label(img_cv2, "cv2.inpaint()", cv2_time, (0, 255, 0))
    img_lama = add_label(img_lama, "LaMa (Deep Learning)", lama_time, (255, 100, 255))

    # 2x2 그리드
    top_row = np.hstack([img_original, img_telea])
    bottom_row = np.hstack([img_cv2, img_lama])
    combined = np.vstack([top_row, bottom_row])

    # 타이틀 바
    title_bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, "Inpainting Methods Comparison", (combined.shape[1] // 2 - 250, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # 하단 정보 바
    info_bar = np.zeros((80, combined.shape[1], 3), dtype=np.uint8)

    # 속도 비교
    if telea_time > 0 and cv2_time > 0:
        speedup = telea_time / cv2_time
        cv2.putText(info_bar, f"cv2 is {speedup:.1f}x faster than custom Telea",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # 품질 안내
    cv2.putText(info_bar, "Quality: LaMa > cv2 = Telea (for large masks)",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    result = np.vstack([title_bar, combined, info_bar])
    return result


def main():
    print("\n" + "=" * 60)
    print("   Inpainting Methods Comparison Demo")
    print("=" * 60)

    # 이미지 로드
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            sys.exit(1)
        print(f"이미지 로드: {image_path}")

        # 마스크 선택
        print("\n마스크 영역을 선택하세요 (드래그 후 Enter)")
        roi = cv2.selectROI("Select Mask Area", image, fromCenter=False)
        cv2.destroyWindow("Select Mask Area")

        x, y, w, h = roi
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if w > 0 and h > 0:
            mask[y:y+h, x:x+w] = 255
        else:
            print("ROI가 선택되지 않음. 중앙 영역 사용")
            center_h, center_w = image.shape[0] // 2, image.shape[1] // 2
            size = min(image.shape[:2]) // 4
            mask[center_h - size:center_h + size, center_w - size:center_w + size] = 255
    else:
        print("테스트 이미지 생성 중...")
        image, mask = create_test_image(400)

    print(f"\n이미지 크기: {image.shape[1]}x{image.shape[0]}")
    print(f"마스크 픽셀 수: {np.sum(mask > 0)}")

    # 각 방식 실행
    print("\n" + "-" * 40)
    print("인페인팅 실행 중...")
    print("-" * 40)

    # 1. Telea 직접 구현
    print("\n[1/3] Telea (직접 구현) 실행 중...")
    telea_result, telea_time = run_telea_custom(image, mask)
    print(f"      완료: {telea_time:.3f}초")

    # 2. cv2.inpaint
    print("\n[2/3] cv2.inpaint() 실행 중...")
    cv2_result, cv2_time = run_cv2_inpaint(image, mask)
    print(f"      완료: {cv2_time:.3f}초")

    # 3. LaMa
    print("\n[3/3] LaMa 실행 중...")
    model_dir = Path("models")
    lama_result, lama_time = run_lama(image, mask, model_dir)
    if lama_result is not None:
        print(f"      완료: {lama_time:.3f}초")
    else:
        print("      LaMa 모델을 찾을 수 없습니다")
        print("      models/lama/lama_fp32.onnx 필요")

    # 비교 뷰 생성
    print("\n" + "-" * 40)
    print("비교 뷰 생성 중...")

    comparison = create_comparison_view(
        image, mask,
        telea_result, telea_time,
        cv2_result, cv2_time,
        lama_result, lama_time
    )

    # 결과 표시
    window_name = "Inpainting Comparison"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 1000)

    print("\n" + "=" * 60)
    print("결과 표시 중")
    print("=" * 60)
    print("\n[조작법]")
    print("  s: 결과 저장")
    print("  q: 종료")
    print("=" * 60)

    while True:
        cv2.imshow(window_name, comparison)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            # 결과 저장
            cv2.imwrite("comparison_result.png", comparison)
            cv2.imwrite("result_telea_custom.png", telea_result)
            cv2.imwrite("result_cv2_inpaint.png", cv2_result)
            if lama_result is not None:
                cv2.imwrite("result_lama.png", lama_result)
            print("\n결과 저장됨:")
            print("  - comparison_result.png")
            print("  - result_telea_custom.png")
            print("  - result_cv2_inpaint.png")
            if lama_result is not None:
                print("  - result_lama.png")

    cv2.destroyAllWindows()

    # 최종 요약
    print("\n" + "=" * 60)
    print("   Performance Summary")
    print("=" * 60)
    print(f"\n{'Method':<25} {'Time (sec)':<15} {'Speed':<15}")
    print("-" * 55)
    print(f"{'Telea (Custom)':<25} {telea_time:<15.4f} {'1.0x (baseline)':<15}")
    if cv2_time > 0:
        print(f"{'cv2.inpaint()':<25} {cv2_time:<15.4f} {f'{telea_time/cv2_time:.1f}x faster':<15}")
    if lama_time > 0:
        print(f"{'LaMa (ONNX)':<25} {lama_time:<15.4f} {f'{telea_time/lama_time:.1f}x':<15}")
    print("=" * 60)


if __name__ == "__main__":
    main()
