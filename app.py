"""
Object Detector & Eraser - 간소화된 OpenCV 기반 버전

키보드 단축키:
    d: YOLO 탐지 실행
    r: ROI 수동 추가 (드래그)
    e: 선택 영역 제거
    t: Telea 직접 구현 모드
    c: cv2.inpaint() 모드
    l: LaMa 모드
    s: 결과 저장
    z: 원본 복원
    q: 종료
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from core.detect import load_default_detector, YOLODetector, DetectionResult
from core.lama import load_lama_model, LamaInpainter
from core.inpaint import telea_inpaint, class_inpaint


class ObjectEraser:
    """OpenCV 기반 객체 제거 애플리케이션"""

    MAX_WINDOW_WIDTH = 1600  # 최대 윈도우 너비
    MAX_WINDOW_HEIGHT = 900  # 최대 윈도우 높이

    def __init__(self, image_path: str):
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

        self.result = self.original.copy()
        self.mode = "telea"  # "telea", "telea_cv", "lama"
        self.detections: List[DetectionResult] = []
        self.rois: List[Tuple[int, int, int, int]] = []
        self.detector: Optional[YOLODetector] = None
        self.lama: Optional[LamaInpainter] = None
        self.window_name = "Object Eraser"

    def run(self) -> None:
        """메인 루프"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.MAX_WINDOW_WIDTH, self.MAX_WINDOW_HEIGHT)

        while True:
            self._show_image()
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('d'):
                self._detect()
            elif key == ord('r'):
                self._add_roi()
            elif key == ord('e'):
                self._erase()
            elif key == ord('t'):
                self.mode = "telea"
                print("모드 변경: Telea 직접 구현 → 'e' 키로 제거 실행")
            elif key == ord('c'):
                self.mode = "telea_cv"
                print("모드 변경: 수업 기법 (erode+가중치평균+GaussianBlur) → 'e' 키로 제거 실행")
            elif key == ord('l'):
                self.mode = "lama"
                print("모드 변경: LaMa → 'e' 키로 제거 실행")
            elif key == ord('s'):
                self._save()
            elif key == ord('z'):
                self.result = self.original.copy()
                self.detections = []
                self.rois = []
                print("원본으로 복원됨")

        cv2.destroyAllWindows()

    def _get_font_scale(self, base_scale: float = 1.0) -> float:
        """이미지 해상도에 따른 폰트 스케일 계산 (1080p 기준)"""
        h, w = self.original.shape[:2]
        # 1080p (1920x1080)를 기준으로 스케일 계산
        reference_height = 1080
        scale_factor = h / reference_height
        return base_scale * scale_factor

    def _get_thickness(self, base_thick: int = 2) -> int:
        """이미지 해상도에 따른 선 두께 계산"""
        h, w = self.original.shape[:2]
        reference_height = 1080
        scale_factor = h / reference_height
        return max(1, int(base_thick * scale_factor))

    def _show_image(self) -> None:
        """Before | After 나란히 표시 + 조작법"""
        h, w = self.original.shape[:2]

        # 탐지 결과와 ROI를 원본 이미지에 표시
        display_original = self.original.copy()
        self._draw_overlays(display_original)

        # 나란히 배치
        combined = np.hstack([display_original, self.result])

        # 해상도에 따른 스케일 계산
        scale = h / 1080  # 1080p 기준

        # 조작법 패널 (오른쪽 상단)
        panel_x = w + int(15 * scale)
        line_h = int(28 * scale)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self._get_font_scale(0.6)
        thick = self._get_thickness(2)

        # 현재 모드
        mode_text = {"telea": "Telea", "telea_cv": "Class", "lama": "LaMa"}
        mode_font_scale = self._get_font_scale(2.0)
        mode_thick = self._get_thickness(4)
        cv2.putText(combined, f"[Mode] {mode_text.get(self.mode, self.mode)}",
                    (panel_x, int(40 * scale)), font, mode_font_scale, (0, 255, 0), mode_thick)

        # 탐지/ROI 개수
        info_font_scale = self._get_font_scale(0.7)
        cv2.putText(combined, f"Detections: {len(self.detections)} | ROIs: {len(self.rois)}",
                    (panel_x, int(80 * scale)), font, info_font_scale, (0, 255, 255), thick)

        # 조작법 안내
        controls = [
            ("=== Controls ===", (255, 255, 255)),
            ("d: YOLO Detect", (100, 255, 100)),
            ("r: Add ROI", (100, 255, 100)),
            ("e: ERASE!", (100, 100, 255)),
            ("=== Mode ===", (255, 255, 255)),
            ("t: Telea", (200, 200, 100)),
            ("c: Class", (200, 200, 100)),
            ("l: LaMa", (200, 200, 100)),
            ("=== Other ===", (255, 255, 255)),
            ("s: Save", (200, 200, 200)),
            ("z: Reset", (200, 200, 200)),
            ("q/ESC: Quit", (200, 200, 200)),
        ]

        y = int(115 * scale)
        for text, color in controls:
            cv2.putText(combined, text, (panel_x, y), font, font_scale, color, thick)
            y += line_h

        # 라벨 (하단)
        label_font_scale = self._get_font_scale(0.7)
        label_offset = int(50 * scale)
        label_y = h - int(15 * scale)
        cv2.putText(combined, "[ Original ]", (w // 2 - label_offset, label_y),
                    font, label_font_scale, (255, 255, 255), thick)
        cv2.putText(combined, "[ Result ]", (w + w // 2 - label_offset, label_y),
                    font, label_font_scale, (255, 255, 255), thick)

        cv2.imshow(self.window_name, combined)

    def _draw_overlays(self, image: np.ndarray) -> None:
        """탐지 결과와 ROI 박스 그리기"""
        font_scale = self._get_font_scale(0.35)
        thick = self._get_thickness(1)
        label_offset = max(3, int(5 * (self.original.shape[0] / 1080)))

        # 탐지 결과
        for det in self.detections:
            x, y, w, h = det.bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thick)
            label = f"{det.label} {det.confidence * 100:.0f}%"
            cv2.putText(image, label, (x, y - label_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thick)

        # 수동 ROI
        for i, roi in enumerate(self.rois):
            x, y, w, h = roi
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), thick)
            cv2.putText(image, f"ROI {i + 1}", (x, y - label_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thick)

    def _detect(self) -> None:
        """YOLO 탐지"""
        try:
            if self.detector is None:
                print("YOLO 모델 로딩 중...")
                self.detector = load_default_detector(Path("models"))

            print("객체 탐지 중...")
            self.detections = self.detector.detect(self.original)
            print(f"탐지 완료: {len(self.detections)}개 객체 발견")
        except FileNotFoundError as e:
            print(f"오류: {e}")
        except Exception as e:
            print(f"탐지 실패: {e}")

    def _add_roi(self) -> None:
        """cv2.selectROI로 ROI 추가"""
        print("ROI 선택: 마우스로 드래그 후 Enter/Space, 취소는 c")
        roi = cv2.selectROI("Select ROI", self.original, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")

        x, y, w, h = roi
        if w > 0 and h > 0:
            self.rois.append((x, y, w, h))
            print(f"ROI 추가됨: ({x}, {y}, {w}, {h})")
        else:
            print("ROI 선택 취소됨")

    def _create_mask(self) -> np.ndarray:
        """GrabCut으로 마스크 생성"""
        h, w = self.original.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        boxes = [(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]) for d in self.detections]
        boxes.extend(self.rois)

        for bx, by, bw, bh in boxes:
            # 경계 보정
            x1 = max(bx, 0)
            y1 = max(by, 0)
            x2 = min(bx + bw, w - 1)
            y2 = min(by + bh, h - 1)

            if x2 - x1 < 10 or y2 - y1 < 10:
                mask[y1:y2, x1:x2] = 255
                continue

            # GrabCut 시도
            try:
                gc_mask = np.zeros((h, w), dtype=np.uint8)
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)
                cv2.grabCut(self.original, gc_mask, (x1, y1, x2 - x1, y2 - y1),
                            bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
                fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                if np.any(fg):
                    mask = cv2.bitwise_or(mask, fg)
                else:
                    mask[y1:y2, x1:x2] = 255
            except cv2.error:
                mask[y1:y2, x1:x2] = 255

        # 마스크 확장
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask

    def _erase(self) -> None:
        """선택된 영역 제거"""
        if not self.detections and not self.rois:
            print("제거할 영역이 없습니다. 'd'로 탐지하거나 'r'로 ROI를 추가하세요.")
            return

        print(f"마스크 생성 중... ({len(self.detections)}개 탐지 + {len(self.rois)}개 ROI)")
        mask = self._create_mask()

        if not np.any(mask):
            print("마스크가 비어있습니다.")
            return

        try:
            if self.mode == "telea":
                print("Telea 인페인팅 처리 중 (직접 구현)...")
                self.result = telea_inpaint(self.result, mask, inpaint_radius=5)
            elif self.mode == "telea_cv":
                print("수업 기법 인페인팅 처리 중 (erode + 가중치 평균 + GaussianBlur)...")
                self.result = class_inpaint(self.result, mask, radius=5)
            elif self.mode == "lama":
                if self.lama is None:
                    print("LaMa 모델 로딩 중...")
                    self.lama = load_lama_model(Path("models"))
                    if self.lama is None:
                        print("LaMa 모델을 찾을 수 없습니다. models/lama/lama_fp32.onnx 필요")
                        return
                print("LaMa 인페인팅 처리 중...")
                self.result = self.lama.inpaint(self.result, mask)

            print("제거 완료!")
            # 처리 후 탐지/ROI 초기화
            self.detections = []
            self.rois = []
        except Exception as e:
            print(f"제거 실패: {e}")

    def _save(self) -> None:
        """결과 저장"""
        output_path = "result.png"
        cv2.imwrite(output_path, self.result)
        print(f"저장됨: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("사용법: python app.py <이미지 경로>")
        print("예: python app.py image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)

    print(f"이미지 로드: {image_path}")
    print("\n=== 키보드 단축키 ===")
    print("d: YOLO 탐지")
    print("r: ROI 수동 추가")
    print("e: 선택 영역 제거")
    print("t: Telea 직접 구현 모드")
    print("c: 수업 기법 모드 (erode+가중치평균)")
    print("l: LaMa 모드")
    print("s: 결과 저장")
    print("z: 원본 복원")
    print("q: 종료")
    print("=====================\n")

    app = ObjectEraser(image_path)
    app.run()


if __name__ == "__main__":
    main()
