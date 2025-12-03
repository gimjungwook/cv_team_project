# PRD: 객체 탐지 및 제거 프로그램 (Object Detector & Eraser)

## 1. 프로젝트 개요
본 프로젝트는 정지 이미지(`Image`) 내의 객체를 자동으로 탐지하고, 선택된 객체를 배경과 자연스럽게 어우러지도록 제거(Erase/Inpainting)하는 데스크톱 애플리케이션을 개발하는 것을 목표로 한다.
교수님의 요구사항에 따라 **수업 시간에 다룬 알고리즘(Class Code)**을 기반으로 한 버전과, **최신 딥러닝 모델(Lama)**을 적용한 버전을 각각 구현하여 비교 분석한다.

## 2. 프로젝트 목표
1.  **Class Code 기반 구현:** 수업 자료(`Codes/`)에 포함된 알고리즘(YOLO, GrabCut 등)만을 활용하여 이미지 내 객체 탐지 및 제거 기능을 Python으로 구현한다.
2.  **Open Source 기반 구현:** 최신 Inpainting 모델인 **Lama (Large Mask Inpainting)**를 도입하여 객체 제거 성능을 극대화한다.
3.  **사용자 친화적 UI:** Python `tkinter`를 사용하여 직관적인 GUI를 제공한다.
4.  **범위 제한:** 비디오 처리는 제외하고 **단일 이미지 처리**에 집중하여 결과물의 품질을 높인다.

## 3. 상세 요구사항

### 3.1. 기능적 요구사항 (Functional Requirements)

#### 공통 기능
*   **이미지 로드:** `jpg`, `png`, `bmp` 등의 이미지 파일 불러오기.
*   **UI (Tkinter):** 이미지 캔버스, 기능 제어 버튼(탐지, 제거, 저장), 모드 전환(Class vs Lama) 기능.

#### [Version 1] Class Code 기반 (수업 자료 활용)
*   **객체 탐지 및 선택 (Detector & Selector):**
    *   **YOLO 탐지:** `Codes/11_2_[Code]Object_Detection_using_Deep_Learning.pdf`의 **YOLO** 알고리즘을 사용하여 이미지 내의 모든 객체를 1차적으로 탐지한다.
    *   **사용자 선택:** 탐지된 객체들 중 사용자가 **제거를 원하는 객체만 선택(Check)**할 수 있도록 한다.
    *   **수동 ROI 추가:** YOLO가 탐지하지 못한 객체나 특정 영역을 지우고 싶은 경우, 사용자가 마우스 드래그로 **직접 관심 영역(ROI)을 지정**할 수 있는 기능을 제공한다.
*   **객체 제거 (Eraser):**
    *   **영역 추출 (Segmentation):** 사용자가 선택한 객체(BBox) 또는 수동으로 지정한 ROI 영역을 입력으로 받아, `Codes/6_2_[Code]Threshold_InRange_GrabCutd.pdf`의 **GrabCut** 알고리즘을 수행하여 전경(Foreground)을 정밀하게 분리한다.
    *   **배경 복원 (Inpainting):** 추출된 마스크 영역에 대해 OpenCV의 `inpaint` 함수(Navier-Stokes 또는 Telea 알고리즘)를 사용하여 빈 공간을 주변 픽셀로 자연스럽게 채운다.

#### [Version 2] Open Source 기반 (Lama 활용)
*   **객체 탐지 (Detector):**
    *   Version 1과 동일하게 YOLO 탐지 및 사용자 선택/ROI 지정 워크플로우를 따른다.
*   **객체 제거 (Eraser):**
    *   **Lama Model:** 선택된 영역(GrabCut 마스크 또는 BBox)을 **Lama** 모델에 입력하여, 딥러닝 기반으로 자연스럽게 지운다.

### 3.2. 비기능적 요구사항 (Non-Functional Requirements)
*   **개발 환경:** 프로젝트 종속성 관리 및 환경 일관성을 위해 Python **가상 환경(venv 또는 conda)** 사용을 명시한다.
*   **언어:** Python 3.x (OpenCV `cv2`, `numpy`, `tkinter`, `torch/onnx` 등 사용)
*   **반응성:** 사용자 인터랙션(버튼 클릭 등)에 대해 UI가 멈추지 않도록 무거운 작업은 스레드 처리 고려.

## 4. 프로젝트 제약 사항 (Limitations)
본 프로젝트는 프로토타입의 완성도와 효율적인 구현을 위해 다음과 같은 제약 사항을 둔다.

1.  **입력 이미지 크기:**
    *   처리 속도와 메모리 효율을 위해 입력 이미지는 **FHD (1920x1080)** 해상도 이하를 권장한다. 초과 시 자동으로 리사이징될 수 있다.
2.  **탐지 대상 객체 (Target Objects):**
    *   YOLO 모델이 학습한 **COCO Dataset의 80개 클래스** (사람, 차, 자전거, 동물 등)로 탐지 대상을 한정한다.
3.  **제거 대상 객체의 크기 및 위치:**
    *   자연스러운 복원을 위해 제거하려는 객체는 이미지 전체 면적의 **30% 이하**여야 한다.
    *   이미지의 **가장자리(Border)**에 걸쳐 있는 객체는 배경 정보 부족으로 복원 품질이 떨어질 수 있다.
4.  **배경 복잡도:**
    *   Class Code(Inpainting) 방식의 경우, 복잡한 패턴이나 텍스처가 있는 배경보다는 **단순한 배경**이나 **반복적인 패턴**이 없는 배경에서 더 나은 결과를 보장한다.
5.  **단일 이미지 처리:**
    *   연속된 프레임(비디오) 간의 시간적 연관성은 고려하지 않으며, 오직 정지 이미지만을 처리 대상으로 한다.

## 5. 사용자 플로우 (User Flow)

1.  **애플리케이션 시작:**
    *   사용자는 애플리케이션을 실행하고 처리 모드(**Class Code Mode** 또는 **Lama Mode**)를 선택한다. (기본값: Class Code Mode)
2.  **이미지 로드:**
    *   `[Open Image]` 버튼을 클릭하여 로컬 디렉토리에서 이미지를 불러온다.
    *   이미지가 메인 캔버스에 표시된다.
3.  **객체 탐지 (Detection):**
    *   `[Run Detection]` 버튼을 클릭한다.
    *   시스템은 YOLO 모델을 수행하여 객체를 탐지하고, 캔버스에 **Bounding Box**를 그린다.
    *   우측 패널 리스트에 탐지된 객체 목록(예: "Person 99%", "Car 85%")이 생성된다.
4.  **객체 선택 및 수정 (Selection & ROI):**
    *   **객체 선택:** 사용자는 리스트에서 지우고 싶은 객체의 체크박스를 선택한다. 캔버스의 해당 BBox가 하이라이트된다.
    *   **수동 ROI 추가:** 탐지되지 않은 객체를 지우고 싶다면, `[Add Manual ROI]` 버튼을 누르고 마우스 드래그로 영역을 직접 지정한다. 지정된 영역은 제거 대상 목록에 "Manual Region"으로 추가된다.
5.  **객체 제거 (Erase/Inpainting):**
    *   제거 대상을 모두 선택한 후 `[Erase Selected]` 버튼을 클릭한다.
    *   **Class Mode:** GrabCut을 통해 선택된 영역의 정밀 마스크를 생성하고, OpenCV Inpainting 알고리즘으로 배경을 채운다.
    *   **Lama Mode:** 선택된 영역 정보를 Lama 모델에 전달하여 딥러닝 기반으로 배경을 생성한다.
    *   처리 완료 후, 결과 이미지가 캔버스에 업데이트된다.
6.  **결과 확인 및 저장:**
    *   사용자는 결과물을 확인하고 마음에 들면 `[Save Result]` 버튼을 눌러 파일로 저장한다.

## 6. 데이터 및 리소스 (수업 자료 매핑)

| 기능 | 수업 자료 PDF 파일 | 활용 내용 |
| :--- | :--- | :--- |
| **DNN 객체 탐지** | `11_2_[Code]Object_Detection_using_Deep_Learning.pdf` | **YOLOv2** 구현 및 DNN 모듈 사용법 |
| **전경 분리** | `6_2_[Code]Threshold_InRange_GrabCutd.pdf` | **GrabCut**을 이용한 정밀 마스크 생성 |
| **영상 처리 기초** | `3_2_...`, `3_3_...` | 이미지 입출력, ROI 설정, 필터링 기법 |

## 6. UI 디자인 (Tkinter Draft)

*   **Window Title:** "Object Eraser (CV Project)"
*   **Layout:**
    *   **Top:** [Open Image] [Save Result]
    *   **Center:** [Image Canvas] (이미지, BBox, ROI 드래그 기능)
    *   **Right Sidebar:**
        *   **Mode Selection:** (Radio) Class Code Mode / Lama Mode
        *   **Detection Control:** [Run YOLO Detection] -> (List of objects appears)
        *   **Manual Tool:** [Add Manual ROI] Button (Activates drag selection)
        *   **Eraser Control:** [Erase Selected Areas] Button
        *   **Status:** "Ready", "Detection Complete", "Processing GrabCut..."

## 7. 구현 순서 (Implementation Sequence)

1.  **기본 환경 및 UI 구축:**
    *   Tkinter 윈도우 생성, 이미지 로드 및 캔버스 출력 기능 구현.
    *   좌우 패널 레이아웃 구성.
2.  **객체 탐지 모듈 (Class Code - YOLO):**
    *   OpenCV DNN을 이용한 YOLO 모델 연동.
    *   탐지된 객체 목록을 UI 리스트박스에 연동하고 캔버스에 시각화.
3.  **사용자 인터랙션 (Selection & ROI):**
    *   리스트박스에서 특정 객체 선택/해제 기능 구현.
    *   마우스 드래그 이벤트를 활용한 **수동 ROI 지정(사각형 그리기)** 기능 구현.
    *   선택된 영역(YOLO BBox + Manual ROI) 좌표 리스트 관리 로직 작성.
4.  **객체 제거 모듈 1 (Class Code - Inpainting):**
    *   확정된 영역 좌표를 받아 GrabCut 알고리즘 적용 (전경 분리).
    *   마스크 생성 후 OpenCV `inpaint` 함수 적용.
    *   결과 캔버스 업데이트.
5.  **객체 제거 모듈 2 (Open Source - Lama):**
    *   Lama 모델 추론 환경 셋업.
    *   동일한 마스크/ROI를 입력으로 받아 딥러닝 Inpainting 수행.
6.  **통합 및 테스트:**
    *   전체 파이프라인(탐지 -> 선택/추가 -> 제거) 연결.
    *   예외 처리 및 UI 사용성 개선.
