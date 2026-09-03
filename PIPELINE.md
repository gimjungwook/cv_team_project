# Object Eraser 파이프라인

## 전체 시스템 아키텍처

```mermaid
flowchart TB
    subgraph INPUT["1️⃣ 입력 단계"]
        A[("🖼️ 이미지 파일<br/>(JPG/PNG)")] --> B["cv2.imread()"]
        B --> C["이미지 검증"]
        C -->|성공| D["self.original<br/>self.result"]
        C -->|실패| E["❌ ValueError<br/>프로그램 종료"]
    end

    subgraph WINDOW["2️⃣ UI 초기화"]
        D --> F["OpenCV 윈도우 생성<br/>(1600x900)"]
        F --> G["메인 이벤트 루프<br/>cv2.waitKey()"]
    end

    subgraph KEYMAP["3️⃣ 키보드 이벤트 처리"]
        G --> H{키 입력}
        H -->|d| I["YOLO 탐지"]
        H -->|r| J["ROI 수동 선택"]
        H -->|t/c/l| K["모드 변경"]
        H -->|e| L["객체 제거"]
        H -->|s| M["결과 저장"]
        H -->|z| N["원본 복원"]
        H -->|q/ESC| O["프로그램 종료"]
    end

    I --> G
    J --> G
    K --> G
    L --> G
    M --> G
    N --> G

    style INPUT fill:#e1f5fe
    style WINDOW fill:#f3e5f5
    style KEYMAP fill:#fff3e0
```

---

## 상세 파이프라인

### 1. YOLO 객체 탐지 파이프라인 (키: `d`)

```mermaid
flowchart LR
    subgraph LOAD["모델 로딩 (최초 1회)"]
        A1["models/*.cfg"] --> B1["cv2.dnn.readNetFromDarknet()"]
        A2["models/*.weights"] --> B1
        A3["models/*.names"] --> C1["클래스명 리스트"]
        B1 --> D1["YOLODetector"]
    end

    subgraph DETECT["탐지 프로세스"]
        E1["원본 이미지<br/>(H×W×3)"] --> F1["Blob 변환<br/>416×416<br/>scale=1/255"]
        F1 --> G1["Forward Pass<br/>net.forward()"]
        G1 --> H1["Raw Detections"]
    end

    subgraph POST["후처리"]
        H1 --> I1["Confidence Filter<br/>(threshold=0.5)"]
        I1 --> J1["좌표 변환<br/>center → corner"]
        J1 --> K1["NMS<br/>(threshold=0.4)"]
        K1 --> L1["DetectionResult[]"]
    end

    D1 --> E1
    L1 --> M1["self.detections<br/>화면에 초록색 박스"]

    style LOAD fill:#c8e6c9
    style DETECT fill:#bbdefb
    style POST fill:#ffe0b2
```

### 2. 수동 ROI 선택 파이프라인 (키: `r`)

```mermaid
flowchart LR
    A["'r' 키 입력"] --> B["cv2.selectROI()<br/>새 윈도우 열림"]
    B --> C["🖱️ 마우스 드래그<br/>영역 선택"]
    C --> D{확정?}
    D -->|Enter/Space| E["ROI 좌표<br/>(x, y, w, h)"]
    D -->|c 키| F["선택 취소"]
    E --> G{크기 검증<br/>w>0 & h>0}
    G -->|통과| H["self.rois.append()<br/>주황색 박스 표시"]
    G -->|실패| F
    F --> I["메인 루프 복귀"]
    H --> I

    style A fill:#fff9c4
    style H fill:#c8e6c9
    style F fill:#ffcdd2
```

### 3. 마스크 생성 파이프라인

```mermaid
flowchart TB
    subgraph INIT["초기화"]
        A["빈 마스크 생성<br/>np.zeros((H,W), uint8)"]
        B["박스 목록 병합<br/>detections + rois"]
    end

    subgraph GRABCUT["GrabCut 처리 (각 박스별)"]
        C["경계 보정<br/>x1,y1,x2,y2"]
        D{크기 검사<br/>10×10 이상?}
        D -->|Yes| E["GrabCut 실행<br/>cv2.grabCut()<br/>iterations=3"]
        D -->|No| F["단순 채우기<br/>mask[y1:y2,x1:x2]=255"]
        E --> G["전경 추출<br/>GC_FGD | GC_PR_FGD"]
        G --> H["마스크 병합<br/>bitwise_or()"]
        F --> H
    end

    subgraph DILATE["마스크 후처리"]
        I["Dilation<br/>kernel=5×5<br/>iterations=2"]
        J["최종 Binary Mask"]
    end

    A --> B
    B --> C
    H --> I
    I --> J

    style INIT fill:#e3f2fd
    style GRABCUT fill:#f3e5f5
    style DILATE fill:#e8f5e9
```

### 4. 인페인팅 파이프라인 (키: `e`)

```mermaid
flowchart TB
    A["'e' 키 입력"] --> B{영역 존재?<br/>detections or rois}
    B -->|No| C["경고 메시지<br/>메인 루프 복귀"]
    B -->|Yes| D["마스크 생성<br/>_create_mask()"]

    D --> E{모드 선택}

    E -->|telea| F["Telea 직접 구현"]
    E -->|telea_cv| G["cv2.inpaint()"]
    E -->|lama| H["LaMa ONNX"]

    subgraph TELEA["MODE: Telea 직접 구현 (키: t)"]
        F --> F1["Fast Marching Method"]
        F1 --> F2["Priority Queue<br/>경계→내부 순서"]
        F2 --> F3["픽셀별 가중 평균<br/>w_dist × w_dir × w_level"]
        F3 --> F4["Eikonal 방정식<br/>거리 계산"]
    end

    subgraph CV2["MODE: cv2.inpaint() (키: c)"]
        G --> G1["OpenCV 내장 함수<br/>C++ 최적화"]
        G1 --> G2["INPAINT_TELEA<br/>radius=5"]
    end

    subgraph LAMA["MODE: LaMa (키: l)"]
        H --> H1["모델 로딩<br/>lama_fp32.onnx"]
        H1 --> H2["전처리<br/>512×512 resize<br/>BGR→RGB<br/>정규화"]
        H2 --> H3["ONNX 추론<br/>session.run()"]
        H3 --> H4["후처리<br/>원본 크기 복원<br/>RGB→BGR"]
    end

    F4 --> I["결과 이미지"]
    G2 --> I
    H4 --> I

    I --> J["self.result 업데이트<br/>detections, rois 초기화"]
    J --> K["메인 루프 복귀<br/>Before|After 표시"]

    style TELEA fill:#fff3e0
    style CV2 fill:#e8f5e9
    style LAMA fill:#fce4ec
```

---

## Telea 알고리즘 상세

```mermaid
flowchart TB
    subgraph INIT["1. 초기화"]
        A["flag 맵 생성"] --> A1["KNOWN(0): 원본 픽셀"]
        A --> A2["UNKNOWN(2): 마스크 내부"]
        B["dist 맵 생성"] --> B1["KNOWN: 0"]
        B --> B2["UNKNOWN: ∞"]
    end

    subgraph BOUNDARY["2. 경계 탐색"]
        C["UNKNOWN이면서<br/>KNOWN과 인접"] --> D["BAND(1)로 설정"]
        D --> E["Priority Queue에 추가<br/>(distance, y, x)"]
    end

    subgraph FMM["3. Fast Marching Loop"]
        F["heappop()<br/>가장 가까운 BAND 픽셀"] --> G["픽셀 복원"]

        G --> G1["반경 내 KNOWN 픽셀 탐색"]
        G1 --> G2["가중치 계산"]

        subgraph WEIGHT["가중치 공식"]
            W1["w_dist = 1/(d² + ε)<br/>거리 역수"]
            W2["w_dir = max(0, -dot + 1)<br/>방향 일치도"]
            W3["w_level = 1/(1 + Δlevel)<br/>등고선 유사도"]
            W1 --> W4["weight = w_dist × w_dir × w_level"]
            W2 --> W4
            W3 --> W4
        end

        G2 --> W4
        W4 --> G3["pixel = Σ(w × neighbor) / Σ(w)"]

        G3 --> H["flag = KNOWN"]
        H --> I["이웃 UNKNOWN → BAND<br/>Eikonal 거리 계산"]
        I --> J{Queue 비었음?}
        J -->|No| F
        J -->|Yes| K["완료"]
    end

    INIT --> BOUNDARY
    BOUNDARY --> FMM

    style INIT fill:#e3f2fd
    style BOUNDARY fill:#fff9c4
    style FMM fill:#f3e5f5
    style WEIGHT fill:#e8f5e9
```

---

## LaMa 추론 파이프라인

```mermaid
flowchart LR
    subgraph INPUT["입력"]
        A["원본 이미지<br/>(H×W×3, BGR)"]
        B["마스크<br/>(H×W, uint8)"]
    end

    subgraph PREPROCESS["전처리"]
        C["스케일 계산<br/>scale = 512/max(H,W)"]
        D["리사이즈<br/>(new_h, new_w)"]
        E["패딩<br/>512×512<br/>mode='reflect'"]
        F["BGR → RGB"]
        G["정규화<br/>[0,255] → [0,1]"]
        H["차원 변환<br/>HWC → NCHW"]
    end

    subgraph INFERENCE["ONNX 추론"]
        I["img: (1,3,512,512)"]
        J["mask: (1,1,512,512)"]
        K["session.run()"]
        L["output: (1,3,512,512)"]
    end

    subgraph POSTPROCESS["후처리"]
        M["NCHW → HWC"]
        N["클리핑 [0,255]<br/>→ uint8"]
        O["패딩 제거"]
        P["원본 크기 리사이즈"]
        Q["RGB → BGR"]
    end

    A --> C
    B --> C
    C --> D --> E --> F --> G --> H
    H --> I
    B --> J
    I --> K
    J --> K
    K --> L --> M --> N --> O --> P --> Q
    Q --> R["결과 이미지<br/>(H×W×3, BGR)"]

    style INPUT fill:#bbdefb
    style PREPROCESS fill:#c8e6c9
    style INFERENCE fill:#ffe0b2
    style POSTPROCESS fill:#f8bbd9
```

---

## 사용자 흐름 (User Flow)

```mermaid
flowchart TB
    START((시작)) --> A["python app.py image.jpg"]
    A --> B["이미지 로드 & 윈도우 표시"]

    B --> C{어떤 작업?}

    C -->|객체 탐지| D["'d' 키: YOLO 자동 탐지"]
    C -->|수동 선택| E["'r' 키: ROI 드래그"]
    C -->|모드 변경| F["'t/c/l' 키"]
    C -->|제거 실행| G["'e' 키: 인페인팅"]
    C -->|저장| H["'s' 키: result.png"]
    C -->|초기화| I["'z' 키: 원본 복원"]
    C -->|종료| J["'q' 키"]

    D --> K["초록색 박스 표시"]
    E --> L["주황색 박스 표시"]
    K --> C
    L --> C

    F --> M["모드 표시 변경<br/>Telea/cv2/LaMa"]
    M --> C

    G --> N["마스크 생성 → 인페인팅"]
    N --> O["Before | After 표시"]
    O --> C

    H --> P["파일 저장 완료"]
    P --> C

    I --> Q["모든 작업 초기화"]
    Q --> C

    J --> END((종료))

    style START fill:#4caf50,color:#fff
    style END fill:#f44336,color:#fff
    style D fill:#8bc34a
    style E fill:#ff9800
    style G fill:#2196f3
```

---

## 인페인팅 모드 비교

```mermaid
flowchart LR
    subgraph COMPARE["인페인팅 모드 비교"]
        direction TB

        subgraph T["Telea 직접 구현 (t)"]
            T1["✅ 알고리즘 학습용"]
            T2["✅ 커스터마이징 가능"]
            T3["❌ Python 구현 (느림)"]
            T4["📊 소규모 영역 적합"]
        end

        subgraph C["cv2.inpaint() (c)"]
            C1["✅ C++ 최적화 (빠름)"]
            C2["✅ 추가 모델 불필요"]
            C3["❌ 대규모 영역 품질↓"]
            C4["📊 소~중규모 영역 적합"]
        end

        subgraph L["LaMa (l)"]
            L1["✅ 최고 품질"]
            L2["✅ 대규모 마스크 강점"]
            L3["❌ ONNX 모델 필요"]
            L4["❌ GPU 권장"]
            L5["📊 대규모 영역 적합"]
        end
    end

    style T fill:#fff3e0
    style C fill:#e8f5e9
    style L fill:#fce4ec
```

---

## 모듈 의존성 그래프

```mermaid
flowchart TB
    subgraph APP["app.py (ObjectEraser)"]
        A1["run()"]
        A2["_detect()"]
        A3["_add_roi()"]
        A4["_create_mask()"]
        A5["_erase()"]
        A6["_save()"]
    end

    subgraph DETECT["core/detect.py"]
        D1["YOLODetector"]
        D2["DetectionResult"]
        D3["load_default_detector()"]
    end

    subgraph INPAINT["core/inpaint.py"]
        I1["TeleaInpainter"]
        I2["telea_inpaint()"]
        I3["cv_inpaint()"]
    end

    subgraph LAMA["core/lama.py"]
        L1["LamaInpainter"]
        L2["load_lama_model()"]
    end

    subgraph MODELS["models/"]
        M1["yolov3.cfg"]
        M2["yolov3.weights"]
        M3["coco.names"]
        M4["lama/lama_fp32.onnx"]
    end

    A2 --> D3
    D3 --> D1
    D1 --> D2

    A5 --> I2
    A5 --> I3
    A5 --> L2

    I2 --> I1
    L2 --> L1

    D1 -.-> M1
    D1 -.-> M2
    D1 -.-> M3
    L1 -.-> M4

    style APP fill:#e3f2fd
    style DETECT fill:#c8e6c9
    style INPAINT fill:#fff3e0
    style LAMA fill:#fce4ec
    style MODELS fill:#f5f5f5
```

---

## 데이터 구조

```mermaid
classDiagram
    class ObjectEraser {
        +ndarray original
        +ndarray result
        +str mode
        +List~DetectionResult~ detections
        +List~Tuple~ rois
        +YOLODetector detector
        +LamaInpainter lama
        +run()
        +_detect()
        +_add_roi()
        +_create_mask()
        +_erase()
        +_save()
    }

    class DetectionResult {
        +str label
        +float confidence
        +Tuple~int~ bbox
    }

    class YOLODetector {
        +Net net
        +List~str~ names
        +float conf_threshold
        +float nms_threshold
        +detect(image) List~DetectionResult~
    }

    class TeleaInpainter {
        +int radius
        +float epsilon
        +inpaint(image, mask) ndarray
        -_init_boundary()
        -_fast_marching()
        -_inpaint_pixel()
        -_compute_distance()
    }

    class LamaInpainter {
        +int MODEL_SIZE
        +InferenceSession session
        +inpaint(image, mask) ndarray
        -_preprocess()
        -_postprocess()
    }

    ObjectEraser --> DetectionResult : contains
    ObjectEraser --> YOLODetector : uses
    ObjectEraser --> TeleaInpainter : uses
    ObjectEraser --> LamaInpainter : uses
```

---

## 실행 예시

```bash
# 1. 가상환경 활성화
source .venv/bin/activate

# 2. 애플리케이션 실행
python app.py photo.jpg

# 3. 키보드 조작
# d → 객체 자동 탐지
# r → ROI 수동 선택
# t/c/l → 모드 변경
# e → 제거 실행
# s → 저장
# z → 복원
# q → 종료
```
