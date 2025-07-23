# Wafer Scratch Detection – A‑Contrario Pipeline  
웨이퍼 스크래치 검출 파이프라인
=============================================

목표: a‑contrario 통계 기법을 이용해 고해상도 웨이퍼 이미지에서 거의 수직인 미세 스크래치를 False‑Negative 없이 검출하고, False‑Positive는 최소화한다.

---

## 1  Folder Structure │ 폴더 구조
```
wafer_defect_detection/
├─ io_utils/           # 경로·타이머·I/O 헬퍼
│  ├─ io.py
│  ├─ path.py
│  └─ timing.py
├─ acontrario.py       # 통계적 의미성(NFA) 검증
├─ candidates.py       # 픽셀 후보 마스크 I_B
├─ orientation.py      # Sobel → gradient θ map
├─ seeds.py            # Hough / LSD 선분 시드
├─ grouping.py         # maximality & exclusion
├─ preprocess.py       # Gaussian / CLAHE 전처리
├─ post.py             # skeletonise / overlay
├─ pipeline.py         # 전체 오케스트레이션 (CLI)
├─ cli.py              # 얇은 wrapper (`python -m ...`)
├─ sample_img/         # 샘플 입력 이미지
└─ result/             # <timestamp>/ 마다 출력 저장
```

*Every run builds* `result/20250722_161223/`:

```
result/<run>/
├─ mask.png              # 최종 binary 마스크 (uint8 0/255)
├─ run.txt               # 파라미터 · 타이밍 로그
└─ debug_img/            # 중간 단계 확인용
   ├─ 01_pre_*.png
   ├─ 02_candidates_*.png
   └─ 99_overlay_*.png
```

---

## 2  Processing Pipeline │ 처리 파이프라인

| Stage | File / Function | 목적 (한글) |
|-------|-----------------|-------------|
| 0 | `io_utils.io.load_gray` | 이미지 로딩 및 정규화 |
| 1 | `preprocess.preprocess` | 가우시안·CLAHE로 대비 향상 |
| 2 | `candidates.make_mask` | 스크래치 가능 픽셀 마스크 |
| 3 | `orientation.sobel_orientation` | 그래디언트 방향 계산 |
| 4 | `seeds.hough_seeds` | 후보 선분 추출 (시드) |
| 5 | `acontrario.validate_segments` | 통계적 의미성 검증 |
| 6 | `grouping.select_segments` | 최대 의미 선분 + 중복 제거 |
| 7 | `grouping.segments_to_mask` | 최종 마스크 raster |
| 8 | `post.overlay` | 결과 시각화 (오버레이) |

---

## 3  Usage │ 사용 방법

### CLI

```bash
# run from project root
python -m wafer_defect_detection.pipeline \
       --img wafer_defect_detection/sample_img/sample_1_P.bmp \
       --out mask.png \
       --preprocess clahe \
       --s_med 0.01 --s_avg 0.08 \
       --eps 1.0 \
       --debug
```

### Parameters │ 주요 파라미터

| Flag | Meaning | 설명 |
|------|---------|------|
| `--preprocess` | `gaussian` `clahe` `gauss_clahe` `none` | 전처리 선택 |
| `--s_med` |  | Gaussian‑median 차 임계값 |
| `--s_avg` |  | 좌/우 평균 차 임계값 |
| `--eps` | NFA ≤ ε threshold | 의미성 한계 |
| `--debug` | save step PNGs | 단계별 이미지 저장 |

---

## 4  Extending │ 추후 확장/개선 포인트

* Add new filters: implement in `preprocess.py` and list in parser.
* Different seed detector: add to `seeds.py` (e.g. EDLines).
* Exact Poisson‑binomial: replace `nfa_hoeffding` if speed allows.
* Embed in other tools: call `pipeline.run()` from your code with a NumPy array.

---



---

### Folder Structure │ 폴더 구조 (continued)

File            | 설명
--------------- | --------------------------------------------------------------
io_utils/io.py  | 이미지 로드·저장 및 uint8/float32 변환
io_utils/path.py| 실행별 result/<timestamp>/ 경로 생성 유틸
io_utils/timing.py | Timer, @timeit 데코레이터, 성능 요약
preprocess.py   | Gaussian / CLAHE 전처리 함수 모음
candidates.py   | 스크래치 후보 픽셀 마스크 생성 (I_B)
orientation.py  | Sobel 필터로 그래디언트 방향 계산
seeds.py        | Probabilistic Hough / LSD로 선분 시드 검출
acontrario.py   | NFA 계산으로 의미 있는 선분 판정
grouping.py     | Maximal meaningful 선분 추려서 마스크 생성
post.py         | skeletonise, dilate, overlay 등 후처리
pipeline.py     | 전체 파이프라인 조립 및 CLI 진입점
cli.py          | pipeline.py 를 래핑한 얇은 실행 스크립트

# 
---

## 5  Output Artefacts │ 실행 결과 파일

| 파일/폴더 | 내용 (한글) |
|---------------|-------------|
| `result/<timestamp>/mask.png` | 최종 바이너리 마스크. 흰색 픽셀이 검출된 스크래치, 검은색은 배경. |
| `result/<timestamp>/run.txt` | 입력 이미지, 모든 파라미터, 각 단계별 실행 시간을 기록한 텍스트 로그. |
| `result/<timestamp>/debug_img/` | 디버깅용 중간 단계 이미지가 저장되는 폴더. |

### debug_img/ 내부 PNG 설명

| Index | File name pattern | 설명 (한글) |
|-------|-------------------|-------------|
| 01 | `01_pre_<input>.png` | 전처리(CLAHE/가우시안) 결과 이미지 |
| 02 | `02_candidates_<input>.png` | 후보 픽셀(스크래치 가능 영역)을 흰색으로 표시 |
| 03 | `03_theta_<input>.png` | 그래디언트 방향(0~π)을 0~255 그레이스케일로 시각화 |
| 04 | `04_seeds_<input>.png` | 원본 이미지 위에 파란색으로 표시된 시드 선분 |
| 05 | `05_validated_<input>.png` | 통계적 검증을 통과한 선분을 초록색으로 표시 |
| 06 | `06_mask_<input>.png` | 최종 마스크(흰색=스크래치) |
| 99 | `99_overlay_<input>.png` | 최종 마스크를 빨간색 반투명으로 원본 위에 오버레이 |

---

## 6  Dependencies │ 사용 라이브러리 및 버전

| 라이브러리 | 버전 | 사용 이유(한글) |
|---------|------|------------------|
| Python | ≥ 3.10 | 프로젝트 실행 환경 |
| NumPy | 1.26 | 기본 배열/선형대수 연산 |
| SciPy | 1.11 | 필터·통계 계산 |
| scikit‑image | 0.22 | 이미지 필터와 Hough 변환 |
| OpenCV‑python | 4.12 | 이미지 입출력, CLAHE, 선분 시각화 |
| pylsd2 (optional) | 0.2 | LSD 선분 검출 (선택) |
| tqdm | 4.66 | 진행률 표시 (선택) |
| joblib | 1.4 | 병렬 처리 (선택) |

### Installation │ 설치 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install "numpy==1.26.*" "scipy==1.11.*" "scikit-image==0.22.*" \
            "opencv-python==4.12.*" "tqdm~=4.66" "joblib~=1.4" \
            pylsd2   # add only if you plan to use LSD seeds
```
