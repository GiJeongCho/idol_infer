# RVC Inference API

RVC(Retrieval-based Voice Conversion) 모델을 사용한 음성 변환 FastAPI 서버.

보컬과 배경음악을 입력받아, 보컬에 RVC 음성 변환을 적용한 뒤 배경음악과 믹싱하여 출력합니다.

## 구조

```
infer/
├── main.py            # FastAPI 서버 (엔드포인트 정의)
├── engine.py          # RVCEngine (모델 로딩, 변환, 믹싱)
├── requirements.txt
└── outputs/           # 변환된 오디오 출력 디렉토리
```

## 설치

```bash
pip install -r requirements.txt
```

> **참고:** RVC 핵심 라이브러리는 `Retrieval-based-Voice-Conversion-WebUI` 저장소에 의존합니다.
> 상위 디렉토리에 해당 저장소가 존재해야 합니다.

## 실행

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 python main.py
```

서버가 `http://0.0.0.0:4998`에서 시작됩니다.

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/health` | 서버 상태 확인 |
| `GET` | `/models` | 사용 가능한 모델 목록 |
| `POST` | `/models/{model_name}` | 모델 전환 |
| `POST` | `/convert` | 보컬 + 배경음악 → 변환 및 믹싱 |
| `GET` | `/outputs` | 출력 파일 목록 |
| `GET` | `/outputs/{filename}` | 출력 파일 다운로드 |

### POST `/convert`

보컬과 배경음악 파일을 업로드하면 RVC 변환 후 믹싱된 WAV 파일을 반환합니다.

**파라미터 (multipart/form-data):**

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `vocal` | file | (필수) | 보컬 오디오 파일 |
| `bg` | file | (필수) | 배경음악 오디오 파일 |
| `model` | string | 현재 모델 | RVC 모델 파일명 |
| `f0_up_key` | int | 0 | 피치 변경 (반음 단위, -12 ~ +12) |
| `f0_method` | string | rmvpe | 피치 추출 방법 |
| `vocal_volume` | float | 1.0 | 보컬 볼륨 배율 |
| `bg_volume` | float | 1.0 | 배경음악 볼륨 배율 |
