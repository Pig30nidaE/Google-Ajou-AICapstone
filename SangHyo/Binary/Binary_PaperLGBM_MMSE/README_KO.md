# Binary_PaperLGBM_MMSE (논문 재현 + MMSE · 누수 방지)

## 한 줄 요약

`Binary_PaperLGBM_NoMMSE`(논문 재현, 웨어러블만)에 **MMSE 인지검사 점수를 더한**
버전입니다. 논문의 풍부한 특징 공학 + LightGBM을 쓰되, **데이터 누수를 방지하여
사람 단위로 정직하게** 평가합니다.

## 배경 — 논문의 0.9는 "하루 단위 K-fold"의 subject leakage

논문(천희웅 외, 2025)은 174명이 아니라 **12,183개 하루-기록**을 무작위 5-fold로
나눠 평가합니다. 같은 사람의 다른 날이 train/val에 동시에 들어가 ROC-AUC가
0.90(튜닝 0.949)까지 부풀려집니다. 논문도 "**사람 단위가 아닌 하루 단위**로 모델을
만들었다"고 인정합니다. 사람 단위로 평가하면 ROC-AUC ≈0.71로 내려갑니다.

## MMSE를 더하면 누수가 더 극단적으로 드러납니다

MMSE 점수는 **사람당 하나의 상수**입니다. 이걸 그 사람의 모든 날에 복제해 넣고
**하루 단위 무작위 K-fold**로 평가하면, 모델은 train에서 본 사람의 MMSE를 val에서
그대로 만나 **사람을 거의 완벽히 외웁니다** → ROC-AUC가 **1.0에 근접**. 이는 누수의
위험성을 가장 극명하게 보여주는 사례입니다. 반면 **사람 단위(누수 방지)** 로 보면
MMSE는 실제로 도움이 되어 웨어러블만보다 ROC-AUC가 올라가지만(≈0.70) 0.9에는
못 미칩니다(정상처럼 보이는 MCI 때문 — 다른 실험들과 동일 결론).

## 이 폴더가 하는 일

1. **정직한 본 지표(누수 방지)**: 논문 특징 + MMSE를 **사람당 1행으로 집계**해
   LightGBM으로 반복 사람 단위 Stratified K-fold OOF ROC-AUC + 검증 33명 held-out.
2. **누수 진단(참고, 명확 표시)**: 같은 하루 특징(+MMSE 복제)을 무작위 K-fold
   (누수 → ~1.0)와 GroupKFold(정직)로 비교.

## 사용 특징

- 웨어러블: 논문 Figure 2 재현(하루 스칼라 + intraday MET 통계 + 활동클래스/수면단계
  개수)을 사람별 평균·표준편차로 집계.
- **MMSE(사람당 상수)**: 총점 + 영역 점수(지남력·주의·**지연회상**·언어) + 핵심 문항.
  진단명(`DIAG_NM`)·진단순서(`DIAG_SEQ`)·행정 메타는 fail-closed 제외.

## 목표와 정직한 기대치

- 목표: **ROC-AUC ≥ 0.90, accuracy ≥ 0.80**.
- **정직한(누수 방지) 사람 단위 ROC-AUC는 약 0.70**으로 목표 0.9 미달. 0.9~1.0은
  오직 하루 단위 무작위 K-fold(누수)에서만 나오며 유효한 성능이 아닙니다.
- MMSE는 웨어러블만보다 정직한 성능을 확실히 올리지만, 데이터 한계(만점 MMSE인
  MCI)로 0.9는 넘지 못합니다.

## 실행 (Colab, base.ipynb)

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_PaperLGBM_MMSE/run.py"
```

- **GPU 불필요, CPU만.** `lightgbm` 자동 설치.
- 산출물: `training/FINAL_REPORT.json`
  (`leakage_free_subject_oof_roc_auc` = 정직 본 지표, `leakage_diagnostic` = 누수 대비).

이 코드는 연구용이며 의료 진단 도구가 아닙니다.
