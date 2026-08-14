# Binary_Google_TabFM_Nested

과제: **CN(0) vs MCI+Dem(1)** — 사람(피험자) 단위 이진 분류.

> **상태: 코드 완성 + 로컬 계약테스트 19개/스텁 smoke 배선 확인만 완료.
> 정식 성능 결과 없음.** 정식 결과는 Colab에서 `run.py` 실행 후
> `FINAL_REPORT.json`에서 확인합니다.

---

## 1. 이 실험의 질문 (사전 고정)

**Google TabFM** — Google Research가 2026-07 공개한 tabular foundation model
(in-context learning, 데이터셋별 학습 없음) — 이 정직한 MMSE anchor(규제
로지스틱 회귀)를 이기는가?

TabPFN 계열 in-context 학습기는 정확히 이 데이터 규모(n<1000)에서 가장
강하다고 알려져 있어, 이 저장소에서 트리/선형 시대 이후 처음으로 시도하는
**진짜 새로운 모델 계열** 가설입니다. 직전 실험(`Binary_Google_CircadianNested`,
run `20260813_060156_utc`)의 결론 — nested 0.7426±0.0220, circadian 기각,
"141명에서는 저용량 모델이 유리" — 위에서, 학습 파라미터가 0개인 foundation
model이 그 교훈의 예외인지를 검정합니다.

## 2. 후보 3개 + 감사 arm 1개 (의도적으로 최소)

| candidate | view | learner | complexity |
|---|---|---|---:|
| `lr_mmse_c001` | mmse | LR(C=0.01) — anchor | 0 |
| `tabfm_mmse` | mmse | **Google TabFM** | 1 |
| `blend_tabfm_lr` | mmse | LR+**TabFM** rank blend (학습 fold ECDF) | 2 |
| `tabfm_fusion` *(고정 감사, 선택 불참)* | mmse_circ | **Google TabFM** | — |

- 선택 규칙: 직전 실험과 동일한 tolerance rule (inner 평균 AUC 0.005 이내면
  complexity 낮은 후보 우선).
- `tabfm_fusion`은 LR/YDF로 기각된 circadian 융합 가설을 **다른 모델 계열로
  재검**하는 감사 arm입니다 (paired 대비: `tabfm_fusion − tabfm_mmse`).

## 3. 직전 실험과의 정합성 (cross-run 검증 장치)

seed(20260813)·라벨 로딩·fold 생성 코드가 `Binary_Google_CircadianNested`와
동일하므로 **outer fold가 repeat 단위로 완전히 일치**합니다. 따라서:

- anchor 트랙 `lr_mmse_c001`은 직전 run의 **0.7617 ± 0.0135**를 재현해야 하며,
  `FINAL_REPORT > anchor_reproduction_check`가 편차를 자동 기록합니다
  (|편차| > 0.02이면 환경 이상으로 간주하고 run을 의심).
- feature fingerprint도 동일해야 합니다 (`b3a0eb0dfd6f8256`).

## 4. TabFM 통합의 방어적 설계 (신생 API 대응)

1. **재현성 핀**: 릴리스 태그 `v1.0.1`
   (commit `d8678b68…`)로 고정 설치. 이미 설치돼 있으면 그 버전을 보고서에 기록.
2. **시그니처 필터링 kwargs**: 문서상 기본 `max_num_rows=100`은 우리 113~141명
   context를 조용히 서브샘플링하므로 **256으로 상향**을 시도하되, 설치된
   `TabFMClassifier` 시그니처가 받는 인자만 전달하고 수락/탈락 목록을
   `FINAL_REPORT > config.google_technology.probe`에 기록.
3. **양성 클래스 열**: `classes_`로 조회 (열 1 가정 금지), 출처 기록.
4. **fail-fast probe**: 특징 구축 전에 합성 데이터로 API 검증 + 실측
   마이크로 벤치마크 → **예상 총 소요시간을 1분 안에 출력** (수십 분 뒤
   크래시/무한 대기 방지).
5. **체크포인트 1회 로드** 후 모든 fold가 공유 (fit은 context 저장일 뿐).
6. **결측**: TabFM의 NaN 처리가 미문서화라 fold-local 중앙값 대치를 어댑터에
   내장 (누수 없음).
7. **fallback 금지**: TabFM 설치/로드 실패 시 다른 모델로 대체하지 않고 중단.
   유일한 예외는 `BGTF_WIRING_STUB=1` + `--profile smoke`에서만 켜지는 배선
   점검용 스텁이며, 모든 산출물에 `NOT_GOOGLE`이 명시되고 다른 프로파일에서는
   즉시 실패합니다.

## 5. 평가 프로토콜 (직전 실험과 동일)

```text
Outer: StratifiedKFold(5) × 10 repeats  (사람 단위, CN/MCI/Dem 층화)
├── Outer-Train (~113명)
│   └── Inner: StratifiedKFold(4) × 2 repeats → 후보 3개 평가·선택·임계값
└── Outer-Test (~28명) → 선택 후보 1회 채점
```

후보별 정직 OOF 트랙, optimism 진단, 사전 등록 paired bootstrap 대비 4건
(`nested/tabfm_mmse/blend − anchor`, `tabfm_fusion − tabfm_mmse`), CN vs MCI
보조 AUC, subject bootstrap 95% CI. 33명 검증은 SHA-256 동결 후 1회 채점.

## 6. 실행 방법

### Colab (`base.ipynb`) — 셀 2만 수정

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary/Binary_Google_TabFM_Nested/run.py"

# 권장: 먼저 quick으로 비용 확인 후 default
# import os; os.environ["BGTF_ARGS"] = "--profile quick"
```

- **런타임: GPU(T4) 필수에 가깝게 권장** — TabFM은 transformer 추론이고 이
  프로토콜은 `default`에서 **967회 fit / 약 67,000행 추론**을 수행합니다.
  CPU 런타임에서는 수 시간~십수 시간이 걸릴 수 있습니다.
- **먼저 `[probe]` 줄을 읽으십시오.** 시작 1분 내에 다음이 출력됩니다.

  ```text
  [probe] fit(113x38) 0.42s | predict(28 rows) 1.10s
  [probe] planned 967 TabFM fits / 66948 predicted rows -> PROJECTED TOTAL 78 min (1.3 h)
  ```

  3시간을 넘기면 경고가 함께 출력됩니다. 이 예측치는 **fit 고정비 + 행당
  추론비** 모델이며, blend 후보가 ECDF를 만들려고 자기 학습 context(~85행)를
  한 번 더 추론하는 비용까지 포함합니다. (2026-08-14 이전 버전은 fit 개수만
  세어 **약 2.4배 과소** 예측했습니다.)
- **진행 상황 확인**: outer fold마다 한 줄씩, 그리고 fold 내부에서도 60초마다
  `[progress] n/N model fits (xx.x%) | elapsed | ETA`가 출력됩니다. 결과
  디렉터리의 `PROGRESS.json`을 다른 셀에서 열어보면 실행을 방해하지 않고
  진척도를 확인할 수 있습니다.
- **비용이 과하면**: `--profile quick`(5×2 repeats)은 `default`의 약 1/5
  비용이며 프로토콜 구조는 동일합니다.
- TabFM(태그 v1.0.1)은 없으면 자동 설치되고, 첫 로드 시 Hugging Face에서
  가중치를 내려받습니다 (가중치 라이선스: 비상업·연구용 — 캡스톤 사용 적합).
- 프로파일: `quick`(5×2, 타당성) → `default`(5×10, 정식) → `max`(5×20).
- 결과: `/content/drive/MyDrive/Binary_Google_TabFM_Nested_result/<UTC_RUN_ID>/`.

### 로컬 (TabFM 설치 불가 환경)

```bash
python -m pytest tests/ -q                              # 계약 테스트 19개
BGTF_WIRING_STUB=1 python run.py --profile smoke        # 배선 확인(NOT GOOGLE)
```

## 7. 산출물과 판독 기준

산출물 구조는 직전 실험과 동일 (`FINAL_REPORT.json`이 source of truth).
실행 후 확인 순서:

1. `engine` — `google_tabfm`인지 (스텁이면 전체 무효), backend, 버전, pin 준수.
2. `anchor_reproduction_check.consistent_within_0p02` — False면 run 의심.
3. **Primary**: `nested_oof.repeat_roc_auc_mean ± sd` — 직전 nested 0.7426,
   anchor 0.7617과 비교.
4. `paired_contrasts_subject_mean.tabfm_mmse__minus__lr_mmse_c001` —
   **CI가 0을 포함하면 TabFM 이득은 "확인 안 됨"** (관찰 부호만 기록).
5. `selection_counts` — inner CV가 실제로 TabFM을 선택했는지.
6. `tabfm_fusion__minus__tabfm_mmse` — circadian 기각의 모델-독립성.
7. `config.google_technology.probe.kwargs_dropped`에 `max_num_rows`가 있으면
   context가 100명으로 잘렸다는 뜻이므로 해석에 명시.
8. Validation 수치는 disclaimer와 함께만 인용.

## 8. Google 기술의 역할

**Google TabFM이 검정 대상 그 자체**입니다: 단독 후보(`tabfm_mmse`), blend의
절반, 감사 arm(`tabfm_fusion`)의 유일한 학습 엔진. 저장소 최초의 foundation
model 계열 실험이며, 앞서 `Binary_Google_CircadianNested`에서 TabFM을 후보에서
제외했던 사유(소스 설치·HF 다운로드의 재현성 리스크)는 본 폴더의 태그 핀 +
probe + 스텁 격리 설계로 해소했습니다.

---

*연구용 분류 실험이며 의료 진단 도구가 아닙니다. smoke/스텁 결과는 어떤
문서에도 성능으로 인용하지 않습니다.*
