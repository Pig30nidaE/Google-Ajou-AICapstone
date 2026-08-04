| 모델        | 증강   | 원 논문 보고값   |   증거 기반 교정 A5 | 누수 통제 non-nested   | Nested Group CV   |   n_dem_subjects |
|:------------|:-------|:-----------------|--------------------:|:-----------------------|:------------------|-----------------:|
| XGBoost     | 없음   | not_reported     |              0.7885 |                        |                   |                8 |
| XGBoost     | VAE    | 0.8103           |              0.7245 |                        |                   |                8 |
| DNN         | 없음   | not_reported     |              0.93   |                        |                   |                8 |
| DNN         | VAE    | 0.8085           |              0.928  |                        |                   |                8 |
| TabNet      | 없음   | not_reported     |              0.5556 |                        |                   |                8 |
| TabNet      | VAE    | 0.7879           |              0.6107 |                        |                   |                8 |
| Wide & Deep | 없음   | 0.8616           |              0.9616 |                        |                   |                8 |
| Wide & Deep | VAE    | 0.8556           |              0.9397 |                        |                   |                8 |

> '원 논문 보고값'은 논문이 보고한 macro F1이며 **기록 단위**다 (Wide & Deep은 표 6, 나머지 3모델은 그림 3). '증거 기반 교정 A5'·'누수 통제'·'Nested' 열은 모두 **피험자 단위**로 통일했다 — 평가단위를 섞으면 성능 변화의 원인을 구분할 수 없기 때문이다. 논문과 직접 비교 가능한 기록 단위 수치는 paper_comparison_record_level.csv에 있다. A5의 scaled VAE와 KL mean은 논문 미보고 사항에 대한 교정 가정이며 원저자 설정이 아니다. 'not_reported'는 논문이 그 조합을 보고하지 않았다는 뜻이다.

> ⚠️ 본 실험의 Dem 클래스는 독립 피험자 12명에서 유래한다. 합성 Dem 행은 각 fold의 train Dem 피험자 기록 분포에서 생성된 것이며 정확한 source 피험자 수는 fold_composition 및 augmentation diagnostics에 기록된다. 새로운 피험자를 의미하지 않는다.
