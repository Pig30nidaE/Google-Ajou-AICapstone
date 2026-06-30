# XAI Paper Reproduction2 결과 상세 기록

대상 논문:

`설명가능 인공지능을 활용한 라이프로그 기반 치매 위험도 산정 방법에 관한 연구.pdf`

검토한 결과 폴더:

```text
training/XAI_Paper_Reproduction2/Training/outputs_paper_exact/
```

이 결과는 최신 노트북 형식으로 생성된 결과다. 다음 파일이 존재하므로, 이전 full-fit DRS 방식이 아니라 5-fold out-of-fold SHAP 기반 DRS 방식으로 실행된 결과로 판단한다.

```text
final/oof_shap_values_positive_class.csv
final/full_fit_shap_values_positive_class.csv
final/dementia_risk_score_against_paper.csv
```

## 1. 최종 판정

`paper_exact_audit.json`의 최종 결과는 다음과 같다.

| 항목 | 결과 |
| --- | --- |
| audit 전체 통과 | `false` |
| 실패 check 수 | 1개 |
| 실패 항목 | `paper_forward_best_k_40` |
| 핵심 원인 | 논문은 forward selection 최고점이 top 40이라고 보고했으나, 현재 재현에서는 local best가 top 39로 나옴 |

따라서 현재 결과는 대부분의 논문 핵심 수치를 매우 가깝게 재현했지만, "feature selection의 best feature count까지 완전히 동일해야 한다"는 엄격 기준에서는 완전 통과가 아니다.

다만 최종 모델은 논문 재현 원칙에 따라 top 40 feature를 사용했고, top 40의 ROC-AUC 자체는 논문 수치와 매우 가깝다.

## 2. 핵심 지표 요약

| 지표 | 논문 | 현재 재현 | 차이 |
| --- | ---: | ---: | ---: |
| daily row 수 | 12,183 | 12,183 | 0 |
| subject 수 | 174 | 174 | 0 |
| CN daily row 수 | 7,737 | 7,737 | 0 |
| MCI/Dem daily row 수 | 4,446 | 4,446 | 0 |
| Baseline LightGBM ROC-AUC | 0.9010 | 0.8995 | -0.0015 |
| SHAP forward selection top 40 ROC-AUC | 0.9037 | 0.9026 | -0.0011 |
| SHAP forward selection local best k | 40 | 39 | -1 |
| SHAP forward selection local best ROC-AUC | 0.9037 | 0.9053 | +0.0016 |
| Final LightGBM ROC-AUC | 0.9492 | 0.9479 | -0.0013 |
| DRS CN mean | 7.59 | 7.4240 | -0.1660 |
| DRS MCI/Dem mean | 15.71 | 15.6235 | -0.0865 |

성능 수치 관점에서는 매우 양호하다. 특히 baseline LightGBM, top 40 forward selection, final LightGBM은 모두 논문과 약 0.001 수준의 차이다.

## 3. 전처리 결과

파일:

```text
data/preprocess_summary.json
```

전처리 결과:

| 항목 | 값 |
| --- | ---: |
| rows | 12,183 |
| subjects | 174 |
| train rows | 9,705 |
| validation rows | 2,478 |
| feature count | 86 |
| merge policy | `left_activity` |

class count:

| class | 의미 | row 수 |
| --- | --- | ---: |
| 0 | CN | 7,737 |
| 1 | MCI/Dem | 4,446 |

판정:

- 논문이 제시한 row 수, subject 수, class count와 완전히 일치한다.
- activity daily row를 기준으로 sleep feature를 left join하는 정책이 논문 count를 맞추는 데 적절했다.
- 이 단계는 재현 성공으로 판단한다.

## 4. Baseline 모델 비교

파일:

```text
baselines/model_comparison_against_paper.csv
```

| 모델 | 논문 ROC-AUC | 재현 ROC-AUC | 차이 | 논문 Accuracy | 재현 Accuracy | 차이 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.9010 | 0.8995 | -0.0015 | 0.8262 | 0.8293 | +0.0031 |
| Random forest | 0.8835 | 0.8897 | +0.0062 | 0.8055 | 0.8089 | +0.0034 |
| Multi-Layer Perceptron | 0.6348 | 0.8849 | +0.2501 | 0.5953 | 0.8112 | +0.2159 |
| Support vector machine | 0.6249 | 0.8731 | +0.2482 | 0.6393 | 0.8130 | +0.1737 |
| K-Nearest Neighbor | 0.6595 | 0.8524 | +0.1929 | 0.6572 | 0.7947 | +0.1375 |
| Logistic regression | 0.6067 | 0.6952 | +0.0885 | 0.6457 | 0.6715 | +0.0258 |
| Decision tree | 0.6806 | 0.6707 | -0.0099 | 0.7041 | 0.6944 | -0.0097 |

판정:

- 핵심 모델인 LightGBM은 논문과 매우 가깝다.
- Random Forest와 Decision Tree도 비교적 근접하다.
- MLP, SVM, KNN은 논문보다 훨씬 높게 나왔다.
- 이 차이는 논문이 baseline 모델별 scaling, imputation, hyperparameter, seed를 공개하지 않았기 때문에 생긴 것으로 보는 것이 타당하다.
- 논문과 동일하게 baseline 비교에서 LightGBM이 가장 좋은 모델로 선택되었다.

## 5. SHAP forward feature selection 결과

파일:

```text
feature_selection/forward_selection_metrics.csv
feature_selection/selected_features.json
```

audit 실패 항목:

```text
paper_forward_best_k_40 = FAIL
```

실패 이유:

| 항목 | 논문 | 현재 재현 |
| --- | ---: | ---: |
| best k | 40 | 39 |
| best ROC-AUC | 0.9037 | 0.9053 |
| top 40 ROC-AUC | 0.9037 | 0.9026 |

forward selection 상위 15개 결과:

| 순위 | feature 수 | ROC-AUC | Accuracy | F1 macro |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 39 | 0.9053 | 0.8317 | 0.8097 |
| 2 | 31 | 0.9045 | 0.8310 | 0.8090 |
| 3 | 37 | 0.9043 | 0.8313 | 0.8087 |
| 4 | 59 | 0.9043 | 0.8329 | 0.8105 |
| 5 | 38 | 0.9043 | 0.8311 | 0.8091 |
| 6 | 41 | 0.9040 | 0.8311 | 0.8090 |
| 7 | 36 | 0.9036 | 0.8300 | 0.8074 |
| 8 | 42 | 0.9036 | 0.8312 | 0.8089 |
| 9 | 52 | 0.9029 | 0.8309 | 0.8086 |
| 10 | 32 | 0.9026 | 0.8290 | 0.8067 |
| 11 | 40 | 0.9026 | 0.8301 | 0.8073 |
| 12 | 61 | 0.9023 | 0.8295 | 0.8063 |
| 13 | 35 | 0.9023 | 0.8309 | 0.8091 |
| 14 | 46 | 0.9019 | 0.8289 | 0.8064 |
| 15 | 47 | 0.9018 | 0.8266 | 0.8036 |

35-45개 feature 구간:

| feature 수 | ROC-AUC | Accuracy | F1 macro |
| ---: | ---: | ---: | ---: |
| 35 | 0.9023 | 0.8309 | 0.8091 |
| 36 | 0.9036 | 0.8300 | 0.8074 |
| 37 | 0.9043 | 0.8313 | 0.8087 |
| 38 | 0.9043 | 0.8311 | 0.8091 |
| 39 | 0.9053 | 0.8317 | 0.8097 |
| 40 | 0.9026 | 0.8301 | 0.8073 |
| 41 | 0.9040 | 0.8311 | 0.8090 |
| 42 | 0.9036 | 0.8312 | 0.8089 |
| 43 | 0.9003 | 0.8262 | 0.8034 |
| 44 | 0.9012 | 0.8267 | 0.8031 |
| 45 | 0.9011 | 0.8297 | 0.8072 |

해석:

- top 40의 ROC-AUC는 논문과 거의 일치한다.
- 하지만 local best는 top 39이므로, 논문 Figure 4의 "top 40이 최고"라는 조건은 엄밀히 일치하지 않는다.
- top 39와 top 40의 ROC-AUC 차이는 0.0027 수준이다.
- 최종 모델은 strict mode에 따라 top 40 feature를 사용했으므로, final model reproduction에는 논문 설정을 따른다.

## 6. 선택된 top 40 feature

파일:

```text
feature_selection/selected_features.json
```

최종 모델에 사용된 feature는 논문 기준에 맞춰 top 40으로 고정되었다.

| 순위 | feature |
| ---: | --- |
| 1 | `sleep_breath_average` |
| 2 | `sleep_hr_lowest` |
| 3 | `sleep_hr_average` |
| 4 | `sleep_bedtime_end_hour` |
| 5 | `sleep_rmssd` |
| 6 | `sleep_restless` |
| 7 | `activity_cal_total` |
| 8 | `sleep_score_rem` |
| 9 | `activity_score_training_volume` |
| 10 | `activity_met_min_low` |
| 11 | `activity_score_meet_daily_targets` |
| 12 | `sleep_midpoint_at_delta` |
| 13 | `sleep_score_deep` |
| 14 | `activity_daily_movement` |
| 15 | `sleep_hypnogram_5min_ratio_3` |
| 16 | `activity_class_5min_transition_count` |
| 17 | `activity_met_1min_max` |
| 18 | `activity_score` |
| 19 | `activity_met_min_inactive` |
| 20 | `sleep_light` |
| 21 | `sleep_score_disturbances` |
| 22 | `activity_met_min_medium` |
| 23 | `activity_medium` |
| 24 | `activity_met_1min_std` |
| 25 | `sleep_bedtime_start_hour` |
| 26 | `sleep_hypnogram_5min_transition_count` |
| 27 | `activity_class_5min_count_3` |
| 28 | `sleep_score` |
| 29 | `activity_class_5min_count_2` |
| 30 | `activity_high` |
| 31 | `activity_rest` |
| 32 | `sleep_temperature_delta` |
| 33 | `sleep_hypnogram_5min_ratio_2` |
| 34 | `activity_score_recovery_time` |
| 35 | `sleep_score_alignment` |
| 36 | `activity_score_training_frequency` |
| 37 | `sleep_hypnogram_5min_ratio_1` |
| 38 | `activity_class_5min_ratio_3` |
| 39 | `sleep_total` |
| 40 | `activity_met_min_high` |

논문에서 언급한 주요 feature와의 대응:

| 논문 언급 feature | 현재 SHAP ranking | 최종 full-fit SHAP ranking | 판정 |
| --- | ---: | ---: | --- |
| `sleep_breath_average` | 1 | 1 | 일치 |
| `activity_class_5min_count_3` | 27 | 29 | top 40에 포함 |

## 7. 최종 LightGBM 결과

파일:

```text
final/final_cv_metrics.json
```

최종 모델 설정:

| 항목 | 값 |
| --- | --- |
| feature source | `paper_top40` |
| selected feature count | 40 |
| `min_child_samples` | 41 |
| `num_leaves` | 330 |
| `n_estimators` | 1000 |
| `learning_rate` | 0.08 |

최종 5-fold CV 성능:

| 지표 | 값 |
| --- | ---: |
| Accuracy | 0.8816 |
| ROC-AUC | 0.9479 |
| Precision macro | 0.8819 |
| Recall macro | 0.8598 |
| F1 macro | 0.8688 |
| Positive precision | 0.8828 |
| Positive recall | 0.7791 |
| Positive F1 | 0.8277 |

fold별 ROC-AUC:

| fold | ROC-AUC | Accuracy | F1 macro |
| ---: | ---: | ---: | ---: |
| 0 | 0.9464 | 0.8794 | 0.8648 |
| 1 | 0.9419 | 0.8728 | 0.8587 |
| 2 | 0.9459 | 0.8843 | 0.8713 |
| 3 | 0.9497 | 0.8859 | 0.8749 |
| 4 | 0.9557 | 0.8859 | 0.8740 |

논문과의 비교:

| 항목 | 논문 | 현재 재현 | 차이 |
| --- | ---: | ---: | ---: |
| Final LightGBM ROC-AUC | 0.9492 | 0.9479 | -0.0013 |

판정:

- 최종 LightGBM 성능은 논문과 매우 가깝다.
- 논문 Table 2 파라미터와 최종 feature 수 40을 모두 만족한다.
- 이 부분은 재현 성공으로 판단한다.

## 8. SHAP 해석 결과

파일:

```text
feature_selection/shap_importance_full.csv
final/shap_importance_positive.csv
```

feature selection ranking 상위 30개:

| rank | feature | mean abs SHAP |
| ---: | --- | ---: |
| 1 | `sleep_breath_average` | 1.4804 |
| 2 | `sleep_hr_lowest` | 1.0164 |
| 3 | `sleep_hr_average` | 0.8561 |
| 4 | `sleep_bedtime_end_hour` | 0.8165 |
| 5 | `sleep_rmssd` | 0.7483 |
| 6 | `sleep_restless` | 0.6588 |
| 7 | `activity_cal_total` | 0.5949 |
| 8 | `sleep_score_rem` | 0.5747 |
| 9 | `activity_score_training_volume` | 0.5355 |
| 10 | `activity_met_min_low` | 0.5183 |
| 11 | `activity_score_meet_daily_targets` | 0.4914 |
| 12 | `sleep_midpoint_at_delta` | 0.4815 |
| 13 | `sleep_score_deep` | 0.4782 |
| 14 | `activity_daily_movement` | 0.4615 |
| 15 | `sleep_hypnogram_5min_ratio_3` | 0.4358 |
| 16 | `activity_class_5min_transition_count` | 0.4180 |
| 17 | `activity_met_1min_max` | 0.4135 |
| 18 | `activity_score` | 0.3778 |
| 19 | `activity_met_min_inactive` | 0.3553 |
| 20 | `sleep_light` | 0.3504 |
| 21 | `sleep_score_disturbances` | 0.3458 |
| 22 | `activity_met_min_medium` | 0.3246 |
| 23 | `activity_medium` | 0.3204 |
| 24 | `activity_met_1min_std` | 0.3002 |
| 25 | `sleep_bedtime_start_hour` | 0.2970 |
| 26 | `sleep_hypnogram_5min_transition_count` | 0.2668 |
| 27 | `activity_class_5min_count_3` | 0.2606 |
| 28 | `sleep_score` | 0.2602 |
| 29 | `activity_class_5min_count_2` | 0.2553 |
| 30 | `activity_high` | 0.2503 |

최종 full-fit model SHAP 상위 10개:

| rank | feature | mean abs SHAP |
| ---: | --- | ---: |
| 1 | `sleep_breath_average` | 1.7044 |
| 2 | `sleep_hr_lowest` | 1.0969 |
| 3 | `sleep_hr_average` | 1.0522 |
| 4 | `sleep_rmssd` | 0.9182 |
| 5 | `activity_class_5min_count_2` | 0.8581 |
| 6 | `sleep_bedtime_end_hour` | 0.8499 |
| 7 | `sleep_restless` | 0.7863 |
| 8 | `activity_score_training_volume` | 0.7088 |
| 9 | `activity_cal_total` | 0.6887 |
| 10 | `activity_daily_movement` | 0.6812 |

해석:

- 논문에서 가장 중요하다고 설명한 `sleep_breath_average`가 현재 결과에서도 1위다.
- 상위 feature 대부분이 sleep 관련 변수다.
- 논문에서 행동 관련 주요 변수로 언급한 `activity_class_5min_count_3`도 top 40 안에 포함된다.

## 9. Dementia Risk Score, DRS 결과

파일:

```text
final/dementia_risk_score_summary.json
final/dementia_risk_score_against_paper.csv
final/dementia_risk_scores.csv
```

DRS 산출 방식:

```text
5-fold out-of-fold positive-class SHAP values
```

논문 DRS 요약값과 비교:

| class | 논문 min | 재현 min | 차이 | 논문 max | 재현 max | 차이 | 논문 mean | 재현 mean | 차이 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | 1.06 | 0.5225 | -0.5375 | 24.99 | 24.7072 | -0.2828 | 7.59 | 7.4240 | -0.1660 |
| MCI/Dem | 1.79 | 1.3930 | -0.3970 | 31.28 | 32.4858 | +1.2058 | 15.71 | 15.6235 | -0.0865 |

class별 DRS summary:

| class | count | min | max | mean | std |
| --- | ---: | ---: | ---: | ---: | ---: |
| CN | 7,737 | 0.5225 | 24.7072 | 7.4240 | 2.8815 |
| MCI/Dem | 4,446 | 1.3930 | 32.4858 | 15.6235 | 4.9755 |

daily one-sided t-test:

| 항목 | 값 |
| --- | ---: |
| CN mean | 7.4240 |
| MCI/Dem mean | 15.6235 |
| t statistic | 109.8837 |
| p-value | 0.0 |
| alternative | impaired mean > CN mean |

subject-level 보조 t-test:

| 항목 | 값 |
| --- | ---: |
| CN subject mean | 7.5145 |
| MCI/Dem subject mean | 15.6141 |
| t statistic | 20.1348 |
| p-value | 3.4382e-29 |
| alternative | impaired subject mean > CN subject mean |

판정:

- DRS 방향성은 논문과 일치한다.
- DRS 평균도 논문 보고값과 충분히 가깝다.
- 이전 full-fit DRS 결과보다 논문 DRS 분포에 훨씬 가깝다.
- DRS 재현은 성공으로 판단한다.

## 10. Audit check 상세

| check | 결과 | 해석 |
| --- | --- | --- |
| `paper_row_count` | PASS | row 수 일치 |
| `paper_subject_count` | PASS | subject 수 일치 |
| `paper_class_counts` | PASS | class count 일치 |
| `all_7_models` | PASS | 논문 baseline 7종 모두 실행 |
| `baseline_lightgbm_auc_close` | PASS | LightGBM baseline AUC 근접 |
| `paper_forward_best_k_40` | FAIL | local best가 40이 아니라 39 |
| `paper_forward_top40_auc_close` | PASS | top 40 AUC는 논문과 근접 |
| `paper_selection_mode_uses_top40` | PASS | 최종 모델은 top 40 사용 |
| `paper_selected_feature_count_40` | PASS | 최종 feature 수 40 |
| `final_params_match_paper` | PASS | LightGBM Table 2 파라미터 일치 |
| `final_auc_close` | PASS | 최종 AUC 근접 |
| `drs_source_oof_shap` | PASS | OOF SHAP 기반 DRS 사용 |
| `drs_cn_mean_close_to_paper` | PASS | CN DRS 평균 근접 |
| `drs_impaired_mean_close_to_paper` | PASS | MCI/Dem DRS 평균 근접 |
| `drs_range_close_to_paper` | PASS | DRS min/max 범위 근접 |
| `drs_impaired_mean_gt_cn` | PASS | MCI/Dem 평균 DRS가 CN보다 큼 |
| `drs_p_lt_0_05` | PASS | 단측 t-test 유의 |
| `drs_row_count` | PASS | 전체 12,183 row에 DRS 계산 |

## 11. 보고서에 남길 최종 문장

현재 결과를 보고서에 적을 때는 다음처럼 쓰는 것이 가장 정확하다.

```text
본 재현에서는 논문에서 제시한 daily row 수 12,183건, subject 수 174명,
CN 7,737건, MCI/Dem 4,446건을 정확히 재현하였다.
Baseline LightGBM ROC-AUC는 논문 0.9010 대비 0.8995로 차이 -0.0015였고,
최종 LightGBM 5-fold CV ROC-AUC는 논문 0.9492 대비 0.9479로 차이 -0.0013이었다.

SHAP forward selection에서는 논문이 top 40 feature에서 ROC-AUC 0.9037의 최고 성능을 보고했으나,
본 재현에서는 top 39 feature에서 ROC-AUC 0.9053으로 local best가 나타났다.
다만 top 40 feature의 ROC-AUC는 0.9026으로 논문 수치와 차이 -0.0011에 불과하며,
최종 모델은 논문 재현 원칙에 따라 top 40 feature를 사용하였다.

DRS는 5-fold out-of-fold positive-class SHAP value의 양수 합으로 계산하였다.
CN 평균 DRS는 논문 7.59 대비 7.4240, MCI/Dem 평균 DRS는 논문 15.71 대비 15.6235로 근접하게 재현되었다.
단측 one-sample t-test에서도 MCI/Dem 그룹의 DRS가 CN 그룹보다 유의하게 큰 것으로 나타났다(p=0.0).
```

## 12. 최종 결론

현재 결과는 다음 항목에서 논문 재현에 성공했다.

- 데이터 전처리 count
- baseline LightGBM 성능
- top 40 forward selection 성능 수준
- 최종 LightGBM 파라미터
- 최종 LightGBM ROC-AUC
- SHAP 주요 feature 해석
- DRS 평균과 검정 방향

남은 불일치는 하나다.

- forward selection의 최고점이 논문처럼 k=40이 아니라 k=39다.

따라서 "논문 핵심 결과 재현" 기준으로는 성공에 가깝고, "feature selection best-k까지 완전 동일" 기준으로는 아직 한계가 있다.

