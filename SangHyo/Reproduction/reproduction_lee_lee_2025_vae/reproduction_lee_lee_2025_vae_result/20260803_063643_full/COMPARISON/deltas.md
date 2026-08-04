| delta_type   | scope        | model     | metric            |   baseline |   compared |   delta |
|:-------------|:-------------|:----------|:------------------|-----------:|-----------:|--------:|
| vae_effect   | experiment_A | xgboost   | macro_f1          |     0.7885 |     0.7245 | -0.064  |
| vae_effect   | experiment_A | xgboost   | balanced_accuracy |     0.7247 |     0.6544 | -0.0703 |
| vae_effect   | experiment_A | xgboost   | dem_recall        |     0.625  |     0.375  | -0.25   |
| vae_effect   | experiment_A | xgboost   | dem_f1            |     0.7692 |     0.5455 | -0.2238 |
| vae_effect   | experiment_A | dnn       | macro_f1          |     0.93   |     0.928  | -0.002  |
| vae_effect   | experiment_A | dnn       | balanced_accuracy |     0.894  |     0.8905 | -0.0034 |
| vae_effect   | experiment_A | dnn       | dem_recall        |     0.75   |     0.75   |  0      |
| vae_effect   | experiment_A | dnn       | dem_f1            |     0.8571 |     0.8571 |  0      |
| vae_effect   | experiment_A | tabnet    | macro_f1          |     0.5556 |     0.6107 |  0.0551 |
| vae_effect   | experiment_A | tabnet    | balanced_accuracy |     0.5172 |     0.5657 |  0.0486 |
| vae_effect   | experiment_A | tabnet    | dem_recall        |     0.375  |     0.5    |  0.125  |
| vae_effect   | experiment_A | tabnet    | dem_f1            |     0.5455 |     0.6667 |  0.1212 |
| vae_effect   | experiment_A | wide_deep | macro_f1          |     0.9616 |     0.9397 | -0.022  |
| vae_effect   | experiment_A | wide_deep | balanced_accuracy |     0.9387 |     0.9036 | -0.0351 |
| vae_effect   | experiment_A | wide_deep | dem_recall        |     0.875  |     0.75   | -0.125  |
| vae_effect   | experiment_A | wide_deep | dem_f1            |     0.9333 |     0.8571 | -0.0762 |

> ⚠️ 본 실험의 Dem 클래스는 독립 피험자 12명에서 유래한다. 합성 Dem 행은 각 fold의 train Dem 피험자 기록 분포에서 생성된 것이며 정확한 source 피험자 수는 fold_composition 및 augmentation diagnostics에 기록된다. 새로운 피험자를 의미하지 않는다. A→B delta는 행 분할→피험자 분할, all-data→train-only fit, 평가 이상치 보존 등 검증 프로토콜 변경의 결합 차이다. 단일 누수 요인의 인과효과로 해석할 수 없다.
