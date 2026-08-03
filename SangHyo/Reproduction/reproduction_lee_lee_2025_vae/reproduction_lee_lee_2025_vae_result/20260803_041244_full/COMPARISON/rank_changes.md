| experiment   | augmentation   | model     |   macro_f1 |   rank |
|:-------------|:---------------|:----------|-----------:|-------:|
| A            | none           | dnn       |     0.93   |      1 |
| A            | none           | wide_deep |     0.9288 |      2 |
| A            | none           | xgboost   |     0.7885 |      3 |
| A            | none           | tabnet    |     0.5311 |      4 |
| A            | vae            | wide_deep |     0.9756 |      1 |
| A            | vae            | dnn       |     0.9413 |      2 |
| A            | vae            | xgboost   |     0.7661 |      3 |
| A            | vae            | tabnet    |     0.5972 |      4 |
| B            | none           | xgboost   |     0.3705 |      1 |
| B            | none           | dnn       |     0.2885 |      2 |
| B            | none           | wide_deep |     0.2702 |      3 |
| B            | none           | tabnet    |     0.2596 |      4 |
| B            | vae            | xgboost   |     0.3611 |      1 |
| B            | vae            | tabnet    |     0.3118 |      2 |
| B            | vae            | wide_deep |     0.2875 |      3 |
| B            | vae            | dnn       |     0.2618 |      4 |
