"""전역 Gemma feature-program을 위한 고정 prompt.

논문의 아이디어는 다음처럼 누수 없는 형태로만 사용한다.

* DeepFeature: LLM은 의미 있는 조합을 *설명*하고, 수치 계산은 검증된
  deterministic DSL이 담당한다.
* Rubric representation learning: 모든 사람에게 동일한 global rubric/program을
  적용한다. 라벨 예시에서 rubric을 학습하는 변형은 사용하지 않는다.
* FeatLLM: 선언적 규칙이라는 표현 방식만 취하고 class-conditioned few-shot
  example, feature bagging, sample value는 제공하지 않는다.
* Concept activation: 하나의 출력은 해석 가능한 행동 concept를 나타내지만,
  의료 판단이나 직접 예측은 만들지 않는다.

Gemma에 전달되는 가변 정보는 primitive catalog의 이름, 영역, 설명, 단위뿐이다.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from .program_schema import (
    MAX_DEPENDENCIES,
    MAX_PROGRAM_FEATURES,
    MIN_DEPENDENCIES,
    MIN_PROGRAM_FEATURES,
    MISSING_POLICY,
    OPERATIONS,
    PROGRAM_VERSION,
    PrimitiveSpec,
    catalog_payload,
)

__all__ = [
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "prompt_hash",
    "render_user_prompt",
]

SYSTEM_PROMPT = f"""\
You design one reusable, global feature program for de-identified wrist-wearable
summary variables. You never score an individual and you never see rows,
examples, outcomes, class assignments, cognitive-test values, diagnoses, or
identifiers.

Scientific design context only: the downstream research studies subtle
age-related change in everyday functioning. Wearable literature motivates
behaviour-only concepts around circadian fragmentation, routine stability,
sleep continuity and architecture, activity fragmentation, nocturnal autonomic
regulation, and cross-domain consistency. This context is not a request for a
medical feature or an outcome proxy; feature names and rationales must stay
strictly at the observable-behaviour level and must not mention the research
outcome.

Your role is semantic feature design, not numerical execution:
- Use the supplied metadata catalog only.
- Create behaviour concepts that can be applied identically to every future row.
- Prefer complementary concepts spanning activity, sleep, circadian organisation,
  temporal stability, autonomic dynamics, and cross-domain consistency when
  those domains are available.
- A concept is descriptive. It must not be a medical conclusion, risk estimate,
  probability, category, alert, or proxy for an unavailable clinical variable.
- Do not ask for examples or values. Do not infer population norms.
- Do not emit Python, pseudocode, formulas outside the declared DSL, thresholds,
  cutoffs, learned coefficients, custom constants, or custom weights.
- Use only these operations: {", ".join(OPERATIONS)}.
- Every feature uses {MIN_DEPENDENCIES}--{MAX_DEPENDENCIES} distinct catalog
  dependencies and one direction (+1 or -1) per dependency.
- Direction means semantic orientation only: +1 uses the standardized primitive
  as written; -1 reverses it. Never choose direction from outcome association.
- Every missing_policy is exactly {MISSING_POLICY}.
- Produce {MIN_PROGRAM_FEATURES}--{MAX_PROGRAM_FEATURES} non-duplicate features.
- Every feature name starts with `llmfp__` and contains lowercase letters,
  digits and underscores only.
- Every rationale is 20--240 characters.
- Feature names and rationales must not contain diagnosis names or
  abbreviations, cognition/decline/impairment terms, clinical/medical/screening
  terms, risk/probability/prediction terms, labels, outcomes, classes, or
  identifiers.
- Return strict JSON matching the exact output contract below. Return no
  markdown, code fence, commentary, or extra key.

The deterministic engine, not you, defines execution. It clips each fold-local
standardized z value to [-5, 5], then evaluates:
- signed_mean: tanh(mean(direction_i * z_i))
- signed_product: product(tanh(direction_i * z_i))
- absolute_gap: tanh(max(direction_i * z_i) - min(direction_i * z_i))

All imputation, clipping statistics, and scaling are fitted outside this prompt
on training folds only. Your output may not alter that policy.
"""

USER_PROMPT_TEMPLATE = f"""\
Create exactly one global `{PROGRAM_VERSION}` feature program from the metadata
catalog below.

Design guidance:
1. Treat the catalog as a global rubric shared unchanged by all rows.
2. Translate semantically plausible wearable relationships into the restricted
   declarative operations. The executable engine will perform all arithmetic.
3. Make each rationale a short explanation of the behavioural relationship,
   based only on names, domains, descriptions, and units in the catalog.
4. Use several complementary dependency sets rather than renaming the same
   executable definition.
5. Never request or assume a row value, example, outcome, population reference,
   cognitive-test variable, diagnosis, or identifier.

Exact JSON output contract:
- The top-level object has exactly `program_version` and `features`.
- `program_version` is exactly `{PROGRAM_VERSION}`.
- Every item in `features` has exactly `name`, `operation`, `dependencies`,
  `directions`, `missing_policy`, and `rationale`.
- `dependencies` is an array containing only exact names from the supplied
  catalog. `directions` is an equally long array containing only -1 or 1.
- The feature-count, dependency-count, allowed-operation, name, rationale, and
  missing-policy constraints from the system instruction are mandatory.

Primitive catalog JSON:
{{catalog_json}}

Return only the schema-conformant JSON object. It must have exactly two top-level
keys: `program_version` and `features`.
"""


def render_user_prompt(catalog: Sequence[PrimitiveSpec]) -> str:
    """Render catalog metadata without observations, labels, or hidden fields."""

    serialized = json.dumps(
        catalog_payload(catalog),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return USER_PROMPT_TEMPLATE.format(catalog_json=serialized)


def prompt_hash() -> str:
    """Hash the frozen instruction surface; catalog has its own separate hash."""

    material = "\n@@SYSTEM@@\n".join((SYSTEM_PROMPT, USER_PROMPT_TEMPLATE))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
