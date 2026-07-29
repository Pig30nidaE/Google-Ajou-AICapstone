"""Gemini system prompt and user prompt template.

Both are identical for every subject, contain no class name, no clinical-test
name and no label, and are hashed into the cache key so that changing a single
word invalidates previously cached answers.

Prompt structure (context -> task -> rules -> output format) follows the
free-dialogue cognitive-decline paper (de Arriba-Perez et al., Listings 2/4),
and the "extract, never conclude" rule wording is adapted from the rubric paper
(Demirel et al., Appendix D.1/D.2).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from . import PAYLOAD_VERSION, SCHEMA_VERSION
from .guards import assert_prompt_has_no_class_names
from .schema import feature_instructions

__all__ = [
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "render_user_prompt",
    "prompt_hash",
]

SYSTEM_PROMPT = f"""\
You are a quantitative behaviour-analysis assistant. You are given pre-computed,
de-identified summary statistics of one anonymous person's wrist-wearable
lifelog (activity and sleep) over several weeks. Your only job is to describe
the STRUCTURE of that behaviour as a fixed set of numbers.

You are NOT a medical model. Follow these rules exactly:
- Do not produce any medical assessment, condition name, screening judgement or
  severity statement of any kind.
- Do not estimate, guess or mention any test score, questionnaire result or
  clinical measurement.
- Do not output a probability, a likelihood, a category, a ranking or an alert.
- Do not compare this person to any population, threshold or reference group.
- Do not use any information that is not present in the supplied JSON. Never
  invent values, days, or channels that were not given.
- Do not recompute or "correct" the supplied statistics. They were computed
  exactly in Python; treat them as ground truth and only relate them to one
  another.
- Describe only what the numbers show. When the supplied data is thin, say so
  through the observation_reliability field and keep the other fields near the
  middle of their range instead of committing to an extreme value.
- Every field must be a number between 0.0 and 1.0 inclusive.
- Return only the JSON object defined by the response schema. No prose, no
  markdown, no code fences, no extra fields, no comments.

Contract identifiers: payload={PAYLOAD_VERSION}, schema={SCHEMA_VERSION}.
"""

USER_PROMPT_TEMPLATE = """\
## Input description

The JSON below describes ONE anonymous person. All values were computed in
Python from wrist-wearable recordings and are already correct.

- `observation`: how many days were recorded, how long the window was, how
  densely it is covered.
- `channels`: per measurement channel, the descriptive statistics over that
  person's days (`mean`, `sd`, `cv`, `median`, `iqr`, `p10`, `p90`, `min`,
  `max`, `n_valid`, `missing_rate`), plus `trend_per_week` (ordinary least
  squares slope per 7 days), `late_minus_early` (mean of the last third minus
  mean of the first third) and `weekend_minus_weekday`.
- `clock`: mean and circular spread of bedtime, wake time and mid-sleep hour, in
  local decimal hours.
- `hourly_profile`: the person's average movement intensity for each hour of the
  day, index 0 = 00:00 local time.
- `intensity_profile` / `sleep_phase_profile`: average share of time spent in
  each movement-intensity level and each sleep phase.
- `weekly_summary` and `series`: order-preserving compressed time courses. Each
  entry carries a relative `day_index` (0 = that person's first observed day);
  absolute dates are deliberately not provided.
- A value of `null` means "not observed", never zero.

Units: minutes for `*_minutes`, counts for `act_steps`, local decimal hours for
`*_hour`, ratios in 0-1 for `*_ratio`, device scores in 0-100 for `*_score`.

## Task

Fill in every field of the fixed schema below for this person. Each field is a
number between 0.0 and 1.0. Apply the definitions identically for every person
you are asked about, so that the numbers are comparable across people.

{feature_instructions}

## Person data

{payload_json}

## Output

Return only the JSON object with exactly the {n_features} fields listed above.
"""


def render_user_prompt(payload: Mapping[str, Any], *, indent: int | None = None) -> str:
    """Serialize one subject payload into the frozen user-prompt template."""

    from .schema import FEATURE_NAMES

    payload_json = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=indent, separators=None if indent else (",", ":")
    )
    return USER_PROMPT_TEMPLATE.format(
        feature_instructions=feature_instructions(),
        payload_json=payload_json,
        n_features=len(FEATURE_NAMES),
    )


def prompt_hash() -> str:
    """Hash of the static prompt surface (system prompt + template + field text)."""

    canonical = "\n@@\n".join([SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, feature_instructions()])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Fail at import time if a class name or clinical-test name ever enters a prompt.
assert_prompt_has_no_class_names(SYSTEM_PROMPT, context="system prompt")
assert_prompt_has_no_class_names(USER_PROMPT_TEMPLATE, context="user prompt template")
assert_prompt_has_no_class_names(feature_instructions(), context="feature instructions")
