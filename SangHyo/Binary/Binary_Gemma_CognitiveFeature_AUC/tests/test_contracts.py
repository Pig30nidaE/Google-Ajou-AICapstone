from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SangHyo.Binary.Binary_Gemma_CognitiveFeature_AUC import run
from SangHyo.Binary.Binary_Google_ROCAUC_Champion.data import (
    AccessAudit,
    LeakageContractError,
    MMSE_DOMAINS,
    MMSE_ITEMS,
)


def _anchor() -> tuple[np.ndarray, tuple[str, ...]]:
    names = (
        "mmse__total",
        *(f"mmse__domain__{domain}_score" for domain in MMSE_DOMAINS),
        *(f"mmse__item__{item.lower()}_correct" for item in MMSE_ITEMS),
        "mmse__failed_items",
        "mmse__recall_deficit",
    )
    assert len(names) == 39
    values = np.ones(39, dtype=float)
    values[names.index("mmse__total")] = 28.0
    for domain, items in MMSE_DOMAINS.items():
        values[names.index(f"mmse__domain__{domain}_score")] = float(len(items))
    values[names.index("mmse__failed_items")] = 2.0
    values[names.index("mmse__recall_deficit")] = 1.0
    return values, tuple(names)


def _valid_features(value: float = 0.5) -> dict[str, float]:
    return {name: value for name in run.FEATURE_NAMES}


def test_payload_excludes_label_identifier_admin_date_and_collection_metadata() -> None:
    values, names = _anchor()
    payload = run.build_anonymous_payload(values, names)
    run.assert_private_payload(payload)
    serialized = run.canonical_json(payload).lower()
    for forbidden in (
        "subject",
        "patient",
        "email",
        "sample",
        "diagnosis",
        '"diag',
        "label",
        "doctor",
        "admin",
        "timestamp",
        "observation",
        "coverage",
    ):
        assert forbidden not in serialized


def test_prompt_is_task_aware_but_dynamic_payload_has_no_subject_label() -> None:
    values, names = _anchor()
    payload = run.build_anonymous_payload(values, names)
    prompt = run.render_user_prompt(payload)
    static = " ".join(run.SYSTEM_PROMPT.lower().split())
    assert "cognitively normal (cn)" in static
    assert "mild cognitive impairment (mci)" in static
    assert "hard and most important boundary is cn versus mci" in static
    assert "delayed-recall" in static
    assert "do not infer or output a diagnosis" in static
    assert "SOME_REAL_EMAIL" not in prompt
    assert "DIAG_NM" not in prompt


def test_strict_feature_schema() -> None:
    result = run.validate_feature_response(_valid_features())
    assert tuple(result) == run.FEATURE_NAMES
    with pytest.raises(ValueError):
        run.validate_feature_response({**_valid_features(), "extra": 0.1})
    broken = _valid_features()
    broken[run.FEATURE_NAMES[0]] = 1.1
    with pytest.raises(ValueError):
        run.validate_feature_response(broken)
    broken = _valid_features()
    broken[run.FEATURE_NAMES[0]] = True
    with pytest.raises(ValueError):
        run.validate_feature_response(broken)


def test_subject_cv_is_disjoint_and_oof_once_per_repeat() -> None:
    subjects = np.asarray([f"s{index}" for index in range(20)])
    y = np.asarray([0] * 10 + [1] * 10)
    records = run.build_subject_splits(
        y, subjects, folds=5, repeats=3, seed=17
    )
    for repeat in range(3):
        seen = np.zeros(len(subjects), dtype=int)
        for record in [item for item in records if item.repeat == repeat]:
            assert not set(subjects[record.train_indices]) & set(
                subjects[record.test_indices]
            )
            seen[record.test_indices] += 1
        assert np.all(seen == 1)


def test_cache_identity_and_record_are_payload_bound_and_identifier_free(
    tmp_path: Path,
) -> None:
    values, names = _anchor()
    payload = run.build_anonymous_payload(values, names)
    client = run.GemmaFeatureClient(
        run.ClientConfig(offline=True), tmp_path
    )
    identity = client.request_identity(payload)
    client._write_cache(  # contract-level cache test; no network
        identity,
        _valid_features(),
        {"prompt_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
    )
    cache_path = client.cache_path(identity["request_hash"])
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "payload" not in record
    assert "subject_id" not in record
    assert "features" in record
    assert client._read_cache(identity) == _valid_features()

    changed = json.loads(json.dumps(payload))
    changed["mmse"]["total"]["score"] = 27.0
    assert (
        client.request_identity(changed)["request_hash"]
        != identity["request_hash"]
    )


def test_extraction_summary_counts_successful_cache_misses() -> None:
    values, names = _anchor()
    payloads = [
        run.build_anonymous_payload(values, names),
        run.build_anonymous_payload(values.copy(), names),
    ]

    class FakeClient:
        calls = 0

        def extract_one(
            self, _payload: object
        ) -> tuple[dict[str, float], str, int, dict[str, int]]:
            self.calls += 1
            if self.calls == 1:
                return _valid_features(), "cached", 0, {
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                }
            return _valid_features(), "fresh", 1, {
                "prompt_tokens": 10,
                "output_tokens": 5,
                "thinking_tokens": 2,
            }

    _matrix, summary = run.extract_payloads(
        payloads, FakeClient()  # type: ignore[arg-type]
    )
    assert summary.cached == 1
    assert summary.fresh == 1
    assert summary.cache_misses == 1


def test_cli_parser_supports_required_stages_and_profiles() -> None:
    parser = run.build_parser()
    args = parser.parse_args(["--stage", "inspect", "--profile", "smoke"])
    assert args.stage == "inspect"
    assert args.profile == "smoke"
    assert run.notebook_argv(
        {"BGCFA_ARGS": '--stage train --output-dir "/tmp/path with spaces"'}
    ) == ["--stage", "train", "--output-dir", "/tmp/path with spaces"]


def test_historical_validation_fails_before_api_on_subject_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    values, names = _anchor()
    payload = run.build_anonymous_payload(values, names)
    monkeypatch.setattr(
        run,
        "_load_anchor_split",
        lambda *_args, **_kwargs: (
            pd.DataFrame(),
            np.asarray(["same-subject"]),
            values.reshape(1, -1),
            names,
            [payload],
        ),
    )

    class NeverCalled:
        def extract_one(self, _payload: object) -> object:
            raise AssertionError("API/cache extraction must not be reached")

    with pytest.raises(LeakageContractError):
        run.historical_validation(
            data_root=tmp_path,
            output=tmp_path,
            audit=AccessAudit(),
            client=NeverCalled(),  # type: ignore[arg-type]
            bundle={},
            training_subject_ids=["same-subject"],
            salt="test",
        )
