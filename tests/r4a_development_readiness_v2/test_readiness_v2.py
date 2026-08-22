from __future__ import annotations

from pathlib import Path

from r4a_development_readiness_v2.core import _finite_matrix, _finite_vector, _loaded_pose_valid, load_protocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "poseloop_r4a_xyzibd_train_pbr_development_readiness_v2.json"


class ArrayLike:
    def __init__(self, shape: tuple[int, ...], value: object) -> None:
        self.shape = shape
        self._value = value

    def tolist(self) -> object:
        return self._value


def test_v2_protocol_seals_v1_failure_and_forbids_val() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["immutable_v1_failure"]["rerun_permitted"] is False
    assert protocol["schema_only_diagnosis"]["pose_values_recorded"] is False
    assert protocol["boundaries"]["xyzibd_val_access_permitted"] is False
    assert "visibility fraction" in protocol["manifest_contract"]["excludes"]


def test_official_nested_raw_and_loader_shapes_are_valid() -> None:
    rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    translation = [1.0, 2.0, 3.0]
    assert _finite_matrix(rotation, 3, 3)
    assert _finite_vector(translation, 3)
    assert _loaded_pose_valid(
        {
            "cam_R_m2c": ArrayLike((3, 3), rotation),
            "cam_t_m2c": ArrayLike((3, 1), [[1.0], [2.0], [3.0]]),
        }
    )


def test_v2_target_manifest_source_has_no_label_exports() -> None:
    source = (ROOT / "r4a_development_readiness_v2" / "core.py").read_text(encoding="utf-8")
    manifest_source = source[source.index("targets.append(") :]
    assert '"cam_R_m2c"' not in manifest_source
    assert '"cam_t_m2c"' not in manifest_source
    assert '"visibility"' not in manifest_source
