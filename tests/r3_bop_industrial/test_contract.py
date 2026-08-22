from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from r3_bop_industrial.core import (
    ContractError,
    assert_r3_output_scope,
    build_official_commands,
    dataset_preflight,
    freeze_input_lock,
    load_protocol,
    sha256_file,
    toolkit_preflight,
    toolkit_source_tree_audit,
    validate_predicted_association,
    validate_input_lock,
    validate_label_access_receipt,
    validate_prediction_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "protocols" / "poseloop_r3_bop_industrial_protocol.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class R3ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol(PROTOCOL_PATH)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "xyzibd"
        self.dataset_root.mkdir()
        self.input_root = self.root / "inputs"
        self.input_root.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_valid_bundle(self) -> dict[str, Path]:
        filenames = self.protocol["prediction_bundle"]["files"]
        paths = {role: self.input_root / name for role, name in filenames.items()}
        coco = [
            {
                "scene_id": 1,
                "image_id": 10,
                "category_id": 2,
                "score": 0.9,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "segmentation": {"counts": "encoded-a", "size": [480, 640]},
                "time": 0.12,
            },
            {
                "scene_id": 1,
                "image_id": 11,
                "category_id": 2,
                "score": 0.8,
                "bbox": [11.0, 21.0, 31.0, 41.0],
                "segmentation": {"counts": "encoded-b", "size": [480, 640]},
                "time": 0.13,
            },
        ]
        write_json(paths["coco_predictions"], coco)
        association = [
            {
                "scene_id": 1,
                "image_id": 10,
                "category_id": 2,
                "detection_index": 0,
                "predicted_track_id": "track-001",
                "association_score": 0.95,
                "view_rank": 0,
                "is_target_view": True,
            },
            {
                "scene_id": 1,
                "image_id": 11,
                "category_id": 2,
                "detection_index": 1,
                "predicted_track_id": "track-001",
                "association_score": 0.90,
                "view_rank": 1,
                "is_target_view": False,
            },
        ]
        write_jsonl(paths["predicted_association"], association)
        pose_text = (
            "scene_id,im_id,obj_id,score,R,t,time\n"
            "1,10,2,0.9,1 0 0 0 1 0 0 0 1,10 20 300,0.20\n"
        )
        paths["single_view_pose"].write_text(pose_text, encoding="utf-8")
        paths["multi_view_pose"].write_text(
            pose_text.replace(",0.9,", ",0.95,").replace(",0.20", ",0.30"),
            encoding="utf-8",
        )
        lineage = self.protocol["lineage"]
        provenance = {
            "schema_version": "poseloop.r3.prediction-provenance.v1",
            "protocol_id": self.protocol["protocol_id"],
            "input_origin": "audited_prediction",
            "producer": "unit-test-producer",
            "producer_version": "1",
            "source_uri_or_run_id": "unit-test-run",
            "created_utc": "2026-08-16T00:00:00Z",
            "label_blind": True,
            "uses_gt_visible_masks": False,
            "uses_oracle_association": False,
            "result_selection_uses_evaluator_metrics": False,
            "source_artifacts": [
                {
                    "role": "producer_code",
                    "uri": "unit-test://producer-code",
                    "sha256": "1" * 64,
                },
                {
                    "role": "model",
                    "uri": "unit-test://model",
                    "sha256": "2" * 64,
                },
            ],
            "files": {
                role: {"filename": paths[role].name, "sha256": sha256_file(paths[role])}
                for role in (
                    "coco_predictions",
                    "predicted_association",
                    "single_view_pose",
                    "multi_view_pose",
                )
            },
            "lineage": lineage,
        }
        write_json(paths["provenance"], provenance)
        return paths

    def test_protocol_freezes_new_id_and_leakage_boundary(self) -> None:
        self.assertEqual(
            self.protocol["protocol_id"], "poseloop.r3.bop-industrial.e2e.v1"
        )
        self.assertEqual(self.protocol["state"], "frozen_before_r3_predictions")
        self.assertFalse(self.protocol["anti_leakage"]["result_tuning_after_score"])
        self.assertIn(
            "oracle_association",
            self.protocol["anti_leakage"]["forbidden_prediction_fields"],
        )

    def test_valid_bundle_links_coco_association_and_pose_outputs(self) -> None:
        self._write_valid_bundle()
        audit = validate_prediction_bundle(
            self.input_root, self.dataset_root, self.protocol
        )
        self.assertTrue(audit["ready"], audit["errors"])
        self.assertEqual(audit["predicted_association"]["track_count"], 1)
        self.assertEqual(audit["comparison_population"]["shared_target_key_count"], 1)
        self.assertTrue(audit["provenance"]["hash_linkage_verified"])

    def test_coco_schema_rejects_legacy_gt_field(self) -> None:
        paths = self._write_valid_bundle()
        coco = json.loads(paths["coco_predictions"].read_text(encoding="utf-8"))
        coco[0]["visible_mask_pixel_count"] = 123
        write_json(paths["coco_predictions"], coco)
        audit = validate_prediction_bundle(
            self.input_root, self.dataset_root, self.protocol
        )
        self.assertFalse(audit["ready"])
        self.assertIn("fields differ from frozen schema", audit["errors"][0])

    def test_association_schema_rejects_oracle_field(self) -> None:
        paths = self._write_valid_bundle()
        rows = [
            json.loads(line)
            for line in paths["predicted_association"].read_text(encoding="utf-8").splitlines()
        ]
        rows[0]["oracle_association"] = {"track_id": "forbidden"}
        write_jsonl(paths["predicted_association"], rows)
        audit = validate_prediction_bundle(
            self.input_root, self.dataset_root, self.protocol
        )
        self.assertFalse(audit["ready"])
        self.assertIn("fields differ from frozen schema", audit["errors"][0])

    def test_predicted_track_ids_are_scoped_by_scene(self) -> None:
        coco = [
            {
                "scene_id": scene_id,
                "image_id": 10,
                "category_id": 2,
            }
            for scene_id in (1, 2)
        ]
        rows = [
            {
                "scene_id": scene_id,
                "image_id": 10,
                "category_id": 2,
                "detection_index": index,
                "predicted_track_id": "track-001",
                "association_score": 0.9,
                "view_rank": 0,
                "is_target_view": True,
            }
            for index, scene_id in enumerate((1, 2))
        ]
        path = self.input_root / "association.jsonl"
        write_jsonl(path, rows)
        _, audit = validate_predicted_association(
            path,
            coco,
            self.protocol["prediction_bundle"]["association_required"],
        )
        self.assertEqual(audit["track_count"], 2)

    def test_audited_predictions_require_code_and_model_hashes(self) -> None:
        paths = self._write_valid_bundle()
        provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        provenance["source_artifacts"] = provenance["source_artifacts"][:1]
        write_json(paths["provenance"], provenance)
        audit = validate_prediction_bundle(
            self.input_root, self.dataset_root, self.protocol
        )
        self.assertFalse(audit["ready"])
        self.assertIn("missing source artifact roles: model", audit["errors"][0])

    def test_pinned_archive_tree_is_accepted_without_git_metadata(self) -> None:
        toolkit_root = self.root / "archive-toolkit"
        (toolkit_root / "bop_toolkit_lib").mkdir(parents=True)
        (toolkit_root / "bop_toolkit_lib" / "core.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (toolkit_root / "scripts").mkdir()
        for relative in self.protocol["toolkit"]["official_evaluators"].values():
            path = toolkit_root / relative
            path.write_text("# evaluator\n", encoding="utf-8")
        (toolkit_root / "pyproject.toml").write_text(
            "[project]\nname='fixture'\n", encoding="utf-8"
        )
        fixture_protocol = json.loads(json.dumps(self.protocol))
        tree = toolkit_source_tree_audit(toolkit_root, fixture_protocol)
        fixture_protocol["toolkit"]["source_tree_contract"]["sha256"] = tree["sha256"]
        fixture_protocol["toolkit"]["source_tree_contract"]["file_count"] = tree[
            "file_count"
        ]
        audit = toolkit_preflight(toolkit_root, fixture_protocol)
        self.assertTrue(audit["ready"], audit["errors"])
        self.assertEqual(audit["source_kind"], "pinned_archive_tree")

    def test_provenance_hash_detects_post_freeze_candidate_mutation(self) -> None:
        paths = self._write_valid_bundle()
        with paths["single_view_pose"].open("a", encoding="utf-8") as handle:
            handle.write("1,10,2,0.7,1 0 0 0 1 0 0 0 1,11 21 301,0.20\n")
        audit = validate_prediction_bundle(
            self.input_root, self.dataset_root, self.protocol
        )
        self.assertFalse(audit["ready"])
        self.assertIn("SHA-256 mismatch", audit["errors"][0])

    def test_commands_cover_official_ar_pose_ap_and_coco_ap(self) -> None:
        commands = build_official_commands(
            self.input_root,
            self.root / "bop_toolkit",
            self.root / "eval",
            self.protocol,
            python_executable="python3",
        )
        self.assertEqual(len(commands), 6)
        pairs = {(row["metric"], row["variant"]) for row in commands}
        self.assertIn(
            ("official_bop19_localization_ar", "single_view"), pairs
        )
        self.assertIn(
            ("official_bop19_localization_ar", "multi_view"), pairs
        )
        self.assertIn(
            ("official_bop24_pose_detection_ap", "single_view"), pairs
        )
        self.assertIn(
            ("official_bop22_coco_segm_ap", "shared_predicted_input"), pairs
        )

    def test_dataset_preflight_does_not_enumerate_evaluator_labels(self) -> None:
        for relative in self.protocol["dataset"]["required_public_inputs"]:
            path = self.dataset_root / relative
            if Path(relative).suffix:
                write_json(path, {})
            else:
                path.mkdir(parents=True, exist_ok=True)
        hidden_label = self.dataset_root / "val" / "000001" / "scene_gt_realsense.json"
        write_json(hidden_label, {"sealed": "must-not-be-read"})
        audit = dataset_preflight(self.dataset_root, self.protocol)
        self.assertTrue(audit["ready"], audit["errors"])
        self.assertEqual(audit["evaluator_only_label_paths_accessed"], [])
        self.assertNotIn(str(hidden_label), audit["accessed_paths"])

    def test_lock_and_label_access_are_hash_bound(self) -> None:
        preflight = {
            "status": "ready",
            "protocol_id": self.protocol["protocol_id"],
            "fingerprint_sha256": "a" * 64,
        }
        lock = freeze_input_lock(preflight)
        lock_path = self.root / "input_lock.json"
        write_json(lock_path, lock)
        current = dict(preflight)
        validate_input_lock(lock_path, current)
        authorization_path = self.root / "authorization.json"
        write_json(
            authorization_path,
            {
                "schema_version": "poseloop.r3.label-access.v1",
                "protocol_id": self.protocol["protocol_id"],
                "approved": True,
                "scope": "official_bop_toolkit_evaluator_only",
                "approved_by": "reviewer",
                "approved_at_utc": "2026-08-16T00:00:00Z",
                "input_lock_sha256": sha256_file(lock_path),
            },
        )
        validate_label_access_receipt(authorization_path, lock_path)
        write_json(lock_path, {**lock, "fingerprint_sha256": "b" * 64})
        with self.assertRaisesRegex(ContractError, "does not match"):
            validate_label_access_receipt(authorization_path, lock_path)

    def test_output_scope_cannot_touch_frozen_report_namespace(self) -> None:
        with self.assertRaisesRegex(ContractError, "R3 namespace"):
            assert_r3_output_scope(
                REPO_ROOT / "reports" / "r1" / "forbidden.json",
                REPO_ROOT,
                self.protocol,
            )
        assert_r3_output_scope(
            REPO_ROOT / "reports" / "r3_bop_industrial" / "allowed.json",
            REPO_ROOT,
            self.protocol,
        )


if __name__ == "__main__":
    unittest.main()
