from __future__ import annotations

import unittest

import numpy as np
import torch
import pandas as pd

from dog_pose_mvp.gait import _build_patellar_luxation_summary, _extract_primary_points, _stabilize_trend_metrics
from dog_pose_mvp.skeleton import DOG_KEYPOINT_NAMES


class _FakeBoxes:
    def __init__(self) -> None:
        self.conf = torch.tensor([0.9], dtype=torch.float32)

    def __len__(self) -> int:
        return 1


class _FakeKeypoints:
    def __init__(self) -> None:
        xy = np.stack([[index * 10.0, index * 5.0] for index in range(len(DOG_KEYPOINT_NAMES))])
        conf = np.ones(len(DOG_KEYPOINT_NAMES), dtype=np.float32)
        self.xy = torch.tensor(np.expand_dims(xy, axis=0), dtype=torch.float32)
        self.conf = torch.tensor(np.expand_dims(conf, axis=0), dtype=torch.float32)


class _FakeResult:
    def __init__(self) -> None:
        self.boxes = _FakeBoxes()
        self.keypoints = _FakeKeypoints()


class ExtractPrimaryPointsTest(unittest.TestCase):
    def test_extracts_xy_points_for_visible_keypoints(self) -> None:
        point_map, conf_map, detection_conf = _extract_primary_points(_FakeResult(), keypoint_conf=0.35)

        self.assertAlmostEqual(detection_conf, 0.9, places=6)
        self.assertAlmostEqual(conf_map["nose"], 1.0, places=6)
        np.testing.assert_allclose(point_map["front_left_paw"], np.array([0.0, 0.0]))
        np.testing.assert_allclose(point_map["nose"], np.array([160.0, 80.0]))


class StabilizeTrendMetricsTest(unittest.TestCase):
    def test_brief_occlusion_is_smoothed_across_neighboring_frames(self) -> None:
        trend_df = pd.DataFrame(
            {
                "frame_index": [0, 1, 2, 3, 4],
                "time_sec": [0.0, 0.25, 0.5, 0.75, 1.0],
                "detected": [1, 1, 0, 1, 1],
                "visible_keypoints": [24.0, 23.0, 0.0, 22.0, 24.0],
                "detection_conf": [0.94, 0.91, np.nan, 0.9, 0.92],
                "body_axis_angle_deg": [5.0, 6.0, np.nan, 8.0, 9.0],
                "front_left_paw_phase_norm": [0.1, 0.2, np.nan, 0.4, 0.5],
            }
        )

        stabilized = _stabilize_trend_metrics(trend_df, analyzed_fps=4.0)

        self.assertEqual(int(stabilized.loc[2, "detected"]), 1)
        self.assertEqual(float(stabilized.loc[2, "visible_keypoints"]), 22.0)
        self.assertAlmostEqual(float(stabilized.loc[2, "body_axis_angle_deg"]), 7.0, places=6)
        self.assertAlmostEqual(float(stabilized.loc[2, "front_left_paw_phase_norm"]), 0.3, places=6)

    def test_long_occlusion_is_interpolated_without_missing_gap(self) -> None:
        trend_df = pd.DataFrame(
            {
                "frame_index": [0, 1, 2, 3, 4],
                "time_sec": [0.0, 0.25, 0.5, 0.75, 1.0],
                "detected": [1, 0, 0, 0, 1],
                "visible_keypoints": [24.0, 0.0, 0.0, 0.0, 24.0],
                "detection_conf": [0.95, np.nan, np.nan, np.nan, 0.91],
                "body_axis_angle_deg": [5.0, np.nan, np.nan, np.nan, 9.0],
            }
        )

        stabilized = _stabilize_trend_metrics(trend_df, analyzed_fps=4.0)

        self.assertEqual(stabilized["detected"].tolist(), [1, 1, 1, 1, 1])
        self.assertAlmostEqual(float(stabilized.loc[1, "body_axis_angle_deg"]), 6.0, places=6)
        self.assertAlmostEqual(float(stabilized.loc[2, "body_axis_angle_deg"]), 7.0, places=6)
        self.assertAlmostEqual(float(stabilized.loc[3, "body_axis_angle_deg"]), 8.0, places=6)
        self.assertEqual(stabilized["visible_keypoints"].tolist(), [24.0, 24.0, 24.0, 24.0, 24.0])

    def test_all_missing_series_remains_missing(self) -> None:
        trend_df = pd.DataFrame(
            {
                "frame_index": [0, 1, 2],
                "time_sec": [0.0, 0.25, 0.5],
                "detected": [0, 0, 0],
                "visible_keypoints": [0.0, 0.0, 0.0],
                "detection_conf": [np.nan, np.nan, np.nan],
                "body_axis_angle_deg": [np.nan, np.nan, np.nan],
            }
        )

        stabilized = _stabilize_trend_metrics(trend_df, analyzed_fps=4.0)

        self.assertEqual(stabilized["detected"].tolist(), [0, 0, 0])
        self.assertTrue(np.isnan(float(stabilized.loc[1, "detection_conf"])))
        self.assertTrue(np.isnan(float(stabilized.loc[1, "body_axis_angle_deg"])))


class PatellarLuxationScreenTest(unittest.TestCase):
    def test_flags_unilateral_hindlimb_offloading_pattern(self) -> None:
        trend_df = pd.DataFrame(
            {
                "time_sec": np.arange(8, dtype=float) * 0.25,
                "rear_left_paw_phase_norm": [0.05, 0.08, 0.1, 0.12, 0.13, 0.14, 0.16, 0.18],
                "rear_right_paw_phase_norm": [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15],
                "rear_left_paw_drop_norm": [0.1] * 8,
                "rear_right_paw_drop_norm": [0.25] * 8,
                "rear_left_proximal_angle_deg": [109.0, 110.0, 111.0, 110.0, 109.0, 110.0, 111.0, 110.0],
                "rear_right_proximal_angle_deg": [135.0, 136.0, 137.0, 136.0, 135.0, 136.0, 137.0, 136.0],
            }
        )
        gait_stats = {
            "valid_frame_ratio": 1.0,
            "analysis_duration_sec": 1.75,
            "rear_amplitude_balance": 0.35,
        }

        summary, stats, note, status, reasons, primary_side, evidence = _build_patellar_luxation_summary(
            trend_df,
            gait_stats,
        )

        self.assertEqual(status, "주의")
        self.assertEqual(primary_side, "왼쪽 후지")
        self.assertGreater(float(stats["rear_offloading_ratio"]), 0.9)
        self.assertFalse(summary.empty)
        self.assertTrue(reasons)
        self.assertIn("weight-bearing", summary["Interpretation"].iloc[1])
        self.assertIn("pubmed", evidence.lower())
        self.assertIn("왼쪽 후지", note)

    def test_keeps_symmetric_hindlimb_pattern_near_normal(self) -> None:
        trend_df = pd.DataFrame(
            {
                "time_sec": np.arange(8, dtype=float) * 0.25,
                "rear_left_paw_phase_norm": [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15],
                "rear_right_paw_phase_norm": [0.12, 0.26, 0.41, 0.56, 0.71, 0.84, 1.01, 1.14],
                "rear_left_paw_drop_norm": [0.24, 0.25, 0.24, 0.25, 0.24, 0.25, 0.24, 0.25],
                "rear_right_paw_drop_norm": [0.25, 0.24, 0.25, 0.24, 0.25, 0.24, 0.25, 0.24],
                "rear_left_proximal_angle_deg": [132.0, 134.0, 136.0, 134.0, 132.0, 134.0, 136.0, 134.0],
                "rear_right_proximal_angle_deg": [131.0, 133.0, 135.0, 133.0, 131.0, 133.0, 135.0, 133.0],
            }
        )
        gait_stats = {
            "valid_frame_ratio": 1.0,
            "analysis_duration_sec": 1.75,
            "rear_amplitude_balance": 0.96,
        }

        _, stats, _, status, _, primary_side, _ = _build_patellar_luxation_summary(trend_df, gait_stats)

        self.assertEqual(status, "정상 패턴에 가까움")
        self.assertLess(float(stats["rear_offloading_ratio"]), 0.05)
        self.assertIn(primary_side, {"판정 어려움", "양측/혼합 패턴"})


if __name__ == "__main__":
    unittest.main()
