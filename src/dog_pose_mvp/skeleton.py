from __future__ import annotations

from dataclasses import dataclass

DOG_KEYPOINT_NAMES = [
    "front_left_paw",
    "front_left_knee",
    "front_left_elbow",
    "rear_left_paw",
    "rear_left_knee",
    "rear_left_elbow",
    "front_right_paw",
    "front_right_knee",
    "front_right_elbow",
    "rear_right_paw",
    "rear_right_knee",
    "rear_right_elbow",
    "tail_start",
    "tail_end",
    "left_ear_base",
    "right_ear_base",
    "nose",
    "chin",
    "left_ear_tip",
    "right_ear_tip",
    "left_eye",
    "right_eye",
    "withers",
    "throat",
]

NAME_TO_INDEX = {name: index for index, name in enumerate(DOG_KEYPOINT_NAMES)}


def edge(start: str, end: str) -> tuple[int, int]:
    return NAME_TO_INDEX[start], NAME_TO_INDEX[end]


# Human pose overlays are easier to read when limbs are explicitly linked.
# This graph approximates canine anatomy using the 24 keypoints provided by the dataset.
DOG_SKELETON_EDGES = [
    edge("front_left_paw", "front_left_knee"),
    edge("front_left_knee", "front_left_elbow"),
    edge("front_left_elbow", "withers"),
    edge("front_right_paw", "front_right_knee"),
    edge("front_right_knee", "front_right_elbow"),
    edge("front_right_elbow", "withers"),
    edge("rear_left_paw", "rear_left_knee"),
    edge("rear_left_knee", "rear_left_elbow"),
    edge("rear_left_elbow", "tail_start"),
    edge("rear_right_paw", "rear_right_knee"),
    edge("rear_right_knee", "rear_right_elbow"),
    edge("rear_right_elbow", "tail_start"),
    edge("withers", "tail_start"),
    edge("tail_start", "tail_end"),
    edge("withers", "throat"),
    edge("throat", "chin"),
    edge("chin", "nose"),
    edge("nose", "left_eye"),
    edge("nose", "right_eye"),
    edge("left_eye", "left_ear_base"),
    edge("right_eye", "right_ear_base"),
    edge("left_ear_base", "left_ear_tip"),
    edge("right_ear_base", "right_ear_tip"),
    edge("left_ear_base", "right_ear_base"),
    edge("throat", "left_ear_base"),
    edge("throat", "right_ear_base"),
]

LEFT_KEYPOINTS = {
    "front_left_paw",
    "front_left_knee",
    "front_left_elbow",
    "rear_left_paw",
    "rear_left_knee",
    "rear_left_elbow",
    "left_ear_base",
    "left_ear_tip",
    "left_eye",
}

RIGHT_KEYPOINTS = {
    "front_right_paw",
    "front_right_knee",
    "front_right_elbow",
    "rear_right_paw",
    "rear_right_knee",
    "rear_right_elbow",
    "right_ear_base",
    "right_ear_tip",
    "right_eye",
}

MIDLINE_KEYPOINTS = set(DOG_KEYPOINT_NAMES) - LEFT_KEYPOINTS - RIGHT_KEYPOINTS


@dataclass(frozen=True)
class SkeletonPalette:
    left_joint: tuple[int, int, int] = (56, 189, 248)
    right_joint: tuple[int, int, int] = (249, 115, 22)
    mid_joint: tuple[int, int, int] = (250, 204, 21)
    left_bone: tuple[int, int, int] = (14, 165, 233)
    right_bone: tuple[int, int, int] = (234, 88, 12)
    mid_bone: tuple[int, int, int] = (163, 230, 53)
    box: tuple[int, int, int] = (244, 114, 182)


PALETTE = SkeletonPalette()
