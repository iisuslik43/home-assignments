#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
from tqdm import tqdm

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def calc_frame_corners(image: np.array, block_size: int):
    corners = cv2.goodFeaturesToTrack(image,
                                      maxCorners=1000,
                                      qualityLevel=0.05,
                                      minDistance=block_size * 6,
                                      blockSize=block_size,
                                      useHarrisDetector=False)
    return to_frame_corners(corners, block_size)


def to_frame_corners(corners: np.array, block_size, ids=None):
    if ids is None:
        ids = np.array([i for i in range(len(corners))], dtype=np.int64)
    return FrameCorners(
        ids,
        corners,
        np.array([block_size for _ in range(len(corners))])
    )

def calc_lk(image_0, image_1, prev_corners, block_size):
    win_size = (block_size * 2, block_size * 2)
    max_level = 2
    new_corners, st, err = cv2.calcOpticalFlowPyrLK(
        image_0,
        image_1,
        prev_corners,
        None,
        winSize=win_size,
        maxLevel=max_level
    )
    st = st.reshape(-1)
    return new_corners[st == 1], st


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = (255 * frame_sequence[0]).astype(np.uint8)
    block_size = int(max(image_0.shape[0] * image_0.shape[1] / 120_000, 5))
    corners = calc_frame_corners(image_0, block_size)
    last_id = len(corners.points)
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in tqdm(enumerate(frame_sequence[1:], 1), total=len(frame_sequence[1:]), desc='Calculating corners'):
        image_1 = (255 * image_1).astype(np.uint8)
        prev_corners = builder._corners[frame - 1].points
        prev_ids = builder._corners[frame - 1].ids.reshape(-1)
        next_corners, st = calc_lk(image_0, image_1, prev_corners, block_size)
        next_ids = prev_ids[st == 1]
        new_corners = calc_frame_corners(image_1, block_size).points
        new_corners = np.array([p for p in new_corners
                                if np.min(np.linalg.norm(next_corners - p, axis=1)) > block_size])
        if len(new_corners > 0):
            new_ids = np.arange(last_id, last_id + len(new_corners), dtype=np.int64)
            last_id += len(new_corners)
            next_corners = np.concatenate([next_corners, new_corners])
            next_ids = np.concatenate([next_ids, new_ids])
        corners = to_frame_corners(next_corners, block_size, next_ids)
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
