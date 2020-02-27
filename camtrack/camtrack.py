#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

from corners import CornerStorage
from tqdm import tqdm as tqdm
from data3d import CameraParameters, PointCloud, Pose
import frameseq
import cv2
import pims
from sklearn.preprocessing import normalize
import sortednp as snp

from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences, pose_to_view_mat3x4, triangulate_correspondences, TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mat3x4_1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat3x4_2 = pose_to_view_mat3x4(known_view_2[1])
    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])

    triangulation_parameters = TriangulationParameters(100, 1e-2, 1e-5)
    delta = 10

    points3d, corr_ids, _ = triangulate_correspondences(correspondences,
                                                        view_mat3x4_1, view_mat3x4_2,
                                                        intrinsic_mat,
                                                        triangulation_parameters)

    view_mats, point_cloud_builder = [], PointCloudBuilder(corr_ids.astype(np.int64), points3d)
    descr_template = 'Calculating point cloud - {} inliers, {} points found, cloud size is {}'
    tqdm_iterator = tqdm(zip(corner_storage, rgb_sequence), total=len(rgb_sequence),
                         desc=descr_template.format('?', '?', '?'))
    for i, (corners_i, frame) in enumerate(tqdm_iterator):
        _, (indexes_cloud, indexes_corners) = snp.intersect(point_cloud_builder.ids.flatten(),
                                                            corners_i.ids.flatten(),
                                                            indices=True)
        inliers, rvec, tvec = None, None, None
        try:
            _, rvec, tvec, inliers = cv2.solvePnPRansac(point_cloud_builder.points[indexes_cloud],
                                                             corners_i.points[indexes_corners],
                                                             intrinsic_mat,
                                                             distCoeffs=None)
        except Exception:
            if i == 0:
                raise Exception()
        if inliers is None or rvec is None or tvec is None:
            view_mats.append(view_mats[-1])
            inliers = np.array([])
        else:
            inliers = np.array(inliers).astype(np.int64)
            point_cloud_builder.update_points(inliers, point_cloud_builder.points[indexes_cloud][inliers.flatten()])
            view_mats.append(rodrigues_and_translation_to_view_mat3x4(rvec, tvec))
        if i > delta:
            j = i - delta
            correspondences_i = build_correspondences(corner_storage[j], corner_storage[i])
            points3d_i, corr_ids_i, _ = triangulate_correspondences(correspondences_i,
                                                                    view_mats[j], view_mats[i],
                                                                    intrinsic_mat,
                                                                    triangulation_parameters)
            point_cloud_builder.add_points(corr_ids_i.astype(np.int64), points3d_i)
            tqdm_iterator.set_description(descr_template.format(len(inliers), len(points3d_i),
                                                                len(point_cloud_builder._points)))
        else:
            tqdm_iterator.set_description(descr_template.format(len(inliers), '?', '?'))

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
