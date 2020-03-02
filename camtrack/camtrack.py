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


def get_ransac(point_cloud_builder, corners_i, intrinsic_mat):
    intersection, (indexes_cloud, indexes_corners) = snp.intersect(point_cloud_builder.ids.flatten(),
                                                                   corners_i.ids.flatten(),
                                                                   indices=True)
    if len(intersection) < 6:
        return False, None, None, None
    try:
        res_code, rvec, tvec, inliers = cv2.solvePnPRansac(point_cloud_builder.points[indexes_cloud],
                                                           corners_i.points[indexes_corners],
                                                           intrinsic_mat,
                                                           distCoeffs=None)
    except Exception:
        print('exception')
        return False, None, None, None
    rodrig = None
    cloud_points = None
    if res_code:
        rodrig = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        cloud_points = point_cloud_builder.points[indexes_cloud][inliers.flatten()]
    return res_code, rodrig, inliers, cloud_points


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
    print('Known frames are', known_view_1[0], 'and', known_view_2[0])
    view_mat3x4_1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat3x4_2 = pose_to_view_mat3x4(known_view_2[1])
    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])

    triangulation_parameters = TriangulationParameters(60, 1e-1, 1e-2)
    delta = (known_view_2[0] - known_view_1[0])
    min_size = 20

    points3d, corr_ids, _ = triangulate_correspondences(correspondences,
                                                        view_mat3x4_1, view_mat3x4_2,
                                                        intrinsic_mat,
                                                        triangulation_parameters)

    view_mats, point_cloud_builder = [view_mat3x4_1 for _ in corner_storage], PointCloudBuilder(corr_ids.astype(np.int64),
                                                                                       points3d)
    err_indexes = set(range(len(corner_storage)))
    err_indexes.remove(known_view_1[0])
    err_indexes.remove(known_view_2[0])
    res_code, rodrig, inliers_1, cloud_points = get_ransac(point_cloud_builder, corner_storage[known_view_1[0]],
                                                           intrinsic_mat)
    view_mats[known_view_1[0]] = rodrig
    res_code, rodrig, inliers_2, cloud_points = get_ransac(point_cloud_builder, corner_storage[known_view_2[0]],
                                                           intrinsic_mat)
    view_mats[known_view_2[0]] = rodrig
    inliers_min_size = min(len(inliers_1), len(inliers_2)) - 100
    prev_len = [0 for _ in view_mats]
    for ep in range(10):
        delta += ep * 4
        descr_template = 'Point cloud calc epoch # ' + str(ep) + ' - {} inliers, {} points found, cloud size is {}'
        tqdm_iterator = tqdm(corner_storage
                             if ep % 2 == 0
                             else reversed(corner_storage), total=len(rgb_sequence),
                             desc=descr_template.format('?', '?', '?'))
        for i, corners_i in enumerate(tqdm_iterator):
            if ep % 2 == 0:
                i = len(corner_storage) - 1 - i
            res_code, rodrig, inliers, cloud_points = get_ransac(point_cloud_builder, corners_i, intrinsic_mat)
            if res_code and len(inliers) > inliers_min_size and i in err_indexes:
                prev_len[i] = len(inliers)
                inliers = np.array(inliers).astype(np.int64)
                point_cloud_builder.update_points(inliers, cloud_points)
                view_mats[i] = rodrig
                if i in err_indexes:
                    err_indexes.remove(i)
            else:
                inliers = np.array([])
            if i not in err_indexes:
                for j in range(max(0, i - delta), min(i + delta, len(view_mats) - 1)):
                    if i != j and j not in err_indexes:
                        correspondences_i = build_correspondences(corner_storage[i], corner_storage[j])
                        points3d_j, corr_ids_j, median_cos = triangulate_correspondences(correspondences_i,
                                                                                         view_mats[i],
                                                                                         view_mats[j],
                                                                                         intrinsic_mat,
                                                                                         triangulation_parameters)
                        if len(points3d_j) > min_size:
                            point_cloud_builder.add_points(corr_ids_j.astype(np.int64), points3d_j)
                            tqdm_iterator.set_description(descr_template.format(len(inliers),
                                                                                len(points3d_j),
                                                                                len(point_cloud_builder._points)))
    print('Not handled', len(err_indexes), 'frames')
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
