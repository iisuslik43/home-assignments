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
    rodrigues_and_translation_to_view_mat3x4, eye3x4)


INLIERS_MIN_SIZE = 0
TRIANG_PARAMS = TriangulationParameters(4, 1e-2, 1e-2)
DELTA = 7
MIN_SIZE = 20
FIND_VIEWS_DELTA_FROM = 1
FIND_VIEWS_DELTA_TO = 60
MIN_INTERSECTION_LEN = 10


def find_views(intrinsic_mat: np.ndarray,
               corner_storage: CornerStorage) -> Tuple[Tuple[int, Pose], Tuple[int, Pose]]:
    best_i, best_j = -1, -1
    best_ids = []
    best_matrix = None
    desc_mask = "finding 2 best frames for init, max indexes = {}"
    iterr = tqdm(range(len(corner_storage)), desc=desc_mask.format("?"))
    first_matrix = eye3x4()
    for i in iterr:
        for j in range(i + FIND_VIEWS_DELTA_FROM, min(i + FIND_VIEWS_DELTA_TO, len(corner_storage))):
            intersection, (indexes_i, indexes_j) = snp.intersect(corner_storage[i].ids.flatten(),
                                                                           corner_storage[j].ids.flatten(),
                                                                           indices=True)
            if len(intersection) > MIN_INTERSECTION_LEN:
                points3d_i = corner_storage[i].points[indexes_i]
                points3d_j = corner_storage[j].points[indexes_j]
                retval_ess, mask_ess = cv2.findEssentialMat(points3d_i, points3d_j, focal=intrinsic_mat[0][0])
                retval, R, t, mask = cv2.recoverPose(retval_ess, points3d_i, points3d_j, focal=intrinsic_mat[0][0])
                correspondences_i = build_correspondences(corner_storage[i], corner_storage[j])
                res_matrix = np.concatenate((R, t), axis=1)
                points3d, corr_ids, median_cos = triangulate_correspondences(correspondences_i,
                                                                             first_matrix,
                                                                             res_matrix,
                                                                             intrinsic_mat,
                                                                             TRIANG_PARAMS)
                if len(corr_ids) >= len(best_ids):
                    best_ids = corr_ids
                    iterr.set_description(desc_mask.format(len(corr_ids)))
                    best_i, best_j = i, j
                    best_matrix = res_matrix

    return (best_i, first_matrix), (best_j, best_matrix)


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
                                                           distCoeffs=None,
                                                           reprojectionError=1.5)
    except Exception:
        print('Exception happened in solving PnP, continuing')
        return False, None, None, None
    rodrig = None
    cloud_points = None
    if res_code:
        rodrig = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        cloud_points = point_cloud_builder.points[indexes_cloud][inliers.flatten()]
    return res_code, rodrig, inliers, cloud_points


def update(i, corner_storage, view_mats, point_cloud_builder, intrinsic_mat, err_indexes, template, tqdm_iter):
    res_code, rodrig, inliers, cloud_points = get_ransac(point_cloud_builder, corner_storage[i], intrinsic_mat)
    if res_code:
        inliers = np.array(inliers).astype(np.int64)
        point_cloud_builder.update_points(inliers, cloud_points)
        view_mats[i] = rodrig
        err_indexes.remove(i)
        tqdm_iter.update()

    def update_cloud(j):
        if i != j and j not in err_indexes:
            correspondences_i = build_correspondences(corner_storage[i], corner_storage[j])
            points3d_j, corr_ids_j, median_cos = triangulate_correspondences(correspondences_i,
                                                                             view_mats[i],
                                                                             view_mats[j],
                                                                             intrinsic_mat,
                                                                             TRIANG_PARAMS)

            if len(points3d_j) > MIN_SIZE:
                point_cloud_builder.add_points(corr_ids_j.astype(np.int64), points3d_j)
                tqdm_iter.set_description(template.format(len(inliers),
                                                          len(points3d_j),
                                                          len(point_cloud_builder._points)))

    if i not in err_indexes:
        for j in range(i - 1, max(0, i - DELTA) - 1, -1):
            update_cloud(j)

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = find_views(intrinsic_mat, corner_storage)
        view_mat3x4_1 = known_view_1[1]
        view_mat3x4_2 = known_view_2[1]
    else:
        view_mat3x4_1 = pose_to_view_mat3x4(known_view_1[1])
        view_mat3x4_2 = pose_to_view_mat3x4(known_view_2[1])
    i_1 = known_view_1[0]
    i_2 = known_view_2[0]
    print('Known frames are', i_1, 'and', i_2)
    global INLIERS_MIN_SIZE, DELTA, MIN_SIZE, TRIANG_PARAMS
    correspondences = build_correspondences(corner_storage[i_1], corner_storage[i_2])
    print(len(correspondences.ids))
    points3d, corr_ids, median_cos = triangulate_correspondences(correspondences,
                                                                 view_mat3x4_1,
                                                                 view_mat3x4_2,
                                                                 intrinsic_mat,
                                                                 TRIANG_PARAMS)
    view_mats, point_cloud_builder = [view_mat3x4_1 for _ in corner_storage], PointCloudBuilder(
        corr_ids.astype(np.int64),
        points3d)
    err_indexes = set(range(len(corner_storage)))
    err_indexes.remove(i_1)
    err_indexes.remove(i_2)
    res_code, rodrig, inliers_1, cloud_points = get_ransac(point_cloud_builder, corner_storage[i_1],
                                                           intrinsic_mat)
    view_mats[i_1] = rodrig
    res_code, rodrig, inliers_2, cloud_points = get_ransac(point_cloud_builder, corner_storage[i_2],
                                                           intrinsic_mat)
    view_mats[i_2] = rodrig
    INLIERS_MIN_SIZE = min(len(inliers_1), len(inliers_2)) - 20 * max(len(corner_storage) // 100, 1)
    descr_template = 'Point cloud calc - {} inliers, {} points found, cloud size is {}'

    tqdm_iterator = tqdm(range(len(corner_storage) - 2), total=len(corner_storage),
                         desc=descr_template.format('?', '?', '?'))

    params = [corner_storage, view_mats, point_cloud_builder, intrinsic_mat, err_indexes, descr_template, tqdm_iterator]
    DELTA = 5 * max(len(corner_storage) // 100, 1)
    MIN_SIZE = 20 * max(len(corner_storage) // 100, 1)
    TRIANG_PARAMS = TriangulationParameters(2, 1e-2, 1e-2)
    for i in range(i_1 + 1, i_2):
        update(i, *params)
    print('Points between handled')
    DELTA = 20 * max(len(corner_storage) // 100, 1)
    MIN_SIZE = 20 * max(len(corner_storage) // 100, 1)
    TRIANG_PARAMS = TriangulationParameters(4, 1e-2, 1e-2)
    for i in range(i_1, -1, -1):
        if i in err_indexes:
            update(i, *params)
    for i in range(i_2, len(corner_storage)):
        if i in err_indexes:
            update(i, *params)

    for i in range(i_1, -1, -1):
        if i in err_indexes:
            view_mats[i] = view_mats[i + 1]

    for i in range(i_2, len(corner_storage)):
        if i in err_indexes:
            view_mats[i] = view_mats[i - 1]
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
