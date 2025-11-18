# -*- coding: utf-8 -*-
# camera_calib_from_images.py
# 직접 촬영한 체커보드 이미지들로 카메라 캘리브레이션 수행
# 1) 사용된 이미지 수 + RMS 에러
# 2) 카메라 내부 파라미터 (K)
# 3) 보정된(undistorted) 이미지
# 4) 카메라 포즈시각화

import argparse
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_chessboard_images(images_dir):
    """폴더에서 이미지 파일 모두 불러옴."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [
        p for p in Path(images_dir).glob("*")
        if p.suffix.lower() in exts
    ]
    paths = sorted(paths)
    if not paths:
        raise RuntimeError(f"이미지 파일을 찾을 수 없습니다: {images_dir}")
    return paths


def find_corners_in_images(image_paths, board_pattern, output_dir):
    pattern_cols, pattern_rows = board_pattern

    # 체커보드 3D 포인트 (Z=0, 보드 평면)
    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)

    obj_points = []  # 3D points in world coordinates
    img_points = []  # 2D points in image plane
    image_size = None

    corner_vis_saved = False

    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for idx, img_path in enumerate(image_paths):
        img = cv.imread(str(img_path))
        if img is None:
            print(f"[경고] 이미지를 읽을 수 없습니다: {img_path}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(
            gray, board_pattern,
            flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
        )

        if not ret:
            print(f"[정보] 체커보드 코너 탐지 실패: {img_path}")
            continue

        corners_refined = cv.cornerSubPix(
            gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
            criteria=criteria
        )

        obj_points.append(objp.copy())
        img_points.append(corners_refined)

        image_size = gray.shape[::-1]  # (width, height)

        # 예시용 코너 시각화 이미지 1장 저장
        if not corner_vis_saved:
            vis_img = img.copy()
            cv.drawChessboardCorners(
                vis_img, board_pattern, corners_refined, ret
            )
            corner_img_path = Path(output_dir) / "chessboard_corners_example.png"
            cv.imwrite(str(corner_img_path), vis_img)
            print(f"[저장] 코너 시각화 이미지: {corner_img_path}")
            corner_vis_saved = True

    if not obj_points:
        raise RuntimeError("어떤 이미지에서도 체커보드 코너를 찾지 못했습니다.")

    return obj_points, img_points, image_size


def calibrate_camera(obj_points, img_points, image_size, cell_size):
    """
    OpenCV의 calibrateCamera를 이용해서 카메라를 캘리브레이션.
    cell_size(한 칸의 실제 크기, m 단위)를 곱해 실제 스케일 반영.
    """
    # 보드 실제 크기를 반영
    scaled_obj_points = [
        pts * float(cell_size) for pts in obj_points
    ]

    # 카메라 캘리브레이션 수행
    rms, K, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        scaled_obj_points,
        img_points,
        image_size,
        None,
        None
    )
    return rms, K, dist_coeffs, rvecs, tvecs


def save_undistorted_example(image_paths, K, dist_coeffs, output_dir):
    """
    첫 번째 이미지 하나를 선택해서 undistort 후 저장.
    """
    if not image_paths:
        return

    img = cv.imread(str(image_paths[0]))
    if img is None:
        print("[경고] undistort 이미지를 읽을 수 없습니다.")
        return

    h, w = img.shape[:2]
    new_K, roi = cv.getOptimalNewCameraMatrix(
        K, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )
    undistorted = cv.undistort(img, K, dist_coeffs, None, new_K)

    undist_path = Path(output_dir) / "undistorted_example.png"
    cv.imwrite(str(undist_path), undistorted)
    print(f"[저장] undistorted 이미지: {undist_path}")


def visualize_camera_poses(rvecs, tvecs, cell_size, board_pattern, output_dir):
    """
    캘리브레이션에서 얻은 카메라 포즈(rvec, tvec)를
    3D 공간에 카메라 좌표축으로 시각화.
    """
    if not rvecs or not tvecs:
        print("[정보] 카메라 포즈가 없습니다. 시각화를 건너뜁니다.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    cols, rows = board_pattern
    board_w = cols * cell_size
    board_h = rows * cell_size
    board_x = [0, board_w, board_w, 0, 0]
    board_y = [0, 0, board_h, board_h, 0]
    board_z = [0] * len(board_x)
    ax.plot(board_x, board_y, board_z, linestyle="--")

    # 각 카메라의 위치 및 좌표축
    axis_len = cell_size * 2.0  # 카메라 축 길이

    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv.Rodrigues(rvec)
        t = tvec.reshape(3, 1)

        # 월드 좌표계에서 카메라 중심 위치: C = -R^T * t
        C = -R.T @ t
        C = C.flatten()

        ax.scatter(C[0], C[1], C[2], marker="o")
        ax.text(C[0], C[1], C[2], f"Cam {i}", fontsize=8)

        cam_axes = R.T  # world에서 본 카메라 축 방향
        colors = ["r", "g", "b"]
        for k in range(3):
            axis_dir = cam_axes[:, k] * axis_len
            ax.plot(
                [C[0], C[0] + axis_dir[0]],
                [C[1], C[1] + axis_dir[1]],
                [C[2], C[2] + axis_dir[2]],
                color=colors[k],
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Camera Poses (Axes) and Chessboard Plane")
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1, 1, 0.5])

    pose_path = Path(output_dir) / "camera_poses.png"
    plt.tight_layout()
    plt.savefig(pose_path, dpi=200)
    plt.close(fig)
    print(f"[저장] 카메라 포즈 시각화 이미지: {pose_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Camera calibration from your own chessboard images."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="체커보드 이미지들이 들어 있는 폴더 경로",
    )
    parser.add_argument(
        "--pattern_cols",
        type=int,
        required=True,
        help="체커보드 내부 코너 수 (가로 방향, 예: 10)",
    )
    parser.add_argument(
        "--pattern_rows",
        type=int,
        required=True,
        help="체커보드 내부 코너 수 (세로 방향, 예: 7)",
    )
    parser.add_argument(
        "--cell_size",
        type=float,
        required=True,
        help="체커보드 한 칸의 실제 크기 (meter 단위, 예: 0.025)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./calib_results",
        help="결과 이미지 및 로그를 저장할 폴더",
    )
    parser.add_argument(
        "--visualize_poses",
        action="store_true",
        help="캘리브레이션에서 얻은 카메라 포즈를 3D로 시각화.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    board_pattern = (args.pattern_cols, args.pattern_rows)

    # 1) 이미지 로드
    image_paths = load_chessboard_images(args.images_dir)
    print(f"[정보] 총 {len(image_paths)}장의 이미지를 발견했습니다.")

    # 2) 코너 탐지 및 2D-3D 포인트 수집 + 코너 시각화 이미지 저장
    obj_points, img_points, image_size = find_corners_in_images(
        image_paths, board_pattern, args.output_dir
    )

    # 3) 카메라 캘리브레이션 수행
    rms, K, dist_coeffs, rvecs, tvecs = calibrate_camera(
        obj_points, img_points, image_size, args.cell_size
    )

    # 4) 결과 출력 (리포트용 정보)
    print("\n==== Camera Calibration Results ====")
    print(f"* 사용된 이미지 수 (corners detected) = {len(obj_points)} / {len(image_paths)}")
    print(f"* RMS reprojection error = {rms:.6f}")
    print("* Camera intrinsic matrix K =")
    print(K)
    print("* Distortion coefficients (k1, k2, p1, p2, k3, ...) =")
    print(dist_coeffs.flatten())

    # 텍스트 파일로도 저장
    result_txt_path = Path(args.output_dir) / "calibration_result.txt"
    with open(result_txt_path, "w") as f:
        f.write("==== Camera Calibration Results ====\n")
        f.write(f"* Used images = {len(obj_points)} / {len(image_paths)}\n")
        f.write(f"* RMS reprojection error = {rms:.6f}\n")
        f.write("* Camera intrinsic matrix K =\n")
        f.write(str(K) + "\n")
        f.write("* Distortion coefficients =\n")
        f.write(str(dist_coeffs.flatten()) + "\n")
    print(f"[저장] 캘리브레이션 결과 텍스트: {result_txt_path}")

    # 5) undistorted 예시 이미지 저장 (옵션 1)
    save_undistorted_example(image_paths, K, dist_coeffs, args.output_dir)

    # 6) 카메라 포즈 3D 시각화 (옵션 2)
    if args.visualize_poses:
        visualize_camera_poses(
            rvecs, tvecs, args.cell_size, board_pattern, args.output_dir
        )


if __name__ == "__main__":
    main()
