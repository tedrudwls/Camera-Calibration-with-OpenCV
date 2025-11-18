# Camera Calibration with OpenCV (Python)

체커보드 이미지를 이용해 **단일 카메라 캘리브레이션**을 수행하고,  
코너 시각화 / RMS 에러 / 카메라 intrinsic 파라미터 / 왜곡 보정 이미지 / 카메라 포즈까지 한 번에 생성하는 스크립트입니다.

**대학원 고급 컴퓨터 비전 수업 과제용**으로 작성되었으며,  
직접 촬영한 체커보드 이미지들만 있으면 바로 카메라 캘리브레이션 결과를 얻을 수 있습니다.

---

## 1. 기능 요약

`camera_calib_from_images.py` 스크립트는 다음 과정을 자동으로 수행합니다.

1. **체커보드 이미지 로드**
   - 지정한 폴더에서 `.jpg`, `.jpeg`, `.png`, `.bmp` 이미지를 모두 읽어옵니다.

2. **체커보드 코너 검출 & 서브픽셀 보정**
   - `cv.findChessboardCorners`, `cv.cornerSubPix`를 이용해 각 이미지에서 체커보드 내부 코너를 탐지합니다. :contentReference[oaicite:0]{index=0}  
   - 코너 검출에 성공한 이미지들만 캘리브레이션에 사용됩니다.
   - 첫 번째 성공 이미지에 코너를 그려 **`chessboard_corners_example.png`** 로 저장합니다.

3. **카메라 캘리브레이션**
   - OpenCV의 `cv.calibrateCamera`를 이용해  
     - 카메라 intrinsic 행렬 **K**  
     - 왜곡 계수 **distortion coefficients**  
     - 각 이미지에서의 회전/이동 벡터 **(rvecs, tvecs)**  
     - **RMS reprojection error**  
     를 계산합니다. :contentReference[oaicite:1]{index=1}  

4. **왜곡 보정(Undistortion) 이미지 생성**
   - 첫 번째 이미지를 이용해 왜곡을 보정하고,  
     **`undistorted_example.png`** 로 저장합니다.

5. **카메라 포즈 3D 시각화 (옵션)**
   - 캘리브레이션에서 얻은 `rvecs`, `tvecs`를 이용해  
     월드 좌표계에서 **카메라 중심과 좌표축(X, Y, Z)** 을 3D로 그립니다.
   - 체커보드 평면도 함께 시각화하여,  
     **`camera_poses.png`** 로 저장합니다.

6. **리포트용 텍스트 결과 저장**
   - 사용된 이미지 개수, RMS 에러, K, 왜곡 계수 등을  
     **`calibration_result.txt`** 에 정리해서 저장합니다.
   - 콘솔에도 동일한 내용을 출력합니다.

---

## 2. 준비물

### 2.1 체커보드 패턴 출력

다음 사이트에서 체커보드를 생성하거나 다운로드할 수 있습니다.

- **Calibration Checkerboard Collection (Mark Hedley Jones)** :contentReference[oaicite:2]{index=2}  
  - 다양한 크기의 고품질 체커보드 PDF 제공
- **Calib.io – Camera Calibration Pattern Generator** :contentReference[oaicite:3]{index=3}  
  - 원하는 내부 코너 개수 / 셀 크기를 설정해서 PDF를 생성할 수 있는 패턴 제너레이터

> **중요:** 프린트할 때 **페이지 스케일링(맞춤/축소)** 옵션을 끄고 출력해야  
> 셀 크기가 정확하게 유지되어 캘리브레이션 오차가 줄어듭니다. :contentReference[oaicite:4]{index=4}  

### 2.2 체커보드 촬영

1. 출력한 체커보드를 단단한 판에 붙입니다.
2. **여러 각도/거리/위치**에서 체커보드를 촬영합니다.
   - 기울기, 거리, 위치가 다양할수록 좋은 캘리브레이션 결과를 얻을 수 있습니다. :contentReference[oaicite:5]{index=5}  
3. 캘리브레이션에 사용할 이미지를 한 폴더에 모읍니다.
   - 예시: `./images/chess_01.jpg`, `./images/chess_02.jpg`, ...

---

## 3. 설치

### 3.1 의존성

- Python 3.x
- OpenCV (Python)
- NumPy
- Matplotlib

### 3.2 설치 명령 예시

```bash
pip install opencv-python numpy matplotlib
```

---

## 4. 사용법

### 4.1 기본 실행
```bash
python camera_calib_from_images.py \
  --images_dir ./images \
  --pattern_cols 10 \
  --pattern_rows 7 \
  --cell_size 0.025 \
  --output_dir ./calib_results \
  --visualize_poses
```

