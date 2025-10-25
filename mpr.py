# mpr.py
from __future__ import annotations
import numpy as np
import SimpleITK as sitk

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def _orthonormal_basis(normal, up_hint):
    n = _normalize(normal)
    # up_hint が n と平行気味なら安全にずらす
    uh = _normalize(up_hint)
    if abs(np.dot(n, uh)) > 0.99:
        uh = _normalize([uh[1], uh[2], uh[0]])
    e1 = _normalize(np.cross(n, uh))  # 面内X
    e0 = _normalize(np.cross(e1, n))  # 面内Y（upに近い）
    # 出力の方向は (e0, e1, n) を列に持つ 3x3 で良い（SimpleITKは列優先でもOK）
    R = np.stack([e0, e1, n], axis=1)  # 3x3
    return R, e0, e1, n

def build_sitk_image(volume_zyx: np.ndarray, spacing_zyx, origin_xyz, direction_9):
    """
    numpy [Z,Y,X] を SimpleITK Image へ（LPS座標系）
    spacing_zyx: (Z,Y,X) -> sitk spacing (X,Y,Z)
    origin_xyz:  (X,Y,Z)
    direction_9: フラットな9要素（行優先 or 列優先でも一致させればOK）
    """
    img = sitk.GetImageFromArray(volume_zyx.astype(np.float32))  # [Z,Y,X]
    # SimpleITK は spacing=(X,Y,Z)
    px = float(spacing_zyx[2]); py = float(spacing_zyx[1]); pz = float(spacing_zyx[0])
    img.SetSpacing((px, py, pz))
    img.SetOrigin(tuple([float(v) for v in origin_xyz]))
    img.SetDirection(tuple([float(v) for v in direction_9]))  # 3x3 flatten
    return img

def reslice_oblique(
    volume_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: np.ndarray,
    direction_9: list[float],
    center_mm: np.ndarray,
    normal_xyz: np.ndarray,
    up_hint_xyz: np.ndarray,
    out_size_xy: tuple[int, int],
    out_spacing_xy: tuple[float, float],
    slab_thickness_mm: float = 1.0,
    default_val: float = 0.0,
):
    """
    任意平面 MPR を 2.5D で1枚生成（slab_thickness_mm を薄くすると純2D）
    - out_size_xy: 出力ピクセル数 (Nx, Ny)
    - out_spacing_xy: 出力ピクセル間隔 (sx, sy) [mm]
    - center_mm: 面の中心の物理座標（LPS, mm）
    - normal_xyz: 面法線
    - up_hint_xyz: 面内の「上」方向のヒント
    """
    # 1) sitk.Image を構築
    src = build_sitk_image(volume_zyx, spacing_zyx, origin_xyz, direction_9)

    # 2) 面の直交基底を作成（e0:出力X方向, e1:出力Y方向, n:法線＝出力Z方向）
    R, e0, e1, n = _orthonormal_basis(normal_xyz, up_hint_xyz)

    # 3) 出力グリッド（方向・原点・間隔・サイズ）
    # 出力画像の方向（3x3行列をフラット）: [e0, e1, n] を列に持つ形で flatten
    out_direction = np.array([e0, e1, n]).T.flatten().tolist()
    sx, sy = float(out_spacing_xy[0]), float(out_spacing_xy[1])
    sz = float(max(slab_thickness_mm, 1e-3))  # 薄い厚みを持たせる（=1スライスでもZ>0が必要）

    nx, ny = int(out_size_xy[0]), int(out_size_xy[1])
    nz = 1  # 1スライスMPR
    # 面の左上（原点） = center - 0.5*( (nx-1)*sx*e0 + (ny-1)*sy*e1 ) - 0.5*sz*n
    origin_out = (
        center_mm
        - 0.5 * ((nx - 1) * sx) * e0
        - 0.5 * ((ny - 1) * sy) * e1
        - 0.5 * sz * n
    )

    # 4) リサンプリング（恒等変換でOK。出力グリッドに方向を持たせているため）
    out = sitk.Resample(
        src,
        size=(nx, ny, nz),
        transform=sitk.Transform(3, sitk.sitkIdentity),
        interpolator=sitk.sitkLinear,
        outputOrigin=tuple(origin_out.tolist()),
        outputSpacing=(sx, sy, sz),
        outputDirection=tuple(out_direction),
        defaultPixelValue=float(default_val),
        outputPixelType=sitk.sitkFloat32,
    )

    # 5) numpy へ（[Z,Y,X] = [1, ny, nx]）→ 2Dへ squeeze
    arr = sitk.GetArrayFromImage(out)  # [Z,Y,X]
    arr2d = arr[0, :, :]               # [Y,X]
    return arr2d