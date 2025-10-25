# dicom_io.py （丸ごと置き換え推奨）
from __future__ import annotations
import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pydicom
from collections import defaultdict

class SeriesData:
    """1シリーズ分のデータ（時系列対応）"""
    def __init__(self, volume: np.ndarray, meta: Dict[str, Any]):
        # volume: t=0 の [Z, Y, X] （後方互換のため）
        self.volume = volume
        self.meta = meta  # spacing, ds0, photometric, window_center/width, volumes(各t), times(表示用) など

def _safe_get(ds, name, default=None):
    return getattr(ds, name, default)

def _get_time_key(ds) -> Tuple[str, float, str]:
    """
    優先順に時間キーを返す。
    return: (used_tag_name, sortable_value, pretty_string)
    """
    # 1) TemporalPositionIdentifier（整数が多い）
    tpi = _safe_get(ds, "TemporalPositionIdentifier", None)
    if tpi is not None:
        try:
            v = float(int(tpi))
            return ("TemporalPositionIdentifier", v, f"TPI={int(tpi)}")
        except Exception:
            pass

    # 2) FrameReferenceTime（ms が入ることが多い）
    frt = _safe_get(ds, "FrameReferenceTime", None)
    if frt is not None:
        try:
            v = float(frt)
            return ("FrameReferenceTime", v, f"FRT={v:.0f}ms")
        except Exception:
            pass

    # 3) AcquisitionTime ("HHMMSS.FFFFFF")
    at = _safe_get(ds, "AcquisitionTime", None)
    if at:
        s = str(at)
        # HHMMSS.FFFFFF → 秒に換算（簡易パーサ）
        try:
            hh = float(s[0:2]); mm = float(s[2:4]); ss = float(s[4:])
            v = hh * 3600.0 + mm * 60.0 + ss
            return ("AcquisitionTime", v, f"AT={s}")
        except Exception:
            # ソート用にはフォールバックとして 0.0
            return ("AcquisitionTime", 0.0, f"AT={s}")

    # 未設定 → t=0 扱い
    return ("None", 0.0, "t=0")

def load_series_from_folder(folder: str) -> Tuple[Optional[SeriesData], Optional[str]]:
    """
    フォルダ内のDICOMを1シリーズだけ読み込み、時系列ごとの3D体積を作成。
    戻り値: (SeriesData or None, エラーメッセージ or None)
    """
    if not os.path.isdir(folder):
        return None, "指定フォルダが存在しません。"

    # 全DICOM読み込み（SeriesUID, time_key, positionZ, inst, path, ds）
    recs = []
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
                if "PixelData" not in ds:
                    continue
                series_uid = _safe_get(ds, "SeriesInstanceUID", None)
                if not series_uid:
                    continue

                tag_name, t_sort, t_pretty = _get_time_key(ds)
                inst = int(_safe_get(ds, "InstanceNumber", 0) or 0)

                ipp = _safe_get(ds, "ImagePositionPatient", [0, 0, 0])
                z = float(ipp[2]) if isinstance(ipp, (list, tuple)) and len(ipp) == 3 else float(inst)

                recs.append((series_uid, tag_name, t_sort, t_pretty, z, inst, path))
            except Exception:
                continue

    if not recs:
        return None, "DICOM画像が見つかりませんでした。"

    # 最初に見つかったシリーズのみ対象
    first_series = recs[0][0]
    recs = [r for r in recs if r[0] == first_series]

    # 時間キーでグルーピング
    # map: t_sort -> List[(z, inst, path, preview_pretty)]
    timeslices = defaultdict(list)
    time_pretty_map: Dict[float, str] = {}
    time_tag_name = None
    for (_, tag_name, t_sort, t_pretty, z, inst, path) in recs:
        time_tag_name = time_tag_name or tag_name
        timeslices[t_sort].append((z, inst, path))
        time_pretty_map[t_sort] = t_pretty

    # 時間順に並べる
    t_sorts = sorted(timeslices.keys())

    volumes: List[np.ndarray] = []
    first_ds0 = None
    spacing = None
    photometric = "MONOCHROME2"
    wc = None
    ww = None

    for i, tkey in enumerate(t_sorts):
        slices = timeslices[tkey]
        # Z位置→Instanceの順で安定ソート（簡易。厳密には IOP から法線計算推奨）
        slices.sort(key=lambda x: (x[0], x[1]))

        pixel_arrays = []
        ds0 = None
        zs = []
        for z, inst, path in slices:
            ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
            if ds0 is None:
                ds0 = ds
            try:
                arr = ds.pixel_array  # 圧縮は pylibjpeg/gdcm があれば自動展開
            except NotImplementedError as e:
                hint = (
                    "圧縮DICOMを展開できません。次をインストールしてください：\n"
                    "  pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg\n"
                    "  （または pip install gdcm）\n"
                    f"詳細: {e}"
                )
                return None, hint
            pixel_arrays.append(np.asarray(arr).astype(np.int16))
            zs.append(z)

        vol = np.stack(pixel_arrays, axis=0).astype(np.float32)  # [Z, Y, X]

        # HU変換
        slope = float(_safe_get(ds0, "RescaleSlope", 1.0))
        inter = float(_safe_get(ds0, "RescaleIntercept", 0.0))
        vol = vol * slope + inter

        # 初回でメタ設定
        if i == 0:
            first_ds0 = ds0
            photometric = str(_safe_get(ds0, "PhotometricInterpretation", "MONOCHROME2")).upper()

            # spacing
            px, py = [float(v) for v in _safe_get(ds0, "PixelSpacing", [1.0, 1.0])]
            if len(zs) >= 2:
                zspacing = float(np.abs(np.diff(zs)).mean())
            else:
                zspacing = float(_safe_get(ds0, "SliceThickness", 1.0))
            spacing = (zspacing, py, px)

            # WL/WW（配列のことがある → 先頭を採用）
            def _first_number(v, default=None):
                try:
                    if v is None:
                        return default
                    if isinstance(v, (list, tuple)):
                        return float(v[0]) if len(v) else default
                    return float(v)
                except Exception:
                    return default
            wc = _first_number(_safe_get(ds0, "WindowCenter", None))
            ww = _first_number(_safe_get(ds0, "WindowWidth", None))

        volumes.append(vol)

    # 表示用の time ラベル
    time_labels = [time_pretty_map[t] for t in t_sorts]

    # 方向行列と原点（LPS座標）
    iop = _safe_get(ds0, "ImageOrientationPatient", [1,0,0, 0,1,0])  # [rx,ry,rz, cx,cy,cz]
    row = np.array(iop[0:3], dtype=float)
    col = np.array(iop[3:6], dtype=float)
    # スライス法線（右手系）
    norm = np.cross(row, col)
    # SimpleITK は LPS 準拠なのでこのまま使える
    direction = np.array([row, col, norm]).T  # 3x3（列が軸ベクトルでも可、flatten順に注意）
    # origin は最初のスライスのIPP
    origin = np.array(_safe_get(ds0, "ImagePositionPatient", [0,0,0]), dtype=float)

    photometric = str(_safe_get(ds0, "PhotometricInterpretation", "MONOCHROME2")).upper()

    def _first_number(v, default=None):
        try:
            if v is None:
                return default
            if isinstance(v, (list, tuple)):
                return float(v[0]) if len(v) else default
            return float(v)
        except Exception:
            return default

    wc = _first_number(_safe_get(ds0, "WindowCenter", None))
    ww = _first_number(_safe_get(ds0, "WindowWidth", None))

    meta = {
        "spacing": (zspacing, py, px),                 # (Z, Y, X)
        "shape": vol.shape,                            # [Z,Y,X]
        "SeriesDescription": _safe_get(ds0, "SeriesDescription", ""),
        "Modality": _safe_get(ds0, "Modality", ""),
        "ds0": ds0,
        "photometric": photometric,
        "window_center": wc,
        "window_width": ww,
        # 追加：
        "origin": origin,                               # (x,y,z) in mm (LPS)
        "direction": direction.flatten().tolist(),      # 3x3 をフラットに
        # 時系列（あなたの最新版なら↓も既にあります）
        "time_tag": time_tag_name if 'time_tag_name' in locals() else "None",
        "time_keys": t_sorts if 't_sorts' in locals() else [0.0],
        "time_labels": time_labels if 'time_labels' in locals() else ["t=0"],
        "volumes": volumes if 'volumes' in locals() else [vol],
    }
    return SeriesData(volumes[0] if 'volumes' in locals() else vol.astype(np.float32), meta), None