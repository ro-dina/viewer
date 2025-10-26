#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySide6 2D DICOM Viewer
- フォルダから DICOM シリーズを読み込み（時系列は未対応 / 最初のシリーズのみ）
- スライス移動 / WL / WW 調整
- 断面: Axial / Coronal / Sagittal
- 「3Dを開く」ボタンで純VTK側(app.py)を別プロセスで起動
"""
import os, sys, glob, subprocess, math
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pydicom
except Exception:
    QtWidgets.QMessageBox.critical(None, "Error", "pydicom がありません。pip install pydicom")
    sys.exit(1)

# ---------- DICOM 読み込み ----------
def load_dicom_series(dcmdir: str):
    files = sorted(glob.glob(os.path.join(dcmdir, "**", "*.dcm"), recursive=True))
    if not files:
        # 拡張子なしケース拾う
        files = [f for f in glob.glob(os.path.join(dcmdir, "**", "*"), recursive=True) if os.path.isfile(f)]
    items = []
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, force=True, stop_before_pixels=False)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        inst = int(getattr(ds, "InstanceNumber", 0))
        try:
            ipp = [float(v) for v in getattr(ds, "ImagePositionPatient")]
            z = ipp[2]
        except Exception:
            z = inst
        arr = ds.pixel_array
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = (arr.astype(np.float32) * slope + intercept).astype(np.int16)

        # spacing / origin は最初のスライスから
        if spacing == (1.0,1.0,1.0):
            try:
                py, px = [float(x) for x in getattr(ds, "PixelSpacing")]
                pz = float(getattr(ds, "SliceThickness", 1.0))
                spacing = (pz, py, px)  # (Z,Y,X)
            except Exception:
                pass
            try:
                ipp = [float(v) for v in getattr(ds, "ImagePositionPatient")]
                origin = (ipp[0], ipp[1], ipp[2])
            except Exception:
                pass

        items.append((z, arr))
    if not items:
        raise RuntimeError("有効な DICOM スライスが見つかりません。")

    items.sort(key=lambda x: x[0])
    vol = np.stack([a for _, a in items], axis=0)  # (Z,Y,X)
    return vol.astype(np.int16, copy=False), spacing, origin


# ---------- WL/WW マッピング ----------
def wlww_to_uint8(slice_i16: np.ndarray, wl: float, ww: float) -> np.ndarray:
    # window: [wl-ww/2, wl+ww/2] を 0..255 に線形マップ
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    sl = slice_i16.astype(np.float32)
    sl = (sl - low) / max(high - low, 1.0)
    sl = np.clip(sl, 0.0, 1.0)
    return (sl * 255.0).astype(np.uint8)

def robust_wl_ww(vol: np.ndarray):
    p1, p99 = np.percentile(vol, [1, 99])
    wl = float((p1 + p99) / 2.0)
    ww = float(max(p99 - p1, 1.0))
    return wl, ww


# ---------- 画像→QImage ----------
def ndarray_to_qimage(gray_u8: np.ndarray) -> QtGui.QImage:
    h, w = gray_u8.shape
    # メモリをコピーして QImage を作る（参照渡しだと寿命管理が難しい）
    buf = gray_u8.copy().tobytes()
    img = QtGui.QImage(buf, w, h, w, QtGui.QImage.Format_Grayscale8)
    return img


# ---------- メインウィジェット ----------
class Viewer2D(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D DICOM Viewer (PySide6)")
        self.resize(1100, 800)

        # 状態
        self._vol = None        # (Z,Y,X) int16
        self._spacing = (1.0,1.0,1.0)
        self._origin = (0.0,0.0,0.0)
        self._plane = "Axial"   # Axial/Coronal/Sagittal
        self._slice = 0
        self._wl = 40.0
        self._ww = 400.0
        self._dcmdir = None

        # UI 構成
        self.view = QtWidgets.QGraphicsView()
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view.setScene(self.scene)

        self.btn_open = QtWidgets.QPushButton("Open DICOM Folder…")
        self.btn_open.clicked.connect(self.on_open)

        self.btn_3d = QtWidgets.QPushButton("Open 3D Viewer (pure VTK)")
        self.btn_3d.clicked.connect(self.on_open_3d)
        self.btn_3d.setEnabled(False)

        self.cmb_plane = QtWidgets.QComboBox()
        self.cmb_plane.addItems(["Axial", "Coronal", "Sagittal"])
        self.cmb_plane.currentTextChanged.connect(self.on_plane_changed)
        self.cmb_plane.setEnabled(False)

        self.sld_slice = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_slice.valueChanged.connect(self.on_slice_changed)
        self.sld_slice.setEnabled(False)

        self.sld_wl = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_wl.setRange(-2000, 3000)
        self.sld_ww = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_ww.setRange(1, 6000)
        self.sld_wl.valueChanged.connect(self.on_wl_ww_changed)
        self.sld_ww.valueChanged.connect(self.on_wl_ww_changed)
        self.sld_wl.setEnabled(False); self.sld_ww.setEnabled(False)

        self.lbl_slice = QtWidgets.QLabel("Slice: -/-")
        self.lbl_wlww  = QtWidgets.QLabel("WL/WW: -/-")
        for l in (self.lbl_slice, self.lbl_wlww):
            l.setStyleSheet("color:#666;")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_3d)
        top.addStretch()
        top.addWidget(QtWidgets.QLabel("Plane:"))
        top.addWidget(self.cmb_plane)

        g = QtWidgets.QGridLayout()
        g.addWidget(QtWidgets.QLabel("Slice"), 0, 0); g.addWidget(self.sld_slice, 0, 1)
        g.addWidget(QtWidgets.QLabel("WL"),    1, 0); g.addWidget(self.sld_wl,    1, 1)
        g.addWidget(QtWidgets.QLabel("WW"),    2, 0); g.addWidget(self.sld_ww,    2, 1)

        info = QtWidgets.QHBoxLayout()
        info.addWidget(self.lbl_slice); info.addSpacing(20); info.addWidget(self.lbl_wlww); info.addStretch()

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        lay.addLayout(g)
        lay.addLayout(info)

        # ショートカット
        QtGui.QShortcut(QtGui.QKeySequence("A"), self, activated=lambda: self.set_plane("Axial"))
        QtGui.QShortcut(QtGui.QKeySequence("C"), self, activated=lambda: self.set_plane("Coronal"))
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, activated=lambda: self.set_plane("Sagittal"))

    # ---------- 動作 ----------
    def on_open(self):
        home = os.path.expanduser("~")
        for cand in ("Documents", "Downloads", "Desktop"):
            p = os.path.join(home, cand)
            if os.path.isdir(p):
                home = p; break
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select DICOM folder", home)
        if not d: return
        try:
            vol, sp, org = load_dicom_series(d)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))
            return
        self._vol, self._spacing, self._origin, self._dcmdir = vol, sp, org, d
        self._wl, self._ww = robust_wl_ww(vol)
        self.sld_wl.setValue(int(self._wl)); self.sld_ww.setValue(int(self._ww))
        self.cmb_plane.setEnabled(True); self.sld_slice.setEnabled(True)
        self.sld_wl.setEnabled(True); self.sld_ww.setEnabled(True); self.btn_3d.setEnabled(True)
        self.set_plane("Axial")

    def on_open_3d(self):
        if not self._dcmdir: return
        # 純VTK側 app.py を別プロセスで起動
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "app.py"), "--dir", self._dcmdir, "--viewer", "3d"]
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Launch Error", str(e))

    def on_plane_changed(self, text):
        self.set_plane(text)

    def on_slice_changed(self, v):
        self._slice = int(v)
        self.update_view()

    def on_wl_ww_changed(self, _=None):
        self._wl = float(self.sld_wl.value()); self._ww = float(max(self.sld_ww.value(), 1))
        self.update_view()

    # ---------- 断面＆表示 ----------
    def set_plane(self, plane: str):
        self._plane = plane
        z, y, x = self._vol.shape
        max_idx = {"Axial": z-1, "Coronal": y-1, "Sagittal": x-1}[plane]
        self.sld_slice.setRange(0, max_idx)
        self._slice = max_idx // 2
        self.sld_slice.setValue(self._slice)
        self.update_view()

    def current_slice(self) -> np.ndarray:
        if self._plane == "Axial":
            sl = self._vol[self._slice, :, :]
        elif self._plane == "Coronal":
            sl = self._vol[:, self._slice, :]
        else:  # Sagittal
            sl = self._vol[:, :, self._slice]
        return np.ascontiguousarray(sl)  # QImage 的に連続メモリが嬉しい

    def update_view(self):
        if self._vol is None: return
        sl = self.current_slice()
        img8 = wlww_to_uint8(sl, self._wl, self._ww)
        qimg = ndarray_to_qimage(img8)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.scene.clear()
        self.scene.addPixmap(pix)
        self.view.fitInView(self.scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        # ラベル更新
        z, y, x = self._vol.shape
        max_idx = {"Axial": z-1, "Coronal": y-1, "Sagittal": x-1}[self._plane]
        self.lbl_slice.setText(f"Slice {self._slice}/{max_idx}  ({self._plane})")
        self.lbl_wlww.setText(f"WL/WW {int(self._wl)}/{int(self._ww)}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Viewer2D()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()