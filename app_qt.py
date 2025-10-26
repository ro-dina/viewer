#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySide6 2D DICOM Viewer (triple-orthogonal)
- フォルダから DICOM シリーズを読み込み
- Axial(Z) / Coronal(Y) / Sagittal(X) を同時表示（各ペインの表示ON/OFFに対応）
- それぞれ独立スライス、WL/WW 共通
- 各ペインに『他2面の位置』ガイド線を表示（ガイド線も面ごとにON/OFF可）
- 「Open 3D Viewer」ボタンで純VTK(app.py)を別プロセスで起動
"""
import os, sys, glob, subprocess
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid
import argparse

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
    buf = gray_u8.copy().tobytes()  # 寿命管理のためコピー
    img = QtGui.QImage(buf, w, h, w, QtGui.QImage.Format_Grayscale8)
    return img


class OrthoView(QtWidgets.QGraphicsView):
    """QGraphicsView ラッパ（簡単な便利設定）"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self.setBackgroundBrush(QtGui.QColor(18,18,22))


class Viewer2D(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D DICOM Viewer (PySide6) - Ortho Triple")
        self.resize(1500, 900)

        # 状態
        self._vol = None        # (Z,Y,X) int16
        self._spacing = (1.0,1.0,1.0)
        self._origin = (0.0,0.0,0.0)
        # 各面のスライス index
        self._idx_z = 0  # Axial (Z)
        self._idx_y = 0  # Coronal (Y)
        self._idx_x = 0  # Sagittal (X)
        # WL/WW 共通
        self._wl = 40.0
        self._ww = 400.0
        self._dcmdir = None

        # ===== UI =====
        # 上段：操作バー
        self.btn_open = QtWidgets.QPushButton("Open DICOM Folder…")
        self.btn_open.clicked.connect(self.on_open)
        self.btn_3d = QtWidgets.QPushButton("Open 3D Viewer (pure VTK)")
        self.btn_3d.clicked.connect(self.on_open_3d)
        self.btn_3d.setEnabled(False)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_3d)
        top.addStretch()

        # 可視/ガイドのトグル群
        self.chk_show_z = QtWidgets.QCheckBox("Show Z (Axial)");    self.chk_show_z.setChecked(True)
        self.chk_show_y = QtWidgets.QCheckBox("Show Y (Coronal)");  self.chk_show_y.setChecked(True)
        self.chk_show_x = QtWidgets.QCheckBox("Show X (Sagittal)"); self.chk_show_x.setChecked(True)
        self.chk_guid_z = QtWidgets.QCheckBox("Guide on Z");       self.chk_guid_z.setChecked(True)
        self.chk_guid_y = QtWidgets.QCheckBox("Guide on Y");       self.chk_guid_y.setChecked(True)
        self.chk_guid_x = QtWidgets.QCheckBox("Guide on X");       self.chk_guid_x.setChecked(True)
        for c in (self.chk_show_z, self.chk_show_y, self.chk_show_x, self.chk_guid_z, self.chk_guid_y, self.chk_guid_x):
            c.setEnabled(False)
            c.stateChanged.connect(self.update_views)

        tog = QtWidgets.QHBoxLayout()
        tog.addWidget(self.chk_show_z); tog.addWidget(self.chk_guid_z)
        tog.addSpacing(12)
        tog.addWidget(self.chk_show_y); tog.addWidget(self.chk_guid_y)
        tog.addSpacing(12)
        tog.addWidget(self.chk_show_x); tog.addWidget(self.chk_guid_x)
        tog.addStretch()

        # 中段：3ペイン（Z / Y / X）
        self.view_z = OrthoView(); self.scene_z = QtWidgets.QGraphicsScene(self); self.view_z.setScene(self.scene_z)
        self.view_y = OrthoView(); self.scene_y = QtWidgets.QGraphicsScene(self); self.view_y.setScene(self.scene_y)
        self.view_x = OrthoView(); self.scene_x = QtWidgets.QGraphicsScene(self); self.view_x.setScene(self.scene_x)

        self.view_z.setMinimumSize(200, 200)
        self.view_y.setMinimumSize(200, 200)
        self.view_x.setMinimumSize(200, 200)

        three = QtWidgets.QHBoxLayout()
        three.addWidget(self.view_z, 1)
        three.addWidget(self.view_y, 1)
        three.addWidget(self.view_x, 1)

        # 下段：スライダ群（Z / Y / X / WL / WW）
        self.sld_z = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_z.valueChanged.connect(self.on_slice_z); self.sld_z.setEnabled(False)
        self.sld_y = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_y.valueChanged.connect(self.on_slice_y); self.sld_y.setEnabled(False)
        self.sld_x = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_x.valueChanged.connect(self.on_slice_x); self.sld_x.setEnabled(False)

        self.sld_wl = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_wl.setRange(-2000, 3000); self.sld_wl.valueChanged.connect(self.on_wl_ww)
        self.sld_ww = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_ww.setRange(1, 6000);   self.sld_ww.valueChanged.connect(self.on_wl_ww)
        self.sld_wl.setEnabled(False); self.sld_ww.setEnabled(False)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Z (Axial)"), 0, 0); grid.addWidget(self.sld_z, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Y (Coronal)"), 1, 0); grid.addWidget(self.sld_y, 1, 1)
        grid.addWidget(QtWidgets.QLabel("X (Sagittal)"), 2, 0); grid.addWidget(self.sld_x, 2, 1)
        grid.addWidget(QtWidgets.QLabel("WL"), 3, 0); grid.addWidget(self.sld_wl, 3, 1)
        grid.addWidget(QtWidgets.QLabel("WW"), 4, 0); grid.addWidget(self.sld_ww, 4, 1)

        # 情報ラベル
        self.lbl_z = QtWidgets.QLabel("Z: -/-")
        self.lbl_y = QtWidgets.QLabel("Y: -/-")
        self.lbl_x = QtWidgets.QLabel("X: -/-")
        self.lbl_wlww = QtWidgets.QLabel("WL/WW: -/-")
        for l in (self.lbl_z, self.lbl_y, self.lbl_x, self.lbl_wlww):
            l.setStyleSheet("color:#888;")

        info = QtWidgets.QHBoxLayout()
        info.addWidget(self.lbl_z); info.addSpacing(12)
        info.addWidget(self.lbl_y); info.addSpacing(12)
        info.addWidget(self.lbl_x); info.addSpacing(24)
        info.addWidget(self.lbl_wlww); info.addStretch()

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(tog)
        lay.addLayout(three)
        lay.addLayout(grid)
        lay.addLayout(info)

        # ガイド線保持
        self._line_z = []  # Axial 画面上のガイド（Y水平 / X垂直）
        self._line_y = []  # Coronal 画面上のガイド（Z水平 / X垂直）
        self._line_x = []  # Sagittal 画面上のガイド（Z水平 / Y垂直）

    # ---------- ファイルオープン ----------
    def on_open(self):
        home = os.path.expanduser("~")
        for cand in ("Documents", "Downloads", "Desktop"):
            p = os.path.join(home, cand)
            if os.path.isdir(p):
                home = p
                break
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select DICOM folder", home)
        if not d:
            return
        self.load_dir(d)

    def load_dir(self, d):
        try:
            vol, sp, org = load_dicom_series(d)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))
            return
        self._vol, self._spacing, self._origin, self._dcmdir = vol, sp, org, d
        # デフォルトはパーセンタイルから
        self._wl, self._ww = robust_wl_ww(vol)

        z, y, x = vol.shape
        self._idx_z = z//2; self._idx_y = y//2; self._idx_x = x//2

        # スライダ範囲セット
        self.sld_z.setRange(0, z-1); self.sld_z.setValue(self._idx_z); self.sld_z.setEnabled(True)
        self.sld_y.setRange(0, y-1); self.sld_y.setValue(self._idx_y); self.sld_y.setEnabled(True)
        self.sld_x.setRange(0, x-1); self.sld_x.setValue(self._idx_x); self.sld_x.setEnabled(True)
        self.sld_wl.setValue(int(self._wl)); self.sld_ww.setValue(int(self._ww))
        self.sld_wl.setEnabled(True); self.sld_ww.setEnabled(True)
        self.btn_3d.setEnabled(True)
        for c in (self.chk_show_z, self.chk_show_y, self.chk_show_x, self.chk_guid_z, self.chk_guid_y, self.chk_guid_x):
            c.setEnabled(True)

        self.update_views()

    # ---------- 3D 起動 ----------
    def on_open_3d(self):
        if not self._dcmdir: return
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "app.py"), "--dir", self._dcmdir, "--viewer", "3d"]
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Launch Error", str(e))

    # ---------- スライス操作 ----------
    def on_slice_z(self, v):
        self._idx_z = int(v); self.update_views()
    def on_slice_y(self, v):
        self._idx_y = int(v); self.update_views()
    def on_slice_x(self, v):
        self._idx_x = int(v); self.update_views()
    def on_wl_ww(self, _=None):
        self._wl = float(self.sld_wl.value()); self._ww = float(max(self.sld_ww.value(), 1)); self.update_views()

    # ---------- スライス取得 ----------
    def _slice_z(self):
        return np.ascontiguousarray(self._vol[self._idx_z, :, :])  # (Y,X)
    def _slice_y(self):
        return np.ascontiguousarray(self._vol[:, self._idx_y, :])  # (Z,X)
    def _slice_x(self):
        return np.ascontiguousarray(self._vol[:, :, self._idx_x])  # (Z,Y)

    # ---------- ガイド描画ユーティリティ ----------
    def _clear_lines(self, lst, scene):
        # 安全に削除（既にQt側で破棄されている可能性に配慮）
        for it in list(lst):
            try:
                if isValid(it) and it.scene() is scene:
                    scene.removeItem(it)
            except Exception:
                pass
        lst.clear()

    def _add_line(self, scene, x1,y1,x2,y2, color=QtGui.QColor(0,255,0)):
        pen = QtGui.QPen(color, 2)
        return scene.addLine(x1,y1,x2,y2, pen)

    # ---------- 描画更新 ----------
    def update_views(self):
        if self._vol is None:
            return
        z, y, x = self._vol.shape

        # 可視切り替え
        self.view_z.setVisible(self.chk_show_z.isChecked())
        self.view_y.setVisible(self.chk_show_y.isChecked())
        self.view_x.setVisible(self.chk_show_x.isChecked())

        # Z (Axial)
        self.scene_z.clear(); self._line_z.clear()
        if self.chk_show_z.isChecked():
            sl = wlww_to_uint8(self._slice_z(), self._wl, self._ww)
            h, w = sl.shape
            pix = QtGui.QPixmap.fromImage(ndarray_to_qimage(sl))
            self.scene_z.addPixmap(pix)
            self.scene_z.setSceneRect(pix.rect())
            self.view_z.fitInView(self.scene_z.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
            self.view_z.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
            if self.chk_guid_z.isChecked():
                # Y 定数 -> 水平線、X 定数 -> 垂直線
                self._line_z.append(self._add_line(self.scene_z, 0, self._idx_y, w, self._idx_y))
                self._line_z.append(self._add_line(self.scene_z, self._idx_x, 0, self._idx_x, h))

        # Y (Coronal) 画像は (Z, X)
        self.scene_y.clear(); self._line_y.clear()
        if self.chk_show_y.isChecked():
            sl = wlww_to_uint8(self._slice_y(), self._wl, self._ww)
            h_y, w_y = sl.shape
            pix = QtGui.QPixmap.fromImage(ndarray_to_qimage(sl))
            self.scene_y.addPixmap(pix)
            self.scene_y.setSceneRect(pix.rect())
            self.view_y.fitInView(self.scene_y.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
            self.view_y.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
            if self.chk_guid_y.isChecked():
                # Z 定数 -> 水平線（row=z）、X 定数 -> 垂直線（col=x）
                self._line_y.append(self._add_line(self.scene_y, 0, self._idx_z, w_y, self._idx_z))
                self._line_y.append(self._add_line(self.scene_y, self._idx_x, 0, self._idx_x, h_y))

        # X (Sagittal) 画像は (Z, Y)
        self.scene_x.clear(); self._line_x.clear()
        if self.chk_show_x.isChecked():
            sl = wlww_to_uint8(self._slice_x(), self._wl, self._ww)
            h_x, w_x = sl.shape
            pix = QtGui.QPixmap.fromImage(ndarray_to_qimage(sl))
            self.scene_x.addPixmap(pix)
            self.scene_x.setSceneRect(pix.rect())
            self.view_x.fitInView(self.scene_x.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
            self.view_x.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
            if self.chk_guid_x.isChecked():
                # Z 定数 -> 水平線（row=z）、Y 定数 -> 垂直線（col=y）
                self._line_x.append(self._add_line(self.scene_x, 0, self._idx_z, w_x, self._idx_z))
                self._line_x.append(self._add_line(self.scene_x, self._idx_y, 0, self._idx_y, h_x))

        # ラベル
        self.lbl_z.setText(f"Z: {self._idx_z}/{z-1}")
        self.lbl_y.setText(f"Y: {self._idx_y}/{y-1}")
        self.lbl_x.setText(f"X: {self._idx_x}/{x-1}")
        self.lbl_wlww.setText(f"WL/WW {int(self._wl)}/{int(self._ww)}")


def main():
    parser = argparse.ArgumentParser(description="PySide6 2D DICOM Viewer (ortho)")
    parser.add_argument("--dir", dest="dcmdir", default=None, help="DICOM folder")
    args, unknown = parser.parse_known_args()
    app = QtWidgets.QApplication(sys.argv)
    w = Viewer2D()
    if args.dcmdir:
        w.load_dir(args.dcmdir)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()