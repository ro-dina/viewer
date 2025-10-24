# app.py （差分：Time スライダと t 対応。丸ごと置き換え推奨）
from __future__ import annotations
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget,
    QVBoxLayout, QLabel, QStatusBar, QSlider, QHBoxLayout, QComboBox
)
from PySide6.QtCore import Qt
from viewer_widget import ImageView
from dicom_io import load_series_from_folder
import numpy as np

def _make_slider(minv, maxv, step=1):
    s = QSlider(Qt.Horizontal)
    s.setRange(minv, maxv)
    s.setSingleStep(step)
    s.setPageStep(step * 5)
    s.setEnabled(False)
    return s

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM TimeSeries Viewer (MVP)")
        self.resize(1150, 900)

        central = QWidget(self)
        self.setCentralWidget(central)
        self.vlayout = QVBoxLayout(central)

        self.info_label = QLabel("File > Open Folder からDICOMフォルダを選んでください。")
        self.viewer = ImageView()

        # --- Plane selector ---
        self.plane_layout = QHBoxLayout()
        self.plane_label = QLabel("Plane:")
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["Axial (Z)", "Coronal (Y)", "Sagittal (X)"])
        self.plane_combo.currentIndexChanged.connect(self.on_plane_changed)
        self.plane_layout.addWidget(self.plane_label)
        self.plane_layout.addWidget(self.plane_combo, 1)
        self.plane_layout.addStretch()

        # --- Time slider ---
        self.t_layout = QHBoxLayout()
        self.t_label = QLabel("Time: -")
        self.t_slider = _make_slider(0, 0)
        self.t_slider.valueChanged.connect(self.on_t_changed)
        self.t_layout.addWidget(QLabel("Time"))
        self.t_layout.addWidget(self.t_slider, 1)
        self.t_layout.addWidget(self.t_label)

        # --- Index (Z/Y/X) slider ---
        self.z_layout = QHBoxLayout()
        self.z_label = QLabel("Index: -")
        self.z_slider = _make_slider(0, 0)
        self.z_slider.valueChanged.connect(self.on_z_changed)
        self.z_layout.addWidget(QLabel("Index"))
        self.z_layout.addWidget(self.z_slider, 1)
        self.z_layout.addWidget(self.z_label)

        # --- WL/WW sliders ---
        self.wl_layout = QHBoxLayout()
        self.wl_label = QLabel("WL: -")
        self.wl_slider = _make_slider(-4000, 4000, step=1)
        self.wl_slider.valueChanged.connect(self.on_wl_changed)
        self.wl_layout.addWidget(QLabel("Level"))
        self.wl_layout.addWidget(self.wl_slider, 1)
        self.wl_layout.addWidget(self.wl_label)

        self.ww_layout = QHBoxLayout()
        self.ww_label = QLabel("WW: -")
        self.ww_slider = _make_slider(1, 8000, step=1)
        self.ww_slider.valueChanged.connect(self.on_ww_changed)
        self.ww_layout.addWidget(QLabel("Width"))
        self.ww_layout.addWidget(self.ww_slider, 1)
        self.ww_layout.addWidget(self.ww_label)

        # Layout
        self.vlayout.addWidget(self.info_label)
        self.vlayout.addLayout(self.plane_layout)
        self.vlayout.addLayout(self.t_layout)
        self.vlayout.addLayout(self.z_layout)
        self.vlayout.addLayout(self.wl_layout)
        self.vlayout.addLayout(self.ww_layout)
        self.vlayout.addWidget(self.viewer, 1)

        self.setStatusBar(QStatusBar())

        # Menu
        menu = self.menuBar().addMenu("&File")
        act_open = menu.addAction("Open &Folder...")
        act_open.triggered.connect(self.on_open_folder)
        act_quit = menu.addAction("&Quit")
        act_quit.triggered.connect(self.close)

        # State
        self._series = None
        self._plane = "Axial"
        self._idx = 0
        self._idx_max = 0
        self._t = 0
        self._tmax = 0
        self._wl = 40.0
        self._ww = 400.0

    # ---------- helpers ----------
    def _current_volume(self):
        if not self._series:
            return None
        vols = self._series.meta.get("volumes")
        if vols and len(vols) > 0:
            return vols[self._t]
        return self._series.volume  # fallback (tなし)

    def _current_slice2d(self):
        vol = self._current_volume()
        if vol is None:
            return None
        if self._plane == "Axial":
            return vol[self._idx, :, :]
        elif self._plane == "Coronal":
            return vol[:, self._idx, :]
        elif self._plane == "Sagittal":
            return vol[:, :, self._idx]
        else:
            raise ValueError("Unknown plane")

    def _reset_index_range(self):
        vol = self._current_volume()
        if vol is None:
            return
        z, y, x = vol.shape
        length = {"Axial": z, "Coronal": y, "Sagittal": x}[self._plane]
        self._idx_max = max(0, length - 1)
        self._idx = length // 2
        self.z_slider.blockSignals(True)
        self.z_slider.setRange(0, self._idx_max)
        self.z_slider.setValue(self._idx)
        self.z_slider.setEnabled(length > 1)
        self.z_slider.blockSignals(False)

    # ---------- file open ----------
    def on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            return
        series, err = load_series_from_folder(folder)
        if err:
            QMessageBox.warning(self, "Load Error", err)
            return
        self._series = series

        # 時系列初期化
        vols = self._series.meta.get("volumes") or [self._series.volume]
        self._t = 0
        self._tmax = len(vols) - 1
        self.t_slider.blockSignals(True)
        self.t_slider.setRange(0, self._tmax)
        self.t_slider.setValue(self._t)
        self.t_slider.setEnabled(self._tmax > 0)
        self.t_slider.blockSignals(False)

        # 面とインデックス初期化
        self._plane = "Axial"
        self.plane_combo.blockSignals(True)
        self.plane_combo.setCurrentIndex(0)
        self.plane_combo.blockSignals(False)
        self._reset_index_range()

        # Photometric & WL/WW初期値
        photometric = series.meta.get("photometric", "MONOCHROME2")
        if hasattr(self.viewer, "set_monochrome_mode"):
            self.viewer.set_monochrome_mode(photometric)

        wc = series.meta.get("window_center")
        ww = series.meta.get("window_width")
        if wc is None or ww is None:
            sl = self._current_slice2d()
            p1, p99 = np.percentile(sl, [1, 99])
            ww = float(ww) if ww is not None else float(max(p99 - p1, 1.0))
            wc = float(wc) if wc is not None else float((p99 + p1) / 2.0)
        self._wl, self._ww = float(wc), max(float(ww), 1.0)

        self.wl_slider.blockSignals(True)
        self.wl_slider.setValue(int(round(self._wl)))
        self.wl_slider.setEnabled(True)
        self.wl_slider.blockSignals(False)

        self.ww_slider.blockSignals(True)
        self.ww_slider.setValue(int(round(self._ww)))
        self.ww_slider.setEnabled(True)
        self.ww_slider.blockSignals(False)

        # Info
        time_labels = self._series.meta.get("time_labels", [])
        time_tag = self._series.meta.get("time_tag", "None")
        z, y, x = vols[0].shape
        self.info_label.setText(
            f"Loaded: T={len(vols)} [{time_tag}] | shape=[Z,Y,X]=[{z},{y},{x}] "
            f"spacing={series.meta.get('spacing')} Modality={series.meta.get('Modality','')} "
            f"Series='{series.meta.get('SeriesDescription','')}'"
        )
        self._update_time_label()
        self._update_slice()

        ds0 = self._series.meta.get("ds0")
        tsuid = getattr(getattr(ds0, "file_meta", None), "TransferSyntaxUID", None)
        self.statusBar().showMessage(f"OK: loaded 1 series | TSUID={tsuid}")

    # ---------- events ----------
    def on_plane_changed(self, idx: int):
        self._plane = ["Axial", "Coronal", "Sagittal"][idx]
        self._reset_index_range()
        self._update_slice()

    def on_t_changed(self, value: int):
        self._t = int(value)
        # 同じ index が範囲外になる可能性があるため、面のレンジを再計算
        self._reset_index_range()
        self._update_time_label()
        self._update_slice()

    def on_z_changed(self, value: int):
        self._idx = int(value)
        self._update_slice()

    def on_wl_changed(self, value: int):
        self._wl = float(value)
        self.wl_label.setText(f"WL: {self._wl:.0f}")
        self.viewer.set_wl_ww(self._wl, self._ww)

    def on_ww_changed(self, value: int):
        self._ww = max(float(value), 1.0)
        self.ww_label.setText(f"WW: {self._ww:.0f}")
        self.viewer.set_wl_ww(self._wl, self._ww)

    # ---------- render ----------
    def _update_time_label(self):
        labels = self._series.meta.get("time_labels", [])
        if 0 <= self._t < len(labels):
            self.t_label.setText(f"Time: {self._t}/{self._tmax}  ({labels[self._t]})")
        else:
            self.t_label.setText(f"Time: {self._t}/{self._tmax}")

    def _update_slice(self):
        if not self._series:
            return
        self.z_label.setText(f"Index: {self._idx}/{self._idx_max}  ({self._plane})")
        sl = self._current_slice2d()
        self.wl_label.setText(f"WL: {self._wl:.0f}")
        self.ww_label.setText(f"WW: {self._ww:.0f}")
        self.viewer.set_slice(sl, wl=self._wl, ww=self._ww)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()