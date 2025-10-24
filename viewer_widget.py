# viewer_widget.py
from __future__ import annotations
from typing import Optional
import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QSize

def apply_window_level(img: np.ndarray, level: float, width: float, invert: bool=False) -> np.ndarray:
    """
    16bit相当の画素値に WL/WW を適用して uint8 に変換。
    invert=True のとき（MONOCHROME1）は出力を反転。
    """
    img = img.astype(np.float32)
    width = max(float(width), 1.0)  # 0除算回避
    low  = level - width / 2.0
    high = level + width / 2.0
    norm = (img - low) / (high - low)
    norm = np.clip(norm, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    if invert:
        u8 = 255 - u8
    return u8

class ImageView(QLabel):
    """
    画像を保持し、WL/WWで 8bit 表示。ウィンドウに自動フィット。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(QSize(200, 200))

        self._img2d: Optional[np.ndarray] = None  # 元の 2D float32（HU 等）
        self._wl: float = 40.0                    # デフォルト（後でセット）
        self._ww: float = 400.0
        self._invert: bool = False                # MONOCHROME1 のとき True

        self._orig_pixmap: Optional[QPixmap] = None

    def set_monochrome_mode(self, photometric: str):
        self._invert = (str(photometric).upper() == "MONOCHROME1")

    def set_wl_ww(self, level: float, width: float):
        self._wl = float(level)
        self._ww = max(float(width), 1.0)
        self._render()

    def set_slice(self, img2d: np.ndarray, wl: Optional[float]=None, ww: Optional[float]=None):
        if img2d.ndim != 2:
            raise ValueError("2D画像を渡してください")
        self._img2d = img2d.astype(np.float32)

        if wl is not None:
            self._wl = float(wl)
        if ww is not None:
            self._ww = max(float(ww), 1.0)

        # 値が未設定ならパーセンタイルからざっくり初期化（CTなら後で上書きされる想定）
        if wl is None or ww is None:
            p1, p99 = np.percentile(self._img2d, [1, 99])
            if ww is None:
                self._ww = max(float(p99 - p1), 1.0)
            if wl is None:
                self._wl = float((p99 + p1) / 2.0)

        self._render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _render(self):
        if self._img2d is None:
            return
        u8 = apply_window_level(self._img2d, self._wl, self._ww, invert=self._invert)
        h, w = u8.shape
        qimg = QImage(u8.data, w, h, w, QImage.Format_Grayscale8)
        self._orig_pixmap = QPixmap.fromImage(qimg.copy())
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if not self._orig_pixmap:
            return
        scaled = self._orig_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)