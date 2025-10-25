# vtk_view.py — VTK-only viewer (no PyVista)
from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer
import os

# VTK (Qt interop + core)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
    vtkColorTransferFunction,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.util.numpy_support import numpy_to_vtk
import vtk  # for VTK_* type ids


class VTKVolumeView(QWidget):
    """VTK volume view using native QVTKRenderWindowInteractor (no PyVista).
    This avoids PyVistaQt timing issues on macOS/Qt.
    """

    def show_axes_test(self):
        """Render a minimal scene (axes only) to verify the rendering path works."""
        print("[VTK] axes test render start")
        try:
            from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
            self.ren.RemoveAllViewProps()
            axes = vtkAxesActor()
            axes.SetTotalLength(50, 50, 50)
            self.ren.AddActor(axes)
            self.ren.ResetCamera()
            self.ren_win.Render()
            from PySide6.QtWidgets import QApplication
            try:
                QApplication.processEvents()
            except Exception:
                pass
            try:
                from PySide6.QtCore import QTimer as _QT
                _QT.singleShot(0, self.ren_win.Render)
            except Exception:
                pass
            print("[VTK] axes test render done")
        except Exception as e:
            print("[VTK] axes test failed:", e)

    def __init__(self, parent=None):
        print("[VTK] VTKVolumeView.__init__ start")
        super().__init__(parent)

        # Qt widget hosting VTK render window
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        print("[VTK] QVTKRenderWindowInteractor created")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.vtk_widget)

        # VTK renderer/window
        self.ren = vtkRenderer()
        self.ren.SetBackground(0.0, 0.0, 0.0)
        self.ren_win = self.vtk_widget.GetRenderWindow()
        self.ren_win.AddRenderer(self.ren)
        try:
            self.ren_win.SetMultiSamples(0)
            self.ren_win.SetAlphaBitPlanes(1)
        except Exception:
            pass
        print("[VTK] renderer attached")

        # Interactor init (synchronous for stability)
        self.iren = self.ren_win.GetInteractor()
        self._init_interactor()

        # State
        self._actor: vtkVolume | None = None
        self._wl = 40.0
        self._ww = 400.0
        self._photometric = "MONOCHROME2"
        print("[VTK] VTKVolumeView.__init__ end (deferred init)")

    def _init_interactor(self):
        print("[VTK] interactor init start")
        try:
            # Disable MSAA to reduce driver quirks
            try:
                self.ren_win.SetMultiSamples(0)
            except Exception:
                pass
            self.vtk_widget.Initialize()
            # Don't call Start() in Qt apps
            self.ren_win.Render()
            # Process queued events to keep UI responsive
            from PySide6.QtWidgets import QApplication
            try:
                QApplication.processEvents()
            except Exception:
                pass
            print("[VTK] interactor init done")
            self.show()  # ensure widget becomes visible
            self.ren_win.Render()
            try:
                self.vtk_widget.update()
                self.vtk_widget.repaint()
            except Exception:
                pass
        except Exception as e:
            print("[VTK] interactor init skipped:", e)


    # ---------------- public API ----------------
    def set_photometric(self, photometric: str):
        self._photometric = str(photometric).upper()

    def set_wl_ww(self, wl: float, ww: float):
        self._wl, self._ww = float(wl), max(float(ww), 1.0)
        if self._actor is None:
            return
        low = float(self._wl - self._ww / 2.0)
        high = float(self._wl + self._ww / 2.0)

        ctf = vtkColorTransferFunction()
        if self._photometric == "MONOCHROME1":
            ctf.AddRGBPoint(low, 1.0, 1.0, 1.0)
            ctf.AddRGBPoint(high, 0.0, 0.0, 0.0)
        else:
            ctf.AddRGBPoint(low, 0.0, 0.0, 0.0)
            ctf.AddRGBPoint(high, 1.0, 1.0, 1.0)

        otf = vtkPiecewiseFunction()
        otf.AddPoint(low, 0.0)
        otf.AddPoint((low + high) / 2.0, 0.2)
        otf.AddPoint(high, 1.0)

        prop = self._actor.GetProperty()
        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)
        try:
            self.ren_win.Render()
        except Exception:
            pass

    def set_volume(
        self,
        vol_zyx: np.ndarray,
        spacing_zyx,
        origin_xyz,
        *,
        downsample: int = 2,
        prefer_int16: bool = True,
        blending: str = "mip",
    ):
        print("[VTK] set_volume", vol_zyx.shape, spacing_zyx)

        # Guard: ensure non-empty and C-contiguous input
        if vol_zyx is None or vol_zyx.size == 0:
            raise ValueError("Empty volume passed to set_volume")
        if not vol_zyx.flags.c_contiguous:
            vol_zyx = np.ascontiguousarray(vol_zyx)

        # 1) downsample
        f = int(downsample) if downsample and downsample > 1 else 1
        if f > 1:
            vol = vol_zyx[::f, ::f, ::f].copy(order="C")
            sp = (spacing_zyx[0] * f, spacing_zyx[1] * f, spacing_zyx[2] * f)
        else:
            vol = vol_zyx
            sp = spacing_zyx
        print("[VTK] after downsample", vol.shape, sp)

        # 2) dtype
        if prefer_int16 and vol.dtype != np.int16:
            vol = vol.astype(np.int16, copy=False)
        print("[VTK] after dtype", str(vol.dtype))

        # 3) numpy -> vtkImageData (XYZ order expected)
        z, y, x = vol.shape
        sx, sy, sz = float(sp[2]), float(sp[1]), float(sp[0])
        ox, oy, oz = [float(v) for v in origin_xyz]

        vtk_img = vtkImageData()
        vtk_img.SetDimensions(int(x), int(y), int(z))
        vtk_img.SetSpacing(sx, sy, sz)
        vtk_img.SetOrigin(ox, oy, oz)

        arr = np.asfortranarray(vol.transpose(2, 1, 0)).ravel(order="F")
        vtk_type = vtk.VTK_SHORT if vol.dtype == np.int16 else vtk.VTK_FLOAT
        vtk_arr = numpy_to_vtk(arr, deep=True, array_type=vtk_type)
        vtk_arr.SetName("values")
        vtk_img.GetPointData().SetScalars(vtk_arr)
        print("[VTK] vtk image ready")

        # 4) transfer functions from WL/WW
        low = float(self._wl - self._ww / 2.0)
        high = float(self._wl + self._ww / 2.0)
        ctf = vtkColorTransferFunction()
        if self._photometric == "MONOCHROME1":
            ctf.AddRGBPoint(low, 1.0, 1.0, 1.0)
            ctf.AddRGBPoint(high, 0.0, 0.0, 0.0)
        else:
            ctf.AddRGBPoint(low, 0.0, 0.0, 0.0)
            ctf.AddRGBPoint(high, 1.0, 1.0, 1.0)
        otf = vtkPiecewiseFunction()
        otf.AddPoint(low, 0.0)
        otf.AddPoint((low + high) / 2.0, 0.2)
        otf.AddPoint(high, 1.0)

        prop = vtkVolumeProperty()
        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)
        prop.ShadeOff()
        prop.SetInterpolationTypeToLinear()

        # 4.5) choose mapper: default to CPU fixed-point on macOS to avoid GPU stalls
        use_cpu = True
        if os.environ.get("VIEWER_VTK_GPU", "0") == "1":
            use_cpu = False
        try:
            if not use_cpu:
                # Try GPU via SmartVolumeMapper
                mapper = vtkSmartVolumeMapper()
                mapper.SetInputData(vtk_img)
                if blending == "mip":
                    mapper.SetBlendModeToMaximumIntensity()
                else:
                    mapper.SetBlendModeToComposite()
                try:
                    mapper.SetRequestedRenderModeToGPU()
                except Exception:
                    pass
                # coarse sampling for stability/perf
                avg_voxel = float((sp[0] + sp[1] + sp[2]) / 3.0)
                try:
                    mapper.SetAutoAdjustSampleDistances(False)
                    mapper.SetSampleDistance(max(avg_voxel * 5.0, 1.0))
                except Exception:
                    pass
            else:
                # CPU software mapper (very robust on macOS)
                from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
                mapper = vtkFixedPointVolumeRayCastMapper()
                mapper.SetInputData(vtk_img)
                if blending == "mip":
                    mapper.SetBlendModeToMaximumIntensity()
                else:
                    mapper.SetBlendModeToComposite()
        except Exception:
            # last resort: fall back to SmartVolumeMapper default path
            mapper = vtkSmartVolumeMapper()
            mapper.SetInputData(vtk_img)
            if blending == "mip":
                mapper.SetBlendModeToMaximumIntensity()
            else:
                mapper.SetBlendModeToComposite()

        volume = vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(prop)

        # Debug axes to verify scene renders even if volume is invisible
        try:
            from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
            axes = vtkAxesActor()
            axes.SetTotalLength(50, 50, 50)
            self.ren.AddActor(axes)
        except Exception:
            pass

        # 5) render (synchronous, onscreen)
        try:
            self.ren.RemoveAllViewProps()
        except Exception:
            pass
        self.ren.AddVolume(volume)
        self._actor = volume
        self.ren.ResetCamera()
        try:
            # Ensure onscreen rendering
            try:
                self.ren_win.SetOffScreenRendering(False)
            except Exception:
                pass
            self.ren_win.Render()
        except Exception as e:
            print("[VTK] render skipped:", e)
        from PySide6.QtWidgets import QApplication
        try:
            QApplication.processEvents()
            self.vtk_widget.update()
            self.vtk_widget.repaint()
        except Exception:
            pass
        print("[VTK] end render (vtk native)")