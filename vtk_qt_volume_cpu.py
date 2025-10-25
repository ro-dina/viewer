# vtk_qt_volume_cpu.py
import os, sys, platform
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
# os.environ.setdefault("QT_OPENGL", "software")  # うまくいかない時だけ

import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PySide6.QtGui import QSurfaceFormat
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkVolume, vtkVolumeProperty, vtkColorTransferFunction
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper  # CPU マッパ

def make_synthetic_volume(shape=(64, 128, 128)):
    """中心に球とリングを入れた合成ボリューム（int16）。"""
    z, y, x = [np.arange(n) for n in shape]
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    cz, cy, cx = (s/2 for s in shape)
    r = np.sqrt(((Z-cz)/18)**2 + ((Y-cy)/25)**2 + ((X-cx)/25)**2)
    vol = np.clip((1.0 - r)*2000, 0, None)  # 0..2000
    # 中央に球
    r2 = np.sqrt(((Z-cz)/10)**2 + ((Y-cy)/10)**2 + ((X-cx)/10)**2)
    vol += (r2 < 1.0)*1500
    # int16 へ
    return vol.astype(np.int16)

def main():
    print("PY:", sys.executable)
    print("VER:", sys.version)
    print("PLAT:", platform.platform())

    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setVersion(3, 2)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0)
    vtkw = QVTKRenderWindowInteractor(w)
    lay.addWidget(vtkw)

    ren = vtkRenderer()
    ren.SetBackground(0.0, 0.0, 0.0)
    renWin = vtkw.GetRenderWindow()
    renWin.AddRenderer(ren)
    try:
        renWin.SetMultiSamples(0)
        renWin.SetAlphaBitPlanes(1)
    except Exception:
        pass

    # --- ボクセル作成（Z,Y,X） ---
    vol = make_synthetic_volume()           # int16, だいたい CTっぽい値域
    spacing = (2.0, 2.5, 2.5)               # (Z,Y,X)
    z,y,x = vol.shape
    # VTK 用（X,Y,Z）
    vtk_img = vtkImageData()
    vtk_img.SetDimensions(int(x), int(y), int(z))
    vtk_img.SetSpacing(float(spacing[2]), float(spacing[1]), float(spacing[0]))
    vtk_img.SetOrigin(0.0, 0.0, 0.0)

    # NumPy → VTK（X,Y,Zに転置し Fortran 並びで詰める）
    arr = np.asfortranarray(vol.transpose(2,1,0)).ravel(order="F")
    vtk_arr = numpy_to_vtk(arr, deep=True)  # int16 を自動判定
    vtk_arr.SetName("values")
    vtk_img.GetPointData().SetScalars(vtk_arr)

    # --- 転送関数（WL/WW 的に 200/1500 くらい） ---
    wl, ww = 800.0, 1500.0
    low, high = wl - ww/2, wl + ww/2

    ctf = vtkColorTransferFunction()
    ctf.AddRGBPoint(low, 0.0, 0.0, 0.0)
    ctf.AddRGBPoint(high, 1.0, 1.0, 1.0)

    otf = vtkPiecewiseFunction()
    otf.AddPoint(low, 0.0)
    otf.AddPoint((low+high)/2.0, 0.2)
    otf.AddPoint(high, 1.0)

    prop = vtkVolumeProperty()
    prop.SetColor(ctf)
    prop.SetScalarOpacity(otf)
    prop.ShadeOff()
    prop.SetInterpolationTypeToLinear()

    # --- CPU マッパ（超安定） ---
    mapper = vtkFixedPointVolumeRayCastMapper()
    mapper.SetInputData(vtk_img)
    # Composite / うまくいかない時は MIP に変更:
    # mapper.SetBlendModeToMaximumIntensity()

    volume = vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)
    ren.AddVolume(volume)
    ren.ResetCamera()

    vtkw.Initialize()
    w.resize(900, 700)
    w.setWindowTitle("VTK Qt Volume (CPU)")
    w.show()
    renWin.Render()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()