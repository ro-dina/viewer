# vtk_qt_axes_test.py
import os, sys, platform
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
# Prefer software OpenGL on macOS unless explicitly disabled
if os.environ.get("VIEWER_GL_SOFTWARE", "1") == "1":
    os.environ["QT_OPENGL"] = "software"

import signal
from PySide6.QtCore import QTimer, Qt
# Ensure Qt attributes for OpenGL are set before creating QApplication
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout

import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2

from PySide6.QtGui import QSurfaceFormat
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkActor, vtkPolyDataMapper
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkFiltersSources import vtkCubeSource

def main():
    print("PY:", sys.executable)
    print("VER:", sys.version)
    print("PLAT:", platform.platform())

    # Explicit OpenGL 3.2 Core profile; Qt will fall back if unsupported
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setVersion(3, 2)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    # Set Qt attributes BEFORE creating QApplication
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    if os.environ.get("VIEWER_GL_SOFTWARE", "1") == "1":
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)

    app = QApplication(sys.argv)

    # Graceful Ctrl+C (SIGINT) handling: quit the Qt loop
    signal.signal(signal.SIGINT, lambda *args: app.quit())

    w = QWidget()
    lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0)
    vtkw = QVTKRenderWindowInteractor(w)
    lay.addWidget(vtkw)

    ren = vtkRenderer()
    ren.GradientBackgroundOn()
    ren.SetBackground(0.15, 0.15, 0.18)
    ren.SetBackground2(0.05, 0.05, 0.06)

    renWin = vtkw.GetRenderWindow()
    renWin.AddRenderer(ren)
    try:
        renWin.SetMultiSamples(0)
        renWin.SetAlphaBitPlanes(1)
    except Exception:
        pass

    # Axes
    axes = vtkAxesActor()
    axes.SetTotalLength(80, 80, 80)
    ren.AddActor(axes)

    # Cube
    cube = vtkCubeSource(); cube.SetXLength(50); cube.SetYLength(30); cube.SetZLength(20); cube.Update()
    mapper = vtkPolyDataMapper(); mapper.SetInputConnection(cube.GetOutputPort())
    actor = vtkActor(); actor.SetMapper(mapper)
    ren.AddActor(actor)

    ren.ResetCamera()

    vtkw.Initialize()
    w.resize(800, 600)
    w.setWindowTitle("VTK Qt Axes Test")
    w.show()

    # Bring window to front and ensure an initial frame is swapped
    try:
        w.raise_(); w.activateWindow()
    except Exception:
        pass

    # Defer the first render to avoid paintEvent storms and stalls on macOS
    QTimer.singleShot(0, renWin.Render)

    # Ensure we actually get a frame even if vsync stalls
    QTimer.singleShot(50, renWin.Render)

    # Kick the UI once more after shown
    QTimer.singleShot(0, QApplication.processEvents)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()