# vtk_standalone_axes.py
import sys, platform
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkActor, vtkPolyDataMapper
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkFiltersSources import vtkCubeSource
import vtkmodules.vtkInteractionStyle  # 必須: 省くと event が効かないことがある
import vtkmodules.vtkRenderingOpenGL2  # OpenGL backend

def main():
    print("VER:", sys.version)
    print("PLAT:", platform.platform())

    ren = vtkRenderer()
    ren.GradientBackgroundOn()
    ren.SetBackground(0.15, 0.15, 0.18)
    ren.SetBackground2(0.05, 0.05, 0.06)

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    try:
        renWin.SetMultiSamples(0)
        renWin.SetAlphaBitPlanes(1)
    except Exception:
        pass

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    axes = vtkAxesActor(); axes.SetTotalLength(80,80,80)
    ren.AddActor(axes)

    cube = vtkCubeSource(); cube.SetXLength(50); cube.SetYLength(30); cube.SetZLength(20); cube.Update()
    mapper = vtkPolyDataMapper(); mapper.SetInputConnection(cube.GetOutputPort())
    actor = vtkActor(); actor.SetMapper(mapper)
    ren.AddActor(actor)

    ren.ResetCamera()
    renWin.SetSize(800, 600)
    renWin.Render()
    iren.Initialize()
    iren.Start()

if __name__ == "__main__":
    main()