#!/usr/bin/env python3
# app.py — Pure VTK viewer; Tk is used only for folder selection dialog.
import os, sys, glob, platform
import numpy as np
import argparse
import subprocess
import faulthandler
faulthandler.enable()

# --- VTK required backends ---
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkRenderingVolumeOpenGL2

from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import (
    vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor,
    vtkImageSlice, vtkImageSliceMapper, vtkTextActor, vtkTextProperty,
    vtkVolume, vtkVolumeProperty, vtkColorTransferFunction
)
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkSliderRepresentation2D, vtkSliderWidget, vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper

try:
    import pydicom
except Exception:
    print("pydicom が見つかりません。`pip install pydicom` を実行してください。")
    sys.exit(1)

# ---------- DICOM -> NumPy (Z,Y,X) ----------
def load_dicom_series(dcm_dir: str):
    files = sorted(glob.glob(os.path.join(dcm_dir, "**", "*.dcm"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(dcm_dir, "**", "*"), recursive=True))
        files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise RuntimeError(f"DICOM 候補ファイルが見つかりません: {dcm_dir}")

    slices, zpos = [], []
    spacing, origin = None, (0.0, 0.0, 0.0)

    for fp in files:
        try:
            ds = pydicom.dcmread(fp, force=True, stop_before_pixels=False)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue

        arr = ds.pixel_array
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = (arr.astype(np.float32) * slope + intercept).astype(np.int16)

        if spacing is None:
            try:
                py, px = [float(x) for x in ds.PixelSpacing]
                pz = float(getattr(ds, "SliceThickness", 1.0))
                spacing = (pz, py, px)  # (Z,Y,X)
            except Exception:
                spacing = (1.0, 1.0, 1.0)
            try:
                ipp = [float(v) for v in ds.ImagePositionPatient]
                origin = (ipp[0], ipp[1], ipp[2])
            except Exception:
                origin = (0.0, 0.0, 0.0)

        try:
            ipp = [float(v) for v in ds.ImagePositionPatient]
            z = ipp[2]
        except Exception:
            try:
                z = float(ds.SliceLocation)
            except Exception:
                z = len(slices)

        slices.append(arr)
        zpos.append(z)

    if not slices:
        raise RuntimeError("有効な DICOM スライスがありません。")

    order = np.argsort(np.array(zpos))
    vol = np.stack([slices[i] for i in order], axis=0).astype(np.int16, copy=False)
    return vol, spacing, origin

def numpy_to_vtk_image(vol: np.ndarray, spacing, origin=(0,0,0)):
    z, y, x = vol.shape
    vtk_img = vtkImageData()
    vtk_img.SetDimensions(int(x), int(y), int(z))
    vtk_img.SetSpacing(float(spacing[2]), float(spacing[1]), float(spacing[0]))  # (X,Y,Z)
    vtk_img.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    arr = np.asfortranarray(vol.transpose(2,1,0)).ravel(order="F")
    vtk_arr = numpy_to_vtk(arr, deep=True)
    vtk_arr.SetName("values")
    vtk_img.GetPointData().SetScalars(vtk_arr)
    return vtk_img

def robust_wl_ww(vol: np.ndarray):
    p1, p99 = np.percentile(vol, [1, 99])
    wl = float((p1+p99)/2.0)
    ww = float(max(p99-p1, 1.0))
    return wl, ww

# ---------- Viewer (pure VTK window) ----------
class PureVTKViewer:
    def __init__(self, vtk_img, vol_np, spacing):
        self.vtk_img = vtk_img
        self.vol_np = vol_np
        self.spacing = spacing
        self.mode_3d = False
        self.blend_mip = False
        self.plane = "Axial"

        self.wl, self.ww = robust_wl_ww(vol_np)
        self.slice_index = vol_np.shape[0] // 2

        self.ren = vtkRenderer()
        self.ren.SetBackground(0.02, 0.02, 0.03)
        self.win = vtkRenderWindow()
        self.win.AddRenderer(self.ren)
        try:
            self.win.SetMultiSamples(0)
            self.win.SetAlphaBitPlanes(1)
        except Exception:
            pass
        self.iren = vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.win)

        # Axes (orientation widget)
        self.axes_widget = vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(vtkAxesActor())
        self.axes_widget.SetInteractor(self.iren)
        self.axes_widget.SetViewport(0.0, 0.0, 0.22, 0.22)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()

        # Status text
        self.text = vtkTextActor()
        tp: vtkTextProperty = self.text.GetTextProperty()
        tp.SetFontSize(16)
        tp.SetColor(0.9, 0.9, 0.9)
        self.text.SetDisplayPosition(10, 10)
        self.ren.AddViewProp(self.text)

        # 2D slice
        self.slice_actor = vtkImageSlice()
        self.slice_mapper = vtkImageSliceMapper()
        self.slice_mapper.SetInputData(self.vtk_img)
        self.slice_actor.SetMapper(self.slice_mapper)
        sp = self.slice_actor.GetProperty()
        sp.SetColorWindow(self.ww)
        sp.SetColorLevel(self.wl)
        self.ren.AddViewProp(self.slice_actor)

        # 3D volume (CPU)
        self.ctf = vtkColorTransferFunction()
        self.otf = vtkPiecewiseFunction()
        self.vol_prop = vtkVolumeProperty()
        self.vol_prop.SetColor(self.ctf)
        self.vol_prop.SetScalarOpacity(self.otf)
        self.vol_prop.ShadeOff()
        self.vol_prop.SetInterpolationTypeToLinear()
        self.update_tf()
        self.vol_mapper = vtkFixedPointVolumeRayCastMapper()
        self.vol_mapper.SetInputData(self.vtk_img)
        self.vol_actor = vtkVolume()
        self.vol_actor.SetMapper(self.vol_mapper)
        self.vol_actor.SetProperty(self.vol_prop)

        # Sliders (Slice / WL / WW)
        self.slider_slice = self.make_slider(0, self.max_index_for_plane(), self.slice_index,
                                             0.05, 0.15, "Slice", self.on_slice_changed)
        self.slider_wl = self.make_slider(-2000, 3000, self.wl, 0.35, 0.15, "WL", self.on_wl_changed)
        self.slider_ww = self.make_slider(1, 6000, self.ww, 0.65, 0.15, "WW", self.on_ww_changed)

        self.iren.AddObserver("KeyPressEvent", self.on_key)
        self.update_plane("Axial")
        self.update_status()
        self.ren.ResetCamera()

    def make_slider(self, vmin, vmax, vinit, x_norm, y_norm, title, cb):
        rep = vtkSliderRepresentation2D()
        rep.SetMinimumValue(float(vmin))
        rep.SetMaximumValue(float(vmax))
        rep.SetValue(float(vinit))
        rep.SetTitleText(title)
        rep.GetTitleProperty().SetColor(0.9, 0.9, 0.9)
        rep.GetLabelProperty().SetColor(0.9, 0.9, 0.9)

        # VTK 9.5: SetPoint1Coordinate/SetPoint2Coordinate は存在しないため、
        # GetPointXCoordinate() で取得した座標オブジェクトに対して設定する
        p1 = rep.GetPoint1Coordinate()
        p1.SetCoordinateSystemToNormalizedDisplay()
        p1.SetValue(x_norm, y_norm)

        p2 = rep.GetPoint2Coordinate()
        p2.SetCoordinateSystemToNormalizedDisplay()
        p2.SetValue(x_norm + 0.25, y_norm)

        sl = vtkSliderWidget()
        sl.SetInteractor(self.iren)
        sl.SetRepresentation(rep)
        sl.SetAnimationModeToAnimate()
        sl.EnabledOn()
        sl.AddObserver("InteractionEvent", cb)
        return sl

    def max_index_for_plane(self):
        z,y,x = self.vol_np.shape
        return {"Axial": z-1, "Coronal": y-1}.get(self.plane, x-1)

    def apply_slice(self):
        if self.plane == "Axial":
            self.slice_mapper.SetOrientationToZ()
        elif self.plane == "Coronal":
            self.slice_mapper.SetOrientationToY()
        else:
            self.slice_mapper.SetOrientationToX()
        self.slice_mapper.SetSliceNumber(int(self.slice_index))
        self.win.Render()
        self.update_status()

    def update_plane(self, plane):
        self.plane = plane
        rep = self.slider_slice.GetRepresentation()
        rep.SetMinimumValue(0.0)
        rep.SetMaximumValue(float(self.max_index_for_plane()))
        self.slice_index = int(0.5 * self.max_index_for_plane())
        rep.SetValue(float(self.slice_index))
        self.apply_slice()

    def update_tf(self):
        low = self.wl - self.ww/2.0
        high = self.wl + self.ww/2.0
        self.ctf.RemoveAllPoints()
        self.ctf.AddRGBPoint(low, 0.0, 0.0, 0.0)
        self.ctf.AddRGBPoint(high, 1.0, 1.0, 1.0)
        self.otf.RemoveAllPoints()
        self.otf.AddPoint(low, 0.0)
        self.otf.AddPoint((low+high)/2.0, 0.2)
        self.otf.AddPoint(high, 1.0)
        sp = self.slice_actor.GetProperty()
        sp.SetColorWindow(max(self.ww, 1.0))
        sp.SetColorLevel(self.wl)

    def on_slice_changed(self, obj, ev):
        self.slice_index = int(round(obj.GetRepresentation().GetValue()))
        self.apply_slice()

    def on_wl_changed(self, obj, ev):
        self.wl = float(obj.GetRepresentation().GetValue())
        self.update_tf(); self.update_status(); self.win.Render()

    def on_ww_changed(self, obj, ev):
        self.ww = float(obj.GetRepresentation().GetValue())
        if self.ww < 1.0: self.ww = 1.0
        self.update_tf(); self.update_status(); self.win.Render()

    def on_key(self, obj, ev):
        key = self.iren.GetKeySym().lower()
        if key == "a": self.update_plane("Axial")
        elif key == "c": self.update_plane("Coronal")
        elif key == "s": self.update_plane("Sagittal")
        elif key == "v": self.toggle_volume()
        elif key == "m":
            if self.mode_3d:
                self.blend_mip = not self.blend_mip
                if self.blend_mip: self.vol_mapper.SetBlendModeToMaximumIntensity()
                else: self.vol_mapper.SetBlendModeToComposite()
                self.win.Render()
        self.update_status()

    def toggle_volume(self):
        if not self.mode_3d:
            self.ren.RemoveViewProp(self.slice_actor)
            if self.blend_mip: self.vol_mapper.SetBlendModeToMaximumIntensity()
            else: self.vol_mapper.SetBlendModeToComposite()
            self.ren.AddViewProp(self.vol_actor)
            self.mode_3d = True
        else:
            self.ren.RemoveViewProp(self.vol_actor)
            self.ren.AddViewProp(self.slice_actor)
            self.mode_3d = False
        self.ren.ResetCamera()
        self.win.Render()

    def update_status(self):
        txt = f"{self.plane} | Slice {self.slice_index}/{self.max_index_for_plane()} | WL/WW={int(self.wl)}/{int(self.ww)}"
        if self.mode_3d:
            txt += f" | 3D {'MIP' if self.blend_mip else 'Composite'}"
        self.text.SetInput(txt)

    def start(self, size=(1100, 800)):
        self.win.SetSize(*size)
        self.iren.Initialize()
        self.win.Render()
        print("Keys: A/C/S plane, V 2D<->3D, M MIP(3D). Sliders: Slice/WL/WW")
        self.iren.Start()

class VTK3DWindow:
    """3D専用（純VTK別ウィンドウ）。WL/WWはキーボードで簡易調整、MでMIP切替。"""
    def __init__(self, vtk_img, vol_np, spacing, wl=None, ww=None):
        from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkVolume, vtkVolumeProperty, vtkColorTransferFunction
        from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
        from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction

        self.vtk_img = vtk_img
        self.vol_np = vol_np
        self.spacing = spacing
        self.wl, self.ww = (wl, ww) if (wl is not None and ww is not None) else robust_wl_ww(vol_np)
        self.mip = False

        self.ren = vtkRenderer()
        self.ren.SetBackground(0.02, 0.02, 0.03)
        self.win = vtkRenderWindow()
        self.win.AddRenderer(self.ren)
        try:
            self.win.SetMultiSamples(0)
            self.win.SetAlphaBitPlanes(1)
        except Exception:
            pass
        self.iren = vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.win)

        self.ctf = vtkColorTransferFunction()
        self.otf = vtkPiecewiseFunction()
        self.prop = vtkVolumeProperty()
        self.prop.SetColor(self.ctf)
        self.prop.SetScalarOpacity(self.otf)
        self.prop.SetInterpolationTypeToLinear()
        self.prop.ShadeOff()

        self.mapper = vtkFixedPointVolumeRayCastMapper()
        self.mapper.SetInputData(self.vtk_img)
        self.actor = vtkVolume()
        self.actor.SetMapper(self.mapper)
        self.actor.SetProperty(self.prop)

        self._update_tf()
        self.ren.AddViewProp(self.actor)
        self.ren.ResetCamera()

        self.iren.AddObserver("KeyPressEvent", self._on_key)

    def _update_tf(self):
        low = self.wl - self.ww/2.0
        high = self.wl + self.ww/2.0
        self.ctf.RemoveAllPoints()
        self.ctf.AddRGBPoint(low, 0,0,0)
        self.ctf.AddRGBPoint(high, 1,1,1)
        self.otf.RemoveAllPoints()
        self.otf.AddPoint(low, 0.0)
        self.otf.AddPoint((low+high)/2.0, 0.2)
        self.otf.AddPoint(high, 1.0)

    def _on_key(self, obj, ev):
        key = self.iren.GetKeySym().lower()
        if key == 'm':
            self.mip = not self.mip
            if self.mip:
                self.mapper.SetBlendModeToMaximumIntensity()
            else:
                self.mapper.SetBlendModeToComposite()
            self.win.Render()
        elif key in ('equal', 'plus'):   # '=' / '+' でWW拡大
            self.ww *= 1.1
            self._update_tf(); self.win.Render()
        elif key in ('minus', 'underscore'):  # '-' でWW縮小
            self.ww = max(1.0, self.ww/1.1)
            self._update_tf(); self.win.Render()
        elif key == 'bracketright':  # ']' でWL+
            self.wl += 20; self._update_tf(); self.win.Render()
        elif key == 'bracketleft':   # '[' でWL-
            self.wl -= 20; self._update_tf(); self.win.Render()

    def start(self, size=(900, 700)):
        self.win.SetSize(*size)
        self.iren.Initialize()
        self.win.Render()
        print("3D window: M=MIP toggle, +/-=WW, [ / ]=WL")
        self.iren.Start()

def run_viewer_mode(vtk_img, vol, spacing, mode: str):
    if mode == "3d":
        win3d = VTK3DWindow(vtk_img, vol, spacing)
        win3d.start()
    else:
        viewer = PureVTKViewer(vtk_img, vol, spacing)
        viewer.start()

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Pure VTK DICOM viewer")
    parser.add_argument("--dir", dest="dcmdir", default=None, help="DICOM folder")
    parser.add_argument("--viewer", dest="viewer", default=None, choices=["2d3d","3d"], help="Run viewer directly (no Tk controller)")
    args = parser.parse_args()

    # 1) Determine DICOM directory
    dcmdir = args.dcmdir
    if not dcmdir:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            dcmdir = filedialog.askdirectory(title="Select DICOM folder")
            root.destroy()
        except Exception:
            dcmdir = None

    if not dcmdir:
        print("Usage: python app.py --dir <DICOMフォルダ>  [--viewer 2d3d|3d]")
        sys.exit(1)

    print("VER:", sys.version)
    print("PLAT:", platform.platform())
    print("DIR:", dcmdir)

    vol, spacing, origin = load_dicom_series(dcmdir)
    print("VOL:", vol.shape, "spacing:", spacing, "origin:", origin)
    vtk_img = numpy_to_vtk_image(vol, spacing, origin)

    # 2) If --viewer is specified, run the viewer in this main process (no Tk controller)
    if args.viewer in ("2d3d", "3d"):
        mode = "3d" if args.viewer == "3d" else "2d3d"
        run_viewer_mode(vtk_img, vol, spacing, mode)
        return

    # 3) Otherwise, show a tiny Tk controller that launches viewers as separate processes
    import tkinter as tk
    from tkinter import messagebox, filedialog

    root = tk.Tk()
    root.title("Controller")

    status = tk.StringVar(value=f"Loaded: {os.path.basename(dcmdir)}  shape={vol.shape}")

    def spawn_viewer(mode: str):
        cmd = [sys.executable, sys.argv[0], "--dir", dcmdir, "--viewer", mode]
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to start viewer: {e}")

    def open_2d3d():
        spawn_viewer("2d3d")

    def open_3d_only():
        spawn_viewer("3d")

    def reload_folder():
        nonlocal dcmdir
        newdir = filedialog.askdirectory(title="Select DICOM folder")
        if not newdir:
            return
        try:
            v2, sp2, org2 = load_dicom_series(newdir)
        except Exception as e:
            messagebox.showerror("Load Error", str(e)); return
        # Recreate vtk image
        vtk2 = numpy_to_vtk_image(v2, sp2, org2)
        # Update in-memory references
        dcmdir = newdir
        # Persist to a small cache file to allow quick relaunch with new dir
        status.set(f"Loaded: {os.path.basename(dcmdir)}  shape={v2.shape}")

    # Buttons
    tk.Button(root, text="Open 2D/3D Viewer (pure VTK)", width=28, command=open_2d3d).pack(padx=10, pady=8)
    tk.Button(root, text="Open 3D Viewer (pure VTK)", width=28, command=open_3d_only).pack(padx=10, pady=8)
    tk.Button(root, text="Open another DICOM folder…", width=28, command=reload_folder).pack(padx=10, pady=8)
    tk.Label(root, textvariable=status, anchor="w").pack(fill="x", padx=10, pady=(6,10))

    # Menu
    menubar = tk.Menu(root)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Open DICOM…", command=reload_folder)
    filemenu.add_separator()
    filemenu.add_command(label="Quit", command=root.destroy)
    menubar.add_cascade(label="File", menu=filemenu)

    viewmenu = tk.Menu(menubar, tearoff=0)
    viewmenu.add_command(label="Open 2D/3D Viewer", command=open_2d3d)
    viewmenu.add_command(label="Open 3D Viewer", command=open_3d_only)
    menubar.add_cascade(label="View", menu=viewmenu)
    root.config(menu=menubar)

    root.mainloop()

if __name__ == "__main__":
    main()