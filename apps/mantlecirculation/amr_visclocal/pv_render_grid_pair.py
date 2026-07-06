"""Render for the uniform-vs-adaptive triptych/quad, at high resolution (1600^2) so the panels stay sharp
when zoomed inside an SVG: uniform mesh (uni2_mesh) + adaptive mesh (finest round*_mesh) coloured by
subdivision `level`, and the adaptive |u| solution. PNGs in $VISCLOCAL_OUT/pv. Clipped half-shell."""
import glob, os, re
from paraview.simple import (
    XDMFReader, Clip, Show, Delete, Render, CreateView, SaveScreenshot, ColorBy, GetColorTransferFunction,
)

OUT = os.path.expanduser(os.environ.get("VISCLOCAL_OUT", "~/visclocal_out"))
PV  = os.path.join(OUT, "pv"); os.makedirs(PV, exist_ok=True)
RES = [3000, 3000]  # high-res raster: true-vector 3D mesh SVG isn't possible (GL2PS rasterizes), so oversample

rounds = sorted(glob.glob(os.path.join(OUT, "round*_mesh/step_0.xmf")),
                key=lambda p: int(re.search(r"round(\d+)_mesh", p).group(1)))
uni = os.path.join(OUT, "uni2_mesh/step_0.xmf")
ada = rounds[-1]

NORMAL = [0.5071, -0.8451, 0.0]
nh = [c / (0.5071**2 + 0.8451**2) ** 0.5 for c in NORMAL]
view = CreateView("RenderView")
view.ViewSize = RES
view.UseColorPaletteForBackground = 0
view.BackgroundColorMode = "Single Color"
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 0
view.CameraFocalPoint = [0.0, 0.0, 0.0]
view.CameraPosition   = [-3.1 * nh[0], -3.1 * nh[1], 0.0]
view.CameraViewUp     = [0.0, 0.0, 1.0]

LVL = [[0.82, 0.82, 0.82], [0.55, 0.78, 0.90], [0.99, 0.78, 0.45], [0.92, 0.55, 0.52], [0.62, 0.40, 0.66]]
lvl_lut = GetColorTransferFunction("level")
lvl_lut.InterpretValuesAsCategories = 1
lvl_lut.AnnotationsInitialized = 1
lvl_lut.Annotations = ["0", "0", "1", "1", "2", "2", "3", "3", "4", "4"]
lvl_lut.IndexedColors = [c for rgb in LVL for c in rgb]
lvl_lut.IndexedOpacities = [1.0] * 5

rf = XDMFReader(FileNames=[ada]); rf.UpdatePipeline()
umax = rf.GetDataInformation().GetPointDataInformation().GetArrayInformation("u").GetComponentRange(-1)[1]
Delete(rf)
u_lut = GetColorTransferFunction("u")
u_lut.RGBPoints = [0.0, 0.86, 0.86, 0.86, 0.5 * umax, 0.99, 0.73, 0.40, umax, 0.80, 0.20, 0.16]
u_lut.ColorSpace = "RGB"

# (xmf, field, magnitude?, out)
jobs = [(uni, "level", None, "grid_uniform"), (ada, "level", None, "grid_adaptive"),
        (ada, "u", "Magnitude", "sol_adaptive")]
for xmf, field, comp, out in jobs:
    if not os.path.exists(xmf):
        print("MISSING", xmf, flush=True); continue
    reader = XDMFReader(FileNames=[xmf]); reader.PointArrayStatus = ["level", "u"]; reader.UpdatePipeline()
    cut = Clip(Input=reader); cut.ClipType = "Plane"
    cut.ClipType.Origin = [0.0, 0.0, 0.0]; cut.ClipType.Normal = NORMAL
    cut.Invert = 0; cut.Crinkleclip = 1
    disp = Show(cut, view); disp.Representation = "Surface With Edges"
    disp.EdgeColor = [0.35, 0.35, 0.35]; disp.LineWidth = 0.4
    disp.Ambient = 1.0; disp.Diffuse = 0.0; disp.Specular = 0.0
    disp.SetScalarBarVisibility(view, False)
    if comp:
        ColorBy(disp, ("POINTS", field, comp)); u_lut.RescaleTransferFunction(0.0, umax)
    else:
        ColorBy(disp, ("POINTS", field))
    Render(view)
    png = os.path.join(PV, out + ".png"); SaveScreenshot(png, view, ImageResolution=RES)
    print("  ->", png, flush=True)
    Delete(disp); Delete(cut); Delete(reader)
print("Done.", flush=True)
