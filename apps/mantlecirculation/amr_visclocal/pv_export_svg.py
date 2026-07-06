"""Export the clipped half-shell scenes as TRUE VECTOR SVG (GL2PS) so the mesh cells stay sharp at any zoom
-- unlike SaveScreenshot which rasterises. Adaptive mesh + solution are cell-count-manageable; the uniform
L2 mesh (~2.5M cells after clip) may be huge/slow, so it is attempted last and its size is reported."""
import glob, os, re
from paraview.simple import (
    XDMFReader, Clip, Show, Delete, Render, CreateView, ExportView, ColorBy, GetColorTransferFunction,
)

OUT = os.path.expanduser(os.environ.get("VISCLOCAL_OUT", "~/visclocal_out"))
PV  = os.path.join(OUT, "pv"); os.makedirs(PV, exist_ok=True)

rounds = sorted(glob.glob(os.path.join(OUT, "round*_mesh/step_0.xmf")),
                key=lambda p: int(re.search(r"round(\d+)_mesh", p).group(1)))
uni = os.path.join(OUT, "uni2_mesh/step_0.xmf")
ada = rounds[-1]

NORMAL = [0.5071, -0.8451, 0.0]
nh = [c / (0.5071**2 + 0.8451**2) ** 0.5 for c in NORMAL]
view = CreateView("RenderView")
view.ViewSize = [1200, 1200]
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

# adaptive first (feasible), uniform last (may be huge)
jobs = [(ada, "level", None, "grid_adaptive"), (ada, "u", "Magnitude", "sol_adaptive"),
        (uni, "level", None, "grid_uniform")]
for xmf, field, comp, out in jobs:
    if not os.path.exists(xmf):
        print("MISSING", xmf, flush=True); continue
    reader = XDMFReader(FileNames=[xmf]); reader.PointArrayStatus = ["level", "u"]; reader.UpdatePipeline()
    ncells = reader.GetDataInformation().GetNumberOfCells()
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
    svg = os.path.join(PV, out + ".svg")
    print(f"exporting {out}  ({ncells} cells) ...", flush=True)
    ExportView(svg, view=view, Rasterize3Dgeometry=0, GL2PSdepthsortmethod="BSP sorting (slow, best)")
    sz = os.path.getsize(svg) / 1e6 if os.path.exists(svg) else -1
    print(f"  -> {svg}  ({sz:.1f} MB)", flush=True)
    Delete(disp); Delete(cut); Delete(reader)
print("Done.", flush=True)
