# -*- coding: utf-8 -*-
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import threading
import traceback
import os, sys, json, time
import urllib.request

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_DIR        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(APP_DIR, 'checkpoints')
SAM2_CKPT      = os.path.join(CHECKPOINT_DIR, 'sam2_hiera_small.pt')
SAM2_URL       = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
SAM2_CONFIGS   = ['configs/sam2.1/sam2.1_hiera_s', 'sam2.1_hiera_s', 'configs/sam2/sam2_hiera_s']
SETTINGS_FILE  = os.path.join(APP_DIR, 'settings.json')
CRASH_LOG      = os.path.join(APP_DIR, 'crash.log')

for _sub in ('ZoeDepth', 'MiDaS'):
    _p = os.path.join(APP_DIR, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

LAYER_COLORS = [
    (255, 61,  90,  "Layer 1"),
    (0,   229, 255, "Layer 2"),
    (170, 255, 0,   "Layer 3"),
    (255, 149, 0,   "Layer 4"),
    (191, 95,  255, "Layer 5"),
    (255, 230, 0,   "Layer 6"),
    (0,   255, 178, "Layer 7"),
    (255, 107, 202, "Layer 8"),
]

# ── Crash logging ──────────────────────────────────────────────────────────────

def _log(label, exc_type, exc_value, exc_tb):
    try:
        with open(CRASH_LOG, 'a', encoding='utf-8') as f:
            f.write(f'\n{"="*60}\n[{label}]\n')
            f.writelines(traceback.format_exception(exc_type, exc_value, exc_tb))
    except Exception:
        pass

_orig_hook = sys.excepthook
def _excepthook(t, v, tb):
    _log('MAIN', t, v, tb); _orig_hook(t, v, tb)
sys.excepthook = _excepthook

def _thread_hook(args):
    _log(f'THREAD:{getattr(args.thread,"name","?")}',
         args.exc_type, args.exc_value, args.exc_tb)
threading.excepthook = _thread_hook

# ── DnD ───────────────────────────────────────────────────────────────────────

try:
    import tkinterdnd2; _HAS_TKDND = True
except ImportError:
    _HAS_TKDND = False
try:
    import windnd; _HAS_WINDND = True
except ImportError:
    _HAS_WINDND = False

# ── Pure helpers ───────────────────────────────────────────────────────────────

def _checkerboard(w, h, ts=10):
    arr = np.full((h, w, 4), 40, dtype=np.uint8)
    arr[:, :, 3] = 255
    rows = np.arange(h) // ts
    cols = np.arange(w) // ts
    arr[(rows[:, None] + cols[None, :]) % 2 == 0] = [62, 62, 62, 255]
    return arr


def _feather_mask(mask, radius=6):
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if radius < 1:
        return binary
    k       = np.ones((radius * 2 + 1,) * 2, np.uint8)
    inner   = cv2.erode(binary, k)
    edge    = cv2.subtract(binary, inner)
    bk      = max(3, radius * 4 + 1) | 1
    blurred = cv2.GaussianBlur(edge.astype(np.float32), (bk, bk), radius * 0.5)
    return np.clip(np.maximum(inner.astype(np.float32), blurred), 0, 255).astype(np.uint8)


def _morpho_clean(mask):
    _, b = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return b


def _sample_points(mask, n=10):
    """Grid-distributed foreground points for SAM2 prompts."""
    ys, xs = np.nonzero(mask > 10)
    if len(xs) == 0:
        return None, None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pts = [(int(xs.mean()), int(ys.mean()))]
    side = max(2, int(np.sqrt(n)) + 1)
    for gy in np.linspace(y0, y1, side):
        for gx in np.linspace(x0, x1, side):
            iy = min(int(gy), mask.shape[0] - 1)
            ix = min(int(gx), mask.shape[1] - 1)
            if mask[iy, ix] > 10:
                p = (ix, iy)
                if p not in pts:
                    pts.append(p)
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    pts = pts[:n]
    return np.array(pts, dtype=np.float32), np.ones(len(pts), dtype=np.int32)


# ── Checkpoint download ────────────────────────────────────────────────────────

def _ensure_checkpoint(progress_cb=None):
    if os.path.exists(SAM2_CKPT):
        return True
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # Try HuggingFace hub first
    try:
        from huggingface_hub import hf_hub_download
        if progress_cb:
            progress_cb(0.05, "Baixando SAM2 via HuggingFace...")
        hf_hub_download(
            repo_id="facebook/sam2-hiera-small",
            filename="sam2_hiera_small.pt",
            local_dir=CHECKPOINT_DIR,
        )
        if os.path.exists(SAM2_CKPT):
            return True
    except Exception:
        pass
    # Direct URL fallback
    try:
        tmp = SAM2_CKPT + ".tmp"
        def _hook(count, block, total):
            if total > 0 and progress_cb:
                frac = min(0.99, count * block / total)
                progress_cb(frac, f"Baixando SAM2: {int(frac*100)}%")
        urllib.request.urlretrieve(SAM2_URL, tmp, reporthook=_hook)
        os.rename(tmp, SAM2_CKPT)
        return True
    except Exception as e:
        _log('DOWNLOAD', type(e), e, e.__traceback__)
        return False


# ── Application ────────────────────────────────────────────────────────────────

class ParallaxStudio(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Parallax Studio")
        self.geometry("1520x960")
        self.minsize(1100, 680)

        # Image / mask state
        self.image_path   = None
        self.orig_image   = None        # PIL RGBA
        self.num_layers   = 3
        self.active_layer = 0
        self.masks        = [None] * 8  # raw painted  uint8 H×W
        self.sam2_masks   = [None] * 8  # SAM2-refined uint8 H×W | None
        self._undo        = [None] * 8  # (painted, sam2) snapshots
        self._auto_masks  = None        # None=not yet run, []=ran/empty, [...]= results
        self._depth_map   = None        # float32 H×W, 1=near 0=far
        self._show_depth  = False

        # Canvas / paint
        self._base_cache  = {}
        self.tk_image     = None
        self.zoom         = 1.0
        self.is_painting  = False
        self.last_x = self.last_y = None
        self.brush_size   = 30
        self.tool         = "brush"
        self._brush_oval  = None
        self._thumb_refs  = []
        self._lasso_pts    = []   # [(canvas_x, canvas_y), ...] while drawing lasso
        self._scribble_pts = []   # [(img_x, img_y), ...] accumulated across strokes
        self._scribble_job = None # after() id for debounced SAM2
        self._last_render_t = 0.0 # monotonic time of last fast render (throttle)

        # Models
        self._sam2_model     = None
        self._sam2_predictor = None
        self._sam2_auto_gen  = None
        self._depth_model    = None
        self._sdxl_pipe      = None

        # Locks / flags
        self._sam2_lock        = threading.Lock()
        self._depth_lock       = threading.Lock()
        self._predictor_lock   = threading.Lock()  # serialize predictor.set_image + predict
        self._preprocess_lock  = threading.Lock()
        self._preprocessing_active = False
        self._sam2_running     = False
        self._sam2_refine_job  = None

        # Settings vars (created before _build_ui)
        self.feather_var   = ctk.StringVar(value="Media")
        self.proc_size_var = ctk.StringVar(value="1024")
        self._inpaint_mode = ctk.StringVar(value="OpenCV")
        self.output_dir    = None
        self._load_settings()

        self._build_ui()
        self._setup_dnd()
        self.bind('<Control-z>', self._do_undo)
        self.bind('<Control-Z>', self._do_undo)

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_settings(self):
        try:
            d = json.load(open(SETTINGS_FILE, encoding='utf-8'))
            self.output_dir = d.get('output_dir')
        except Exception:
            pass

    def _save_settings(self):
        try:
            json.dump({'output_dir': self.output_dir},
                      open(SETTINGS_FILE, 'w', encoding='utf-8'))
        except Exception:
            pass

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_canvas_area()
        self._build_results_panel()

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=258, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)
        sb.grid_rowconfigure(99, weight=1)
        r = 0

        ctk.CTkLabel(sb, text="PARALLAX STUDIO",
                     font=ctk.CTkFont(size=13, weight="bold")
                     ).grid(row=r, column=0, padx=16, pady=(18, 2), sticky="w"); r += 1
        ctk.CTkLabel(sb, text="SAM2 + ZoeDepth",
                     font=ctk.CTkFont(size=11), text_color="gray"
                     ).grid(row=r, column=0, padx=16, pady=(0, 10), sticky="w"); r += 1

        ctk.CTkButton(sb, text="Carregar Imagem",
                      command=self._browse_image
                      ).grid(row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        self._sep(sb, r); r += 1

        ctk.CTkLabel(sb, text="Camadas:", font=ctk.CTkFont(size=12)
                     ).grid(row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        self._layer_seg = ctk.CTkSegmentedButton(
            sb, values=["2", "3", "4", "5", "6", "7", "8"],
            command=self._on_layer_count_change)
        self._layer_seg.set("3")
        self._layer_seg.grid(row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        ctk.CTkLabel(sb, text="Layer ativa:", font=ctk.CTkFont(size=12)
                     ).grid(row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1
        self.layer_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self.layer_frame.grid(row=r, column=0, padx=12, pady=2, sticky="ew"); r += 1
        self._rebuild_layer_buttons()

        self._sep(sb, r); r += 1

        ctk.CTkLabel(sb, text="Ferramenta:", font=ctk.CTkFont(size=12)
                     ).grid(row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        self.tool_var = ctk.StringVar(value="brush")
        tf = ctk.CTkFrame(sb, fg_color="transparent")
        tf.grid(row=r, column=0, padx=16, pady=3, sticky="w"); r += 1
        for val, lbl in [("brush", "Pincel"),
                         ("eraser", "Borracha"),
                         ("scribble", "Pincel de Objeto"),
                         ("lasso", "Contorno (Lasso)")]:
            ctk.CTkRadioButton(tf, text=lbl, variable=self.tool_var, value=val,
                               command=lambda v=val: setattr(self, 'tool', v)
                               ).pack(anchor="w", pady=1)

        ctk.CTkLabel(sb, text="Tamanho do pincel:", font=ctk.CTkFont(size=12)
                     ).grid(row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1
        self.brush_slider = ctk.CTkSlider(sb, from_=4, to=300,
                                          command=self._on_brush_change)
        self.brush_slider.set(30)
        self.brush_slider.grid(row=r, column=0, padx=12, pady=2, sticky="ew"); r += 1
        self.brush_label = ctk.CTkLabel(sb, text="30px", font=ctk.CTkFont(size=11))
        self.brush_label.grid(row=r, column=0, padx=16, sticky="w"); r += 1

        self._sep(sb, r); r += 1

        ctk.CTkLabel(sb, text="Resolucao SAM2:", font=ctk.CTkFont(size=12)
                     ).grid(row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        ctk.CTkSegmentedButton(sb, values=["512", "1024", "2048"],
                               variable=self.proc_size_var
                               ).grid(row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        ctk.CTkLabel(sb, text="Suavizacao de bordas:", font=ctk.CTkFont(size=12)
                     ).grid(row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        ctk.CTkSegmentedButton(sb, values=["Fina", "Media", "Suave"],
                               variable=self.feather_var
                               ).grid(row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        self._sep(sb, r); r += 1

        self._depth_btn = ctk.CTkButton(sb, text="Ver Depth Map",
                                        fg_color="transparent", border_width=1,
                                        command=self._toggle_depth)
        self._depth_btn.grid(row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        ctk.CTkButton(sb, text="Selecao Inteligente",
                      fg_color="#7c3aed", hover_color="#6d28d9",
                      command=self._smart_select
                      ).grid(row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        self._sep(sb, r); r += 1

        ctk.CTkLabel(sb, text="Inpainting:", font=ctk.CTkFont(size=12)
                     ).grid(row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        ctk.CTkSegmentedButton(sb, values=["SDXL", "OpenCV", "Desligado"],
                               variable=self._inpaint_mode
                               ).grid(row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        self._sep(sb, r); r += 1

        ctk.CTkLabel(sb, text="Zoom:", font=ctk.CTkFont(size=12)
                     ).grid(row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        zf = ctk.CTkFrame(sb, fg_color="transparent")
        zf.grid(row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1
        ctk.CTkButton(zf, text="-", width=34,
                      command=lambda: self._set_zoom(self.zoom - 0.1)).pack(side="left", padx=1)
        self.zoom_label = ctk.CTkLabel(zf, text="100%", width=46)
        self.zoom_label.pack(side="left", padx=2)
        ctk.CTkButton(zf, text="+", width=34,
                      command=lambda: self._set_zoom(self.zoom + 0.1)).pack(side="left", padx=1)
        ctk.CTkButton(zf, text="Fit", width=50,
                      command=self._zoom_to_fit).pack(side="left", padx=(4, 0))

        bf = ctk.CTkFrame(sb, fg_color="transparent")
        bf.grid(row=r, column=0, padx=12, pady=(10, 3), sticky="ew"); r += 1
        bf.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(bf, text="Limpar", fg_color="transparent", border_width=1,
                      command=self._clear_layer
                      ).grid(row=0, column=0, padx=(0, 2), sticky="ew")
        ctk.CTkButton(bf, text="Desfazer", fg_color="transparent", border_width=1,
                      command=self._do_undo
                      ).grid(row=0, column=1, padx=(2, 0), sticky="ew")

        self._sep(sb, r); r += 1

        ctk.CTkLabel(sb, text="Salvar em:", font=ctk.CTkFont(size=12)
                     ).grid(row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        od = ctk.CTkFrame(sb, fg_color="transparent")
        od.grid(row=r, column=0, padx=12, pady=2, sticky="ew"); r += 1
        od.grid_columnconfigure(0, weight=1)
        self._out_label = ctk.CTkLabel(
            od, text=self._short_path(self.output_dir or "Mesma pasta"),
            font=ctk.CTkFont(size=10), text_color="gray", wraplength=155, anchor="w")
        self._out_label.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(od, text="...", width=34,
                      command=self._choose_output_dir
                      ).grid(row=0, column=1, padx=(4, 0))

        self._sep(sb, r); r += 1
        ctk.CTkButton(sb, text="Processar com IA",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      fg_color="#5eead4", text_color="#000", hover_color="#2dd4bf",
                      command=self._process
                      ).grid(row=r, column=0, padx=12, pady=(10, 10), sticky="ew"); r += 1

        self.status_label = ctk.CTkLabel(
            sb, text="Carregue uma imagem\nou arraste aqui",
            font=ctk.CTkFont(size=11), text_color="gray", wraplength=225)
        self.status_label.grid(row=99, column=0, padx=12, pady=10, sticky="sw")

    def _sep(self, parent, row):
        ctk.CTkFrame(parent, height=1, fg_color="#222233"
                     ).grid(row=row, column=0, padx=12, pady=4, sticky="ew")

    def _build_canvas_area(self):
        self._canvas_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#08080f")
        self._canvas_frame.grid(row=0, column=1, sticky="nsew")
        self._canvas_frame.grid_rowconfigure(0, weight=1)
        self._canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self._canvas_frame, bg="#08080f",
                                cursor="crosshair", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.vscroll = ctk.CTkScrollbar(self._canvas_frame, command=self.canvas.yview)
        self.vscroll.grid(row=0, column=1, sticky="ns")
        self.hscroll = ctk.CTkScrollbar(self._canvas_frame, orientation="horizontal",
                                        command=self.canvas.xview)
        self.hscroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=self.vscroll.set,
                              xscrollcommand=self.hscroll.set)

        self._drop_hint = tk.Label(
            self.canvas,
            text="Arraste uma imagem aqui\nou use Carregar Imagem",
            fg="#3a3a5c", bg="#08080f", font=("Segoe UI", 16), justify="center")
        self._drop_hint.place(relx=0.5, rely=0.5, anchor="center")

        for ev, fn in [("<ButtonPress-1>",   self._on_mouse_down),
                       ("<B1-Motion>",       self._on_mouse_drag),
                       ("<ButtonRelease-1>", self._on_mouse_up),
                       ("<Enter>",           self._on_canvas_enter),
                       ("<Motion>",          self._on_canvas_motion),
                       ("<Leave>",           self._on_canvas_leave),
                       ("<MouseWheel>",      self._on_scroll_zoom),
                       ("<Button-4>",        self._on_scroll_zoom),
                       ("<Button-5>",        self._on_scroll_zoom)]:
            self.canvas.bind(ev, fn)

        # Full-canvas blocking overlay
        self._ov_bg = tk.Frame(self._canvas_frame, bg="#08081a")
        for ev in ("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>", "<MouseWheel>"):
            self._ov_bg.bind(ev, lambda e: "break")

        self._ov = ctk.CTkFrame(self._canvas_frame, corner_radius=18,
                                fg_color="#0d0d22", border_width=1,
                                border_color="#2e2e55")
        self._ov_icon  = ctk.CTkLabel(self._ov, text="",
                                      font=ctk.CTkFont(size=44))
        self._ov_icon.pack(pady=(28, 4))
        self._ov_title = ctk.CTkLabel(self._ov, text="...", text_color="#5eead4",
                                      font=ctk.CTkFont(size=16, weight="bold"))
        self._ov_title.pack(pady=(0, 6))
        self._ov_body  = ctk.CTkLabel(self._ov, text="",
                                      font=ctk.CTkFont(size=12), text_color="#aaa",
                                      wraplength=380, justify="center")
        self._ov_body.pack(pady=(0, 14), padx=30)
        self._ov_bar   = ctk.CTkProgressBar(self._ov, width=340)
        self._ov_bar.pack(pady=(0, 28), padx=30)
        self._ov_bar.set(0)

    def _build_results_panel(self):
        frame = ctk.CTkFrame(self, width=265, corner_radius=0, fg_color="#0b0b18")
        frame.grid(row=0, column=2, sticky="nsew")
        frame.grid_propagate(False)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text="RESULTADOS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#444466"
                     ).grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")
        self._results_scroll = ctk.CTkScrollableFrame(frame, fg_color="transparent")
        self._results_scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._results_scroll.grid_columnconfigure(0, weight=1)

    # ── Overlay ────────────────────────────────────────────────────────────────

    def _overlay_show(self, icon, title, body, progress):
        def _do():
            self._ov_icon.configure(text=icon)
            self._ov_title.configure(text=title)
            self._ov_body.configure(text=body)
            self._ov_bar.set(max(0.0, min(1.0, progress)))
            self._ov_bg.place(x=0, y=0, relwidth=1, relheight=1)
            self._ov_bg.lift()
            self._ov.place(relx=0.5, rely=0.5, anchor="center")
            self._ov.lift()
            self._preprocessing_active = True
        self.after(0, _do)

    def _overlay_hide(self):
        def _do():
            self._ov.place_forget()
            self._ov_bg.place_forget()
            self._preprocessing_active = False
        self.after(0, _do)

    # ── Layer buttons ──────────────────────────────────────────────────────────

    def _rebuild_layer_buttons(self):
        for w in self.layer_frame.winfo_children():
            w.destroy()
        for i in range(self.num_layers):
            rv, gv, bv, name = LAYER_COLORS[i]
            hc = f"#{rv:02x}{gv:02x}{bv:02x}"
            has_s2 = self.sam2_masks[i] is not None and self.sam2_masks[i].max() > 0
            has_p  = self.masks[i] is not None and self.masks[i].max() > 0
            suffix = " [SAM2]" if has_s2 else (" [pincel]" if has_p else "")
            ctk.CTkButton(
                self.layer_frame,
                text=name + suffix,
                fg_color=hc if i == self.active_layer else "transparent",
                text_color="#000" if i == self.active_layer else "#fff",
                border_color=hc, border_width=2,
                command=lambda idx=i: self._set_active_layer(idx)
            ).pack(fill="x", pady=1)

    def _set_active_layer(self, idx):
        self.active_layer = idx
        self._rebuild_layer_buttons()

    def _on_layer_count_change(self, val):
        self.num_layers = int(val)
        self.active_layer = min(self.active_layer, self.num_layers - 1)
        self._rebuild_layer_buttons()

    # ── DnD ────────────────────────────────────────────────────────────────────

    def _setup_dnd(self):
        if _HAS_TKDND:
            try:
                tkinterdnd2.TkinterDnD._require(self)
                self.canvas.drop_target_register(tkinterdnd2.DND_FILES)
                self.canvas.dnd_bind('<<Drop>>', self._on_tkdnd_drop)
                return
            except Exception as e:
                _log('DND_SETUP', type(e), e, e.__traceback__)
        if _HAS_WINDND:
            try:
                windnd.hook_dropfiles(self, func=self._on_windnd_drop)
            except Exception as e:
                _log('WINDND_SETUP', type(e), e, e.__traceback__)

    def _on_tkdnd_drop(self, event):
        try:
            paths = self.tk.splitlist(event.data)
            if paths:
                threading.Thread(target=self._load_image_bg,
                                 args=(paths[0].strip().strip('"'),),
                                 daemon=True).start()
        except Exception as e:
            _log('TKDND_DROP', type(e), e, e.__traceback__)
        return event.action

    def _on_windnd_drop(self, files):
        try:
            if not files:
                return
            raw  = files[0]
            path = raw.decode('mbcs', errors='replace') if isinstance(raw, bytes) else str(raw)
            threading.Thread(target=self._load_image_bg,
                             args=(path.strip().strip('"'),), daemon=True).start()
        except Exception as e:
            _log('WINDND_DROP', type(e), e, e.__traceback__)

    # ── Image loading ──────────────────────────────────────────────────────────

    def _browse_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if path:
            threading.Thread(target=self._load_image_bg, args=(path,), daemon=True).start()

    def _load_image_bg(self, path):
        path = path.strip().strip('"').strip("'")
        if not os.path.isfile(path):
            return
        self.after(0, lambda: self._update_status("Carregando..."))
        try:
            image = Image.open(path).convert("RGBA")
            W, H  = image.size
            masks = [np.zeros((H, W), dtype=np.uint8) for _ in range(8)]
        except Exception as e:
            self.after(0, lambda err=e: self._update_status(f"Erro: {err}"))
            return

        def _apply():
            self.image_path  = path
            self.orig_image  = image
            self.masks       = masks
            self.sam2_masks  = [None] * 8
            self._undo       = [None] * 8
            self._auto_masks = None
            self._depth_map  = None
            self._show_depth = False
            self._base_cache = {}
            self._scribble_pts = []
            self.canvas.delete("scribble")
            self.canvas.delete("lasso")
            self._drop_hint.place_forget()
            self._update_status(f"{os.path.basename(path)}\n{W}x{H}px")
            self._zoom_to_fit()
            threading.Thread(target=self._preprocess_image, daemon=True).start()

        self.after(0, _apply)

    # ── Zoom ───────────────────────────────────────────────────────────────────

    def _zoom_to_fit(self):
        if not self.orig_image:
            return
        self.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        W, H = self.orig_image.size
        self._set_zoom(round(min(cw / W, ch / H, 1.0), 2))

    def _set_zoom(self, z):
        self.zoom = max(0.05, min(8.0, round(z, 2)))
        self.zoom_label.configure(text=f"{int(self.zoom*100)}%")
        self._base_cache = {}
        if self.orig_image:
            self._render_canvas()

    def _on_scroll_zoom(self, event):
        if self.orig_image is None:
            return
        delta = 0.1 if (event.num == 4 or event.delta > 0) else -0.1
        self._set_zoom(self.zoom + delta)

    # ── Canvas rendering ───────────────────────────────────────────────────────

    def _effective_mask(self, i):
        """Union of SAM2 + painted masks so brush strokes are always visible."""
        s = self.sam2_masks[i]
        m = self.masks[i]
        if s is not None and s.max() > 0:
            if m is not None and m.max() > 0:
                return np.maximum(s, m)
            return s
        return m

    def _render_canvas(self, fast=False):
        if not self.orig_image:
            return
        # Throttle fast/paint renders to ~30 fps
        if fast:
            now = time.monotonic()
            if now - self._last_render_t < 0.033:
                return
            self._last_render_t = now

        W, H = self.orig_image.size
        dw = max(1, int(W * self.zoom))
        dh = max(1, int(H * self.zoom))

        z_key = self.zoom
        if z_key not in self._base_cache:
            src = self.orig_image
            if self._show_depth and self._depth_map is not None:
                d8   = (self._depth_map * 255).astype(np.uint8)
                dcol = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
                drgb = cv2.cvtColor(dcol, cv2.COLOR_BGR2RGB)
                orig = np.array(self.orig_image.convert("RGB"))
                if drgb.shape != orig.shape:
                    drgb = cv2.resize(drgb, (W, H))
                blended = (orig * 0.45 + drgb * 0.55).astype(np.uint8)
                src = Image.fromarray(blended).convert("RGBA")
            base_pil = src.resize((dw, dh), Image.BILINEAR if fast else Image.LANCZOS)
            if not fast:
                self._base_cache = {z_key: base_pil}
        else:
            base_pil = self._base_cache[z_key]

        arr = np.array(base_pil.convert("RGB"), dtype=np.float32)
        for i in range(self.num_layers):
            m = self._effective_mask(i)
            if m is None or m.max() == 0:
                continue
            mr = cv2.resize(m, (dw, dh),
                            interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            rv, gv, bv, _ = LAYER_COLORS[i]
            alpha = mr * 0.52
            arr[:, :, 0] = arr[:, :, 0] * (1 - alpha) + rv * alpha
            arr[:, :, 1] = arr[:, :, 1] * (1 - alpha) + gv * alpha
            arr[:, :, 2] = arr[:, :, 2] * (1 - alpha) + bv * alpha

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(arr.astype(np.uint8)))
        self.canvas.delete("img")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image, tags="img")
        self.canvas.configure(scrollregion=(0, 0, dw, dh))
        self.canvas.tag_raise("lasso")
        self.canvas.tag_raise("scribble")
        self.canvas.tag_raise("cursor")

    # ── Brush cursor ───────────────────────────────────────────────────────────

    def _on_canvas_motion(self, e):
        cx = self.canvas.canvasx(e.x)
        cy = self.canvas.canvasy(e.y)
        if self.tool == "lasso":
            if self._brush_oval:
                self.canvas.delete(self._brush_oval)
                self._brush_oval = None
            return
        r = max(2, int(self.brush_size * self.zoom))
        if self.tool == "scribble":
            outline, dash, width = "#00e5ff", (), 2
        elif self.tool == "brush":
            outline, dash, width = "white", (4, 4), 1
        else:
            outline, dash, width = "#ff6060", (4, 4), 1
        if self._brush_oval:
            self.canvas.coords(self._brush_oval, cx - r, cy - r, cx + r, cy + r)
            self.canvas.itemconfigure(self._brush_oval, outline=outline,
                                      dash=dash, width=width)
        else:
            self._brush_oval = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=outline, width=width, dash=dash, tags="cursor")

    def _on_canvas_leave(self, e):
        if self._brush_oval:
            self.canvas.delete(self._brush_oval)
            self._brush_oval = None

    # ── Painting ───────────────────────────────────────────────────────────────

    def _canvas_to_img(self, ex, ey):
        x = int(self.canvas.canvasx(ex) / self.zoom)
        y = int(self.canvas.canvasy(ey) / self.zoom)
        if self.orig_image:
            W, H = self.orig_image.size
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
        return x, y

    def _paint_at(self, x, y):
        mask = self.masks[self.active_layer]
        r    = max(1, self.brush_size)
        if self.tool == "brush":
            cv2.circle(mask, (x, y), r, 255, -1)
        elif self.tool == "eraser":
            cv2.circle(mask, (x, y), int(r * 1.6), 0, -1)
            # Erase SAM2 mask in same region so eraser is immediately visible
            s = self.sam2_masks[self.active_layer]
            if s is not None:
                cv2.circle(s, (x, y), int(r * 1.6), 0, -1)

    def _on_mouse_down(self, e):
        if self.orig_image is None or self._preprocessing_active:
            return
        if self.tool == "lasso":
            cx = self.canvas.canvasx(e.x)
            cy = self.canvas.canvasy(e.y)
            self._lasso_pts = [(cx, cy)]
            self.canvas.delete("lasso")
            m = self.masks[self.active_layer]
            s = self.sam2_masks[self.active_layer]
            if m is not None:
                self._undo[self.active_layer] = (m.copy(), s.copy() if s is not None else None)
            self.is_painting = True
            return
        if self.tool == "scribble":
            # First stroke on this layer → save undo
            if not self._scribble_pts:
                m = self.masks[self.active_layer]
                s = self.sam2_masks[self.active_layer]
                if m is not None:
                    self._undo[self.active_layer] = (
                        m.copy(), s.copy() if s is not None else None)
            x, y = self._canvas_to_img(e.x, e.y)
            self._scribble_pts.append((x, y))
            self.is_painting = True
            self.last_x, self.last_y = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
            return
        x, y = self._canvas_to_img(e.x, e.y)
        # Undo snapshot
        m = self.masks[self.active_layer]
        s = self.sam2_masks[self.active_layer]
        if m is not None:
            self._undo[self.active_layer] = (m.copy(), s.copy() if s is not None else None)
        self.is_painting = True
        self.last_x, self.last_y = x, y
        self._paint_at(x, y)
        self._render_canvas(fast=True)

    def _on_mouse_drag(self, e):
        self._on_canvas_motion(e)
        if not self.is_painting or self._preprocessing_active:
            return
        if self.tool == "lasso":
            cx = self.canvas.canvasx(e.x)
            cy = self.canvas.canvasy(e.y)
            if self._lasso_pts:
                px, py = self._lasso_pts[-1]
                self.canvas.create_line(px, py, cx, cy,
                                        fill="#a855f7", width=2,
                                        tags="lasso")
            self._lasso_pts.append((cx, cy))
            return
        if self.tool == "scribble":
            cx = self.canvas.canvasx(e.x)
            cy = self.canvas.canvasy(e.y)
            # Draw stroke on canvas
            if self.last_x is not None:
                self.canvas.create_line(self.last_x, self.last_y, cx, cy,
                                        fill="#00e5ff", width=max(3, int(self.brush_size * self.zoom * 0.6)),
                                        capstyle=tk.ROUND, joinstyle=tk.ROUND,
                                        tags="scribble")
            self.last_x, self.last_y = cx, cy
            # Collect image-space points
            ix, iy = self._canvas_to_img(e.x, e.y)
            self._scribble_pts.append((ix, iy))
            return
        x, y = self._canvas_to_img(e.x, e.y)
        if self.last_x is not None:
            dx, dy = x - self.last_x, y - self.last_y
            dist   = max(1, int((dx**2 + dy**2) ** 0.5))
            step   = max(1, self.brush_size // 4)
            for i in range(0, dist, step):
                t = i / dist
                self._paint_at(int(self.last_x + dx * t), int(self.last_y + dy * t))
        self.last_x, self.last_y = x, y
        self._render_canvas(fast=True)

    def _on_mouse_up(self, e):
        if not self.is_painting:
            return
        self.is_painting = False
        self.last_x = self.last_y = None
        if self.tool == "lasso":
            pts = self._lasso_pts
            if len(pts) > 4:
                fx, fy = pts[0]
                lx, ly = pts[-1]
                self.canvas.create_line(lx, ly, fx, fy,
                                        fill="#a855f7", width=2, dash=(4, 2),
                                        tags="lasso")
                self._update_status("Contorno detectado — SAM2 refinando...")
                layer = self.active_layer
                threading.Thread(target=self._run_lasso_sam2,
                                 args=(list(pts), layer),
                                 daemon=True).start()
            else:
                self.canvas.delete("lasso")
                self._lasso_pts = []
            return
        if self.tool == "scribble":
            if len(self._scribble_pts) > 2:
                self._schedule_scribble_sam2(self.active_layer)
            return
        self._render_canvas()
        # Brush and eraser just paint — no SAM2 auto-refine

    def _on_canvas_enter(self, e):
        if (e.state & 0x100 and self.orig_image is not None
                and not self._preprocessing_active
                and self.tool in ("brush", "eraser")):
            self.is_painting = True

    def _on_brush_change(self, val):
        self.brush_size = int(val)
        self.brush_label.configure(text=f"{self.brush_size}px")

    def _clear_layer(self):
        m = self.masks[self.active_layer]
        s = self.sam2_masks[self.active_layer]
        if m is not None:
            self._undo[self.active_layer] = (m.copy(), s.copy() if s is not None else None)
            m[:] = 0
            self.sam2_masks[self.active_layer] = None
            self._scribble_pts = []
            self.canvas.delete("scribble")
            self._rebuild_layer_buttons()
            self._render_canvas()

    def _do_undo(self, _event=None):
        snap = self._undo[self.active_layer]
        if snap is not None:
            painted_snap, sam2_snap = snap
            if self.masks[self.active_layer] is not None:
                self.masks[self.active_layer][:] = painted_snap
            self.sam2_masks[self.active_layer] = sam2_snap
            self._undo[self.active_layer] = None
            # Clear any in-progress scribble
            self._scribble_pts = []
            self.canvas.delete("scribble")
            self.canvas.delete("lasso")
            self._rebuild_layer_buttons()
            self._render_canvas()
            self._update_status("Desfeito (Ctrl+Z)")

    # ── Output dir ─────────────────────────────────────────────────────────────

    def _choose_output_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir = d
            self._out_label.configure(text=self._short_path(d))
            self._save_settings()

    def _get_output_dir(self):
        if self.output_dir and os.path.isdir(self.output_dir):
            return self.output_dir
        if self.image_path:
            return os.path.dirname(self.image_path)
        return "."

    @staticmethod
    def _short_path(p):
        if not p:
            return "Mesma pasta da imagem"
        parts = p.replace("\\", "/").split("/")
        return ".../" + "/".join(parts[-2:]) if len(parts) > 2 else p

    # ── SAM2 model loading ─────────────────────────────────────────────────────

    def _load_predictor(self):
        with self._sam2_lock:
            if self._sam2_predictor is not None:
                return self._sam2_predictor
            if not os.path.exists(SAM2_CKPT):
                return None
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model  = None
                for cfg in SAM2_CONFIGS:
                    try:
                        model = build_sam2(cfg, SAM2_CKPT, device=device)
                        break
                    except Exception:
                        continue
                if model is None:
                    raise RuntimeError("Nenhum config SAM2 funcionou")
                self._sam2_model     = model
                self._sam2_predictor = SAM2ImagePredictor(model)
                return self._sam2_predictor
            except Exception as e:
                _log('SAM2_LOAD', type(e), e, e.__traceback__)
                self._update_status(f"SAM2 falhou: {e}")
                return None

    def _load_auto_gen(self):
        if self._sam2_auto_gen is not None:
            return self._sam2_auto_gen
        if self._load_predictor() is None:
            return None
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            self._sam2_auto_gen = SAM2AutomaticMaskGenerator(
                self._sam2_model,
                points_per_side=32,        # mais pontos = mais objetos encontrados
                pred_iou_thresh=0.72,      # mais permissivo = pega mais elementos
                stability_score_thresh=0.80,
                min_mask_region_area=350,  # objetos/animais menores
                box_nms_thresh=0.70,
                crop_n_layers=1,
                crop_overlap_ratio=0.35,
            )
            return self._sam2_auto_gen
        except Exception as e:
            self._update_status(f"AutoMask: {e}")
            return None

    # ── ZoeDepth ───────────────────────────────────────────────────────────────

    def _load_zoedepth(self):
        with self._depth_lock:
            if self._depth_model is not None:
                return self._depth_model
            try:
                from zoedepth.models.builder import build_model
                from zoedepth.utils.config import get_config
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                conf   = get_config("zoedepth", "infer")
                model  = build_model(conf).to(device).eval()
                self._depth_model = model
                return model
            except Exception:
                return None

    # ── Pre-processing (image load → SAM2 auto + depth) ────────────────────────

    def _preprocess_image(self):
        if not self._preprocess_lock.acquire(blocking=False):
            return
        try:
            self._preprocess_inner()
        finally:
            self._preprocess_lock.release()

    def _preprocess_inner(self):
        if self.orig_image is None:
            return
        src = self.orig_image   # snapshot — guards against image change mid-run

        # 1. Ensure checkpoint
        def _ckpt_cb(frac, msg):
            self._overlay_show("", "Baixando SAM2...", msg, frac * 0.15)

        self._overlay_show("", "Verificando SAM2...", "Checando checkpoint...", 0.01)
        if not _ensure_checkpoint(_ckpt_cb):
            self._overlay_show("", "SAM2 indisponivel",
                               "Checkpoint nao encontrado.\nBaixe sam2_hiera_small.pt manualmente.", 0.0)
            time.sleep(2.5)
            self._overlay_hide()
            return

        sam2_result  = []
        depth_result = [None]

        # 2a. SAM2 auto-segmentation — sempre a 512px para ser rápido
        def run_sam2():
            self._overlay_show("", "Identificando objetos...",
                               "SAM2: pessoas, animais, elementos...", 0.20)
            gen = self._load_auto_gen()
            if gen is None or src is not self.orig_image:
                return
            try:
                img_rgb = np.array(src.convert("RGB"))
                H, W    = img_rgb.shape[:2]
                scale   = min(1.0, 512 / max(W, H))   # fixo 512px para velocidade
                small   = (cv2.resize(img_rgb, (int(W * scale), int(H * scale)))
                           if scale < 1.0 else img_rgb)
                self._overlay_show("", "SAM2 analisando...",
                                   f"Segmentando {small.shape[1]}x{small.shape[0]}px...", 0.28)
                with torch.inference_mode():
                    raw = gen.generate(small)
                full = []
                for m in raw:
                    seg  = m['segmentation'].astype(np.uint8) * 255
                    if scale < 1.0:
                        seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
                    area  = int(np.sum(seg > 0))
                    ys, xs = np.nonzero(seg > 0)
                    cx = int(xs.mean()) if len(xs) > 0 else W // 2
                    cy = int(ys.mean()) if len(ys) > 0 else H // 2
                    full.append({'segmentation': seg, 'area': area, 'cx': cx, 'cy': cy})
                self._overlay_show("", "Unindo segmentos...",
                                   "Combinando partes em objetos completos...", 0.42)
                full = self._merge_masks(full)
                sam2_result.extend(full)
                self._overlay_show("", "Objetos identificados",
                                   f"{len(full)} elementos encontrados", 0.55)
            except Exception as e:
                _log('SAM2_AUTO', type(e), e, e.__traceback__)

        # 2b. Depth map: tenta ZoeDepth, fallback para luminancia invertida
        def run_depth():
            self._overlay_show("", "Calculando profundidade...",
                               "Analisando distancias na cena...", 0.60)
            if src is not self.orig_image:
                return
            # Tenta ZoeDepth
            try:
                model = self._load_zoedepth()
                if model is not None:
                    pil = src.convert("RGB")
                    with torch.inference_mode():
                        depth = model.infer_pil(pil)
                    if isinstance(depth, torch.Tensor):
                        depth = depth.squeeze().cpu().numpy()
                    mn, mx = float(depth.min()), float(depth.max())
                    depth_result[0] = ((depth - mn) / max(mx - mn, 1e-6)).astype(np.float32)
                    self._overlay_show("", "Profundidade calculada",
                                       "ZoeDepth: mapa metrico pronto", 0.88)
                    return
            except Exception as e:
                _log('DEPTH_ZOE', type(e), e, e.__traceback__)
            # Fallback: luminancia invertida (claro=longe, escuro=perto)
            try:
                rgb  = np.array(src.convert("RGB"))
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                smooth = cv2.GaussianBlur(gray, (0, 0),
                                          sigmaX=max(1, gray.shape[1] // 30))
                norm = smooth / 255.0
                depth_result[0] = (1.0 - norm).astype(np.float32)
                self._overlay_show("", "Depth map (aproximado)",
                                   "ZoeDepth indisponivel — usando estimativa rapida", 0.88)
            except Exception as e:
                _log('DEPTH_FALLBACK', type(e), e, e.__traceback__)

        # Roda SAM2 primeiro, depois depth (evita conflito de memoria GPU)
        t1 = threading.Thread(target=run_sam2,  daemon=True)
        t1.start(); t1.join()
        t2 = threading.Thread(target=run_depth, daemon=True)
        t2.start(); t2.join()

        if src is self.orig_image:
            self._auto_masks = sam2_result         # [] = ran, found nothing
            if depth_result[0] is not None:
                self._depth_map = depth_result[0]

        parts = []
        if self._auto_masks:
            parts.append(f"{len(self._auto_masks)} objetos")
        if self._depth_map is not None:
            parts.append("depth map")

        if parts:
            self._overlay_show("", "Analise completa!",
                               " | ".join(parts) + "\nIA pronta para uso", 1.0)
            time.sleep(1.0)
        else:
            self._overlay_show("", "Analise parcial",
                               "SAM2 ou ZoeDepth nao encontrados", 1.0)
            time.sleep(1.5)

        self._overlay_hide()
        self._update_status(" | ".join(parts) if parts else "Modelos nao encontrados")

    # ── Mask merging ───────────────────────────────────────────────────────────

    def _merge_masks(self, masks):
        """Absorb masks >50% inside a larger mask → whole-object segments."""
        if len(masks) < 2:
            return masks
        masks    = sorted(masks, key=lambda m: m['area'], reverse=True)
        absorbed = [False] * len(masks)
        for i in range(len(masks)):
            if absorbed[i]:
                continue
            par = masks[i]['segmentation']
            for j in range(i + 1, len(masks)):
                if absorbed[j]:
                    continue
                ca = masks[j]['area']
                if ca == 0:
                    continue
                child = masks[j]['segmentation']
                if int(np.sum((par > 0) & (child > 0))) / ca > 0.50:
                    masks[i]['segmentation'] = np.maximum(par, child)
                    masks[i]['area']         = int(np.sum(masks[i]['segmentation'] > 0))
                    par                      = masks[i]['segmentation']
                    absorbed[j]              = True
        result = []
        for i, m in enumerate(masks):
            if not absorbed[i]:
                ys, xs   = np.nonzero(m['segmentation'] > 0)
                m['cx']  = int(xs.mean()) if len(xs) > 0 else m['cx']
                m['cy']  = int(ys.mean()) if len(ys) > 0 else m['cy']
                result.append(m)
        return result

    # ── Depth helpers ──────────────────────────────────────────────────────────

    def _depth_to_layer(self, img_x, img_y):
        if self._depth_map is None:
            return self.active_layer
        H, W = self._depth_map.shape
        dv   = float(self._depth_map[min(H-1, max(0, img_y)),
                                     min(W-1, max(0, img_x))])
        return min(max(int((1.0 - dv) * self.num_layers), 0), self.num_layers - 1)

    def _mask_mean_depth(self, seg):
        if self._depth_map is None:
            return 0.0
        H, W = self._depth_map.shape
        h, w = seg.shape
        if (h, w) != (H, W):
            seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
        vals = self._depth_map[seg > 127]
        return float(vals.mean()) if len(vals) > 0 else 0.0

    # ── Real-time SAM2 refinement (debounced, runs after brush stroke) ──────────

    def _schedule_sam2_refine(self, layer):
        if self._sam2_refine_job is not None:
            self.after_cancel(self._sam2_refine_job)
        self._sam2_refine_job = self.after(
            350, lambda: self._kick_sam2_refine(layer))

    def _kick_sam2_refine(self, layer):
        self._sam2_refine_job = None
        if self._sam2_running:
            return
        m = self.masks[layer]
        if m is None or m.max() == 0:
            return
        self._sam2_running = True
        self._update_status("Refinando com SAM2...")
        threading.Thread(target=self._run_sam2_refine, args=(layer,),
                         daemon=True).start()

    def _run_sam2_refine(self, layer):
        try:
            mask = self.masks[layer]
            if mask is None or mask.max() == 0:
                return
            predictor = self._load_predictor()
            if predictor is None or self.orig_image is None:
                return

            W, H  = self.orig_image.size
            ps    = int(self.proc_size_var.get())
            scale = min(1.0, ps / max(W, H))
            img   = np.array(self.orig_image.convert("RGB"))
            if scale < 1.0:
                img_s  = cv2.resize(img,  (int(W*scale), int(H*scale)))
                mask_s = cv2.resize(mask, (int(W*scale), int(H*scale)),
                                    interpolation=cv2.INTER_NEAREST)
            else:
                img_s, mask_s = img, mask

            points, labels = _sample_points(mask_s, n=10)
            if points is None:
                return

            with self._predictor_lock:
                predictor.set_image(img_s)
                with torch.inference_mode():
                    masks_out, scores, _ = predictor.predict(
                        point_coords=points, point_labels=labels,
                        multimask_output=True)

            best_i  = int(np.argmax(scores))
            refined = (masks_out[best_i] > 0.5).astype(np.uint8) * 255
            if scale < 1.0:
                refined = cv2.resize(refined, (W, H), interpolation=cv2.INTER_NEAREST)
            refined = _morpho_clean(refined)

            # Only commit if image hasn't changed and layer still has paint
            if self.orig_image and self.masks[layer] is not None and self.masks[layer].max() > 0:
                self.sam2_masks[layer] = refined
                area = int(np.sum(refined > 0))

                def _update():
                    self._rebuild_layer_buttons()
                    self._render_canvas()
                    self._update_status(
                        f"SAM2 pronto\nLayer {layer+1} | {area:,}px | "
                        f"score {scores[best_i]:.2f}")
                self.after(0, _update)

        except Exception as e:
            _log('SAM2_REFINE', type(e), e, e.__traceback__)
            self.after(0, lambda: self._update_status(f"SAM2 erro: {e}"))
        finally:
            self._sam2_running = False

    # ── Lasso select (rough circle → SAM2 precise mask) ───────────────────────

    def _clear_lasso_canvas(self):
        self.canvas.delete("lasso")
        self._lasso_pts = []

    @staticmethod
    def _bg_neg_points(W, H, x0, y0, x1, y1):
        """Background negative prompt points placed outside subject bounding box."""
        pts = []
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        # Sides outside the box
        if x0 > W * 0.08:   pts.append([max(0, x0 // 2),              cy])
        if x1 < W * 0.92:   pts.append([min(W-1, x1 + (W-x1)//2),    cy])
        if y0 > H * 0.08:   pts.append([cx, max(0, y0 // 2)])
        if y1 < H * 0.92:   pts.append([cx, min(H-1, y1 + (H-y1)//2)])
        # Corners far from box
        for px, py in [(4, 4), (W-5, 4), (4, H-5), (W-5, H-5)]:
            if abs(px - cx) > W * 0.25 or abs(py - cy) > H * 0.25:
                pts.append([px, py])
        return np.array(pts[:6], dtype=np.float32) if pts else None

    def _run_lasso_sam2(self, canvas_pts, layer):
        try:
            if self.orig_image is None:
                return
            predictor = self._load_predictor()
            if predictor is None:
                self._update_status("SAM2 indisponivel")
                self.after(0, self._clear_lasso_canvas)
                return

            W, H  = self.orig_image.size
            ps    = int(self.proc_size_var.get())
            scale = min(1.0, ps / max(W, H))
            img   = np.array(self.orig_image.convert("RGB"))

            # Convert canvas coords → image coords
            img_pts = np.array([(int(cx / self.zoom), int(cy / self.zoom))
                                 for cx, cy in canvas_pts], dtype=np.int32)
            img_pts[:, 0] = np.clip(img_pts[:, 0], 0, W - 1)
            img_pts[:, 1] = np.clip(img_pts[:, 1], 0, H - 1)

            x0 = int(img_pts[:, 0].min())
            y0 = int(img_pts[:, 1].min())
            x1 = int(img_pts[:, 0].max())
            y1 = int(img_pts[:, 1].max())

            if x1 - x0 < 4 or y1 - y0 < 4:
                self.after(0, self._clear_lasso_canvas)
                self._update_status("Contorno muito pequeno")
                return

            # Positive: interior polygon points
            lasso_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(lasso_mask, [img_pts], 255)
            pos_pts, pos_lbl = _sample_points(lasso_mask, n=10)
            if pos_pts is None:
                pos_pts = np.array([[(x0+x1)//2, (y0+y1)//2]], dtype=np.float32)
                pos_lbl = np.array([1], dtype=np.int32)

            # Negative: background points outside the contour
            neg_pts = self._bg_neg_points(W, H, x0, y0, x1, y1)
            if neg_pts is not None:
                all_pts = np.vstack([pos_pts, neg_pts])
                all_lbl = np.concatenate([pos_lbl,
                                          np.zeros(len(neg_pts), dtype=np.int32)])
            else:
                all_pts, all_lbl = pos_pts, pos_lbl

            # Box with 15% padding to capture whole object
            pad_x = max(16, int((x1 - x0) * 0.15))
            pad_y = max(16, int((y1 - y0) * 0.15))
            box = np.array([max(0, x0-pad_x), max(0, y0-pad_y),
                            min(W-1, x1+pad_x), min(H-1, y1+pad_y)],
                           dtype=np.float32)

            if scale < 1.0:
                img_s    = cv2.resize(img, (int(W * scale), int(H * scale)))
                all_pts  = all_pts * scale
                box      = box * scale
            else:
                img_s = img

            with self._predictor_lock:
                predictor.set_image(img_s)
                with torch.inference_mode():
                    masks_out, scores, _ = predictor.predict(
                        point_coords=all_pts,
                        point_labels=all_lbl,
                        box=box,
                        multimask_output=True)

            best_i = int(np.argmax(scores))
            result = (masks_out[best_i] > 0.5).astype(np.uint8) * 255
            if scale < 1.0:
                result = cv2.resize(result, (W, H), interpolation=cv2.INTER_NEAREST)
            result = _morpho_clean(result)

            if self.orig_image and result.max() > 0:
                m = self.masks[layer]
                self.masks[layer]      = np.maximum(m if m is not None else result, result)
                self.sam2_masks[layer] = result
                area = int(np.sum(result > 0))

                def _upd():
                    self._clear_lasso_canvas()
                    self._rebuild_layer_buttons()
                    self._render_canvas()
                    self._update_status(
                        f"Lasso SAM2 pronto\nLayer {layer+1} | {area:,}px"
                        f" | score {scores[best_i]:.2f}")
                self.after(0, _upd)
            else:
                self.after(0, self._clear_lasso_canvas)
                self._update_status("Nenhum objeto encontrado no contorno")

        except Exception as ex:
            _log('LASSO_SAM2', type(ex), ex, ex.__traceback__)
            self.after(0, self._clear_lasso_canvas)
            self.after(0, lambda: self._update_status(f"Lasso erro: {ex}"))

    # ── Scribble / Object Paint (Google-Photos-style brush → SAM2) ────────────

    def _clear_scribble_canvas(self):
        self.canvas.delete("scribble")
        self._scribble_pts = []

    def _schedule_scribble_sam2(self, layer):
        """Debounce: wait 500 ms after last stroke before running SAM2."""
        if self._scribble_job is not None:
            self.after_cancel(self._scribble_job)
        self._scribble_job = self.after(
            500, lambda: self._kick_scribble_sam2(layer))

    def _kick_scribble_sam2(self, layer):
        self._scribble_job = None
        if self._sam2_running:
            # SAM2 busy — retry in 300 ms
            self._scribble_job = self.after(
                300, lambda: self._kick_scribble_sam2(layer))
            return
        pts = list(self._scribble_pts)
        if not pts:
            return
        self._sam2_running = True
        self._update_status("Analisando objeto — SAM2...")
        threading.Thread(target=self._run_scribble_sam2,
                         args=(pts, layer), daemon=True).start()

    def _run_scribble_sam2(self, img_pts, layer):
        """Full-quality SAM2 from scribble strokes.
        img_pts: list of (img_x, img_y) collected during all brush strokes.
        """
        try:
            if self.orig_image is None:
                return
            predictor = self._load_predictor()
            if predictor is None:
                self._update_status("SAM2 indisponivel")
                return

            W, H  = self.orig_image.size
            ps    = int(self.proc_size_var.get())
            scale = min(1.0, ps / max(W, H))
            img   = np.array(self.orig_image.convert("RGB"))

            # ── Build prompt from stroke points ────────────────────────────
            arr = np.array(img_pts, dtype=np.int32)
            arr[:, 0] = np.clip(arr[:, 0], 0, W - 1)
            arr[:, 1] = np.clip(arr[:, 1], 0, H - 1)

            # Evenly subsample up to 24 points along the stroke path
            n_want = min(24, len(arr))
            idx    = np.round(np.linspace(0, len(arr) - 1, n_want)).astype(int)
            pts_sel = arr[idx].astype(np.float32)   # (N, 2) x,y
            labels  = np.ones(len(pts_sel), dtype=np.int32)

            # Bounding box with 15% padding to capture whole object
            x0r, y0r = int(arr[:, 0].min()), int(arr[:, 1].min())
            x1r, y1r = int(arr[:, 0].max()), int(arr[:, 1].max())
            pad_x = max(16, int((x1r - x0r) * 0.15))
            pad_y = max(16, int((y1r - y0r) * 0.15))
            bx0 = max(0,     x0r - pad_x);  by0 = max(0,     y0r - pad_y)
            bx1 = min(W - 1, x1r + pad_x);  by1 = min(H - 1, y1r + pad_y)
            box = np.array([bx0, by0, bx1, by1], dtype=np.float32)

            # Negative background points outside the painted region
            neg_pts = self._bg_neg_points(W, H, bx0, by0, bx1, by1)
            if neg_pts is not None:
                all_pts = np.vstack([pts_sel, neg_pts])
                all_lbl = np.concatenate([labels,
                                          np.zeros(len(neg_pts), dtype=np.int32)])
            else:
                all_pts, all_lbl = pts_sel, labels

            # ── Scale to processing resolution ────────────────────────────
            if scale < 1.0:
                img_s    = cv2.resize(img, (int(W * scale), int(H * scale)))
                pts_s    = all_pts * scale
                box_s    = box * scale
            else:
                img_s, pts_s, box_s = img, all_pts, box
            all_lbl_s = all_lbl  # labels don't scale

            # ── SAM2 inference ────────────────────────────────────────────
            with self._predictor_lock:
                predictor.set_image(img_s)
                with torch.inference_mode():
                    masks_out, scores, _ = predictor.predict(
                        point_coords=pts_s,
                        point_labels=all_lbl_s,
                        box=box_s,
                        multimask_output=True)

            best_i = int(np.argmax(scores))
            result = (masks_out[best_i] > 0.5).astype(np.uint8) * 255
            if scale < 1.0:
                result = cv2.resize(result, (W, H), interpolation=cv2.INTER_NEAREST)
            result = _morpho_clean(result)

            if self.orig_image and result.max() > 0:
                m = self.masks[layer]
                self.masks[layer]      = np.maximum(
                    m if m is not None else result, result)
                self.sam2_masks[layer] = result
                area = int(np.sum(result > 0))

                def _upd():
                    # Keep scribble strokes visible so user knows what they painted
                    self._rebuild_layer_buttons()
                    self._render_canvas()
                    self._update_status(
                        f"Objeto selecionado\nLayer {layer+1} | {area:,}px"
                        f" | score {scores[best_i]:.2f}\n"
                        f"Pinte mais para refinar ou use Limpar para recomecar")
                self.after(0, _upd)
            else:
                self._update_status(
                    "Nenhum objeto identificado.\n"
                    "Pinte mais sobre o objeto e aguarde.")

        except Exception as ex:
            _log('SCRIBBLE_SAM2', type(ex), ex, ex.__traceback__)
            self.after(0, lambda: self._update_status(f"Erro: {ex}"))
        finally:
            self._sam2_running = False

    # ── Smart selection (SAM2 auto → depth-sorted layers) ──────────────────────

    def _smart_select(self):
        if self.orig_image is None:
            self._update_status("Carregue uma imagem primeiro")
            return
        if self._preprocessing_active:
            self._update_status("Analise em andamento, aguarde...")
            return
        if self._auto_masks:
            threading.Thread(target=self._apply_auto_to_layers, daemon=True).start()
        else:
            threading.Thread(target=self._preprocess_then_apply, daemon=True).start()

    def _preprocess_then_apply(self):
        self._preprocess_image()
        if self._auto_masks:
            self._apply_auto_to_layers()

    def _apply_auto_to_layers(self):
        """
        Distribui objetos detectados em camadas por profundidade:
          Layer 0 = frente (mais perto)
          Layer 1 = meio
          Layer N-1 = fundo (mais longe)
        Se não houver depth map, usa area (maior = fundo).
        """
        if not self._auto_masks or self.orig_image is None:
            return
        W, H     = self.orig_image.size
        total_px = W * H
        n        = self.num_layers

        valid = [m for m in self._auto_masks
                 if 0.001 * total_px < m['area'] < 0.88 * total_px]

        # Reset todas as camadas
        for i in range(n):
            if self.masks[i] is not None:
                self.masks[i][:] = 0
            self.sam2_masks[i] = None

        if not valid:
            self.after(0, lambda: self._update_status("Nenhum objeto encontrado"))
            return

        if self._depth_map is not None:
            # Atribui profundidade média a cada máscara
            for m in valid:
                m['_d'] = self._mask_mean_depth(m['segmentation'])

            depths = [m['_d'] for m in valid]
            d_min, d_max = min(depths), max(depths)
            d_range = max(d_max - d_min, 1e-4)

            # Agrupa por faixa de profundidade em N buckets
            # d=1.0 (perto) → layer 0 (frente)
            # d=0.0 (longe) → layer n-1 (fundo)
            buckets = [np.zeros((H, W), dtype=np.uint8) for _ in range(n)]
            for m in valid:
                rel = (m['_d'] - d_min) / d_range  # 0=mais longe, 1=mais perto
                idx = min(n - 1, int((1.0 - rel) * n))  # 0=frente, n-1=fundo
                seg = m['segmentation']
                if seg.shape != (H, W):
                    seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
                buckets[idx] = np.maximum(buckets[idx], seg)

            for i in range(n):
                self.masks[i]      = buckets[i]
                self.sam2_masks[i] = buckets[i] if buckets[i].max() > 0 else None

            filled = sum(1 for b in buckets if b.max() > 0)
            method = f"depth map ({filled}/{n} camadas)"
        else:
            # Sem depth map: ordena por área, maior = fundo
            valid.sort(key=lambda m: m['area'], reverse=True)
            for i in range(min(len(valid), n)):
                seg = valid[i]['segmentation']
                if seg.shape != (H, W):
                    seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
                idx = n - 1 - i  # maior área vai para o fundo
                idx = max(0, min(n - 1, idx))
                self.masks[i]      = seg
                self.sam2_masks[i] = seg
            method = "area (sem depth map)"

        def _upd():
            self._rebuild_layer_buttons()
            self._render_canvas()
            self._update_status(
                f"Camadas distribuidas\npor {method}\n"
                f"Layer 1=frente  Layer {n}=fundo")
        self.after(0, _upd)

    # ── Depth map toggle ───────────────────────────────────────────────────────

    def _toggle_depth(self):
        if self._depth_map is None:
            self._update_status("Depth map indisponivel.\nAguarde a analise.")
            return
        self._show_depth = not self._show_depth
        self._depth_btn.configure(
            text="Ocultar Depth Map" if self._show_depth else "Ver Depth Map",
            fg_color="#1e3a5f" if self._show_depth else "transparent")
        self._base_cache = {}
        self._render_canvas()

    # ── Export helpers ─────────────────────────────────────────────────────────

    def _get_feather_radius(self):
        return {"Fina": 3, "Media": 6, "Suave": 14}.get(self.feather_var.get(), 6)

    def _best_mask_for_export(self, layer):
        """SAM2 mask if ready, else morpho-cleaned painted mask."""
        s = self.sam2_masks[layer]
        if s is not None and s.max() > 0:
            return s
        m = self.masks[layer]
        if m is None or m.max() == 0:
            return None
        return _morpho_clean(m)

    def _make_cutout(self, mask):
        """Apply feathered mask as alpha → RGBA PIL."""
        feathered = _feather_mask(mask, self._get_feather_radius())
        arr = np.array(self.orig_image.convert("RGBA"))
        arr[:, :, 3] = feathered
        return Image.fromarray(arr, "RGBA")

    # ── Inpainting ─────────────────────────────────────────────────────────────

    def _inpaint_region(self, working_pil, bin_mask):
        mode = self._inpaint_mode.get()
        if bin_mask.ndim > 2:
            bin_mask = bin_mask[:, :, 0]
        bin_mask = (bin_mask > 127).astype(np.uint8) * 255

        if mode == "SDXL":
            try:
                if self._sdxl_pipe is None:
                    from diffusers import StableDiffusionXLInpaintPipeline
                    self._update_status("Carregando SDXL...")
                    self._sdxl_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                        torch_dtype=torch.float16, variant="fp16")
                    self._sdxl_pipe.enable_model_cpu_offload()
                W, H = working_pil.size
                s    = min(1.0, 1024 / max(W, H))
                nW, nH = max(1, int(W*s)), max(1, int(H*s))
                prompt = self._inpaint_prompt(working_pil, bin_mask)
                out = self._sdxl_pipe(
                    prompt=prompt,
                    negative_prompt="blurry,artifacts,watermark,seam,low quality",
                    image=working_pil.resize((nW, nH), Image.LANCZOS),
                    mask_image=Image.fromarray(bin_mask).resize((nW, nH), Image.NEAREST),
                    num_inference_steps=30, strength=0.99, guidance_scale=8.0
                ).images[0]
                return out.resize((W, H), Image.LANCZOS).convert("RGB")
            except Exception as e:
                self._update_status(f"SDXL erro, OpenCV: {e}")

        img_bgr   = cv2.cvtColor(np.array(working_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        hole_area = int(np.sum(bin_mask > 0))
        if hole_area > 40000:
            H0, W0   = img_bgr.shape[:2]
            img_h    = cv2.resize(img_bgr, (W0//2, H0//2))
            mask_h   = (cv2.resize(bin_mask, (W0//2, H0//2),
                                   interpolation=cv2.INTER_NEAREST) > 127
                        ).astype(np.uint8) * 255
            coarse   = cv2.inpaint(img_h, mask_h, 25, cv2.INPAINT_NS)
            pre      = img_bgr.copy()
            pre[bin_mask > 127] = cv2.resize(coarse, (W0, H0),
                                             interpolation=cv2.INTER_LINEAR)[bin_mask > 127]
            result   = cv2.inpaint(pre, bin_mask, 10, cv2.INPAINT_NS)
        else:
            r      = min(40, max(8, int(np.sqrt(hole_area) * 0.04)))
            result = cv2.inpaint(img_bgr, bin_mask, r, cv2.INPAINT_NS)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    def _inpaint_prompt(self, img_pil, mask_np):
        try:
            border = cv2.subtract(
                cv2.dilate(mask_np, np.ones((21, 21), np.uint8), iterations=3), mask_np)
            arr    = np.array(img_pil.convert("RGB"))
            if arr.shape[:2] != border.shape:
                border = cv2.resize(border, (arr.shape[1], arr.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            pixels = arr[border > 0]
            if len(pixels) == 0:
                return "seamless photorealistic background continuation, high quality"
            r, g, b = pixels.mean(axis=0).astype(int)
            if b > r and b > g:     d = "sky blue"
            elif g > r and g > b:   d = "green vegetation"
            elif r > g and r > b:   d = "warm tones"
            elif min(r,g,b) > 150:  d = "bright background"
            elif max(r,g,b) < 80:   d = "dark background"
            else:                   d = "neutral"
            return (f"seamless photorealistic {d} background continuation, "
                    f"matching surrounding texture and lighting, high quality, no artifacts")
        except Exception:
            return "seamless photorealistic background continuation, high quality"

    # ── Results panel ──────────────────────────────────────────────────────────

    def _show_results(self, results):
        for w in self._results_scroll.winfo_children():
            w.destroy()
        self._thumb_refs.clear()
        for kind, path in results:
            try:
                img = Image.open(path)
                if kind == 'layer':
                    img   = img.convert("RGBA")
                    thumb = img.copy(); thumb.thumbnail((230, 170), Image.LANCZOS)
                    cb    = Image.fromarray(_checkerboard(thumb.width, thumb.height), "RGBA")
                    flat  = Image.alpha_composite(cb, thumb)
                else:
                    img   = img.convert("RGB")
                    thumb = img.copy(); thumb.thumbnail((230, 170), Image.LANCZOS)
                    flat  = thumb
                photo = ImageTk.PhotoImage(flat)
                self._thumb_refs.append(photo)
                card = ctk.CTkFrame(self._results_scroll,
                                    fg_color="#13131f", corner_radius=8)
                card.pack(fill="x", pady=5, padx=4)
                tag  = "Recorte SAM2" if kind == 'layer' else "Fundo inpainted"
                col  = "#5eead4" if kind == 'layer' else "#a78bfa"
                ctk.CTkLabel(card, text=tag,
                             font=ctk.CTkFont(size=9, weight="bold"),
                             text_color=col).pack(pady=(6, 0))
                tk.Label(card, image=photo, bg="#13131f").pack(pady=2)
                ctk.CTkLabel(card, text=os.path.basename(path),
                             font=ctk.CTkFont(size=9),
                             text_color="gray").pack(pady=(0, 2))
                p = path
                ctk.CTkButton(card, text="Abrir", height=26,
                              font=ctk.CTkFont(size=10),
                              command=lambda pp=p: os.startfile(pp)
                              ).pack(pady=(0, 8))
            except Exception:
                pass

    # ── Processing pipeline ────────────────────────────────────────────────────

    def _process(self):
        if self.orig_image is None:
            self._update_status("Carregue uma imagem primeiro!")
            return
        if self._preprocessing_active:
            self._update_status("Analise em andamento, aguarde...")
            return
        painted = [i for i in range(self.num_layers)
                   if ((self.sam2_masks[i] is not None and self.sam2_masks[i].max() > 0) or
                       (self.masks[i] is not None and self.masks[i].max() > 0))]
        if not painted:
            self._update_status("Pinte pelo menos uma layer.")
            return
        threading.Thread(target=self._run_pipeline, args=(painted,), daemon=True).start()

    def _run_pipeline(self, painted):
        try:
            out_dir = self._get_output_dir()
            os.makedirs(out_dir, exist_ok=True)
            stem    = os.path.splitext(os.path.basename(self.image_path or "img"))[0]
            total   = len(painted)
            results = []
            working = self.orig_image.convert("RGB")

            for idx, i in enumerate(painted):
                self._overlay_show(
                    "", f"Exportando Layer {i+1}",
                    f"Gerando recorte SAM2 ({idx+1}/{total})...",
                    idx / total)

                mask = self._best_mask_for_export(i)
                if mask is None:
                    continue

                _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                # Export RGBA cutout with feathered alpha
                cutout   = self._make_cutout(mask)
                out_path = os.path.join(out_dir, f"{stem}_layer_{i+1}.png")
                cutout.save(out_path, optimize=False)
                results.append(('layer', out_path))

                # Inpaint background for next layer iteration
                if self._inpaint_mode.get() != "Desligado" and idx < total - 1:
                    self._overlay_show(
                        "", "Inpainting...",
                        f"Reconstruindo fundo Layer {i+1} ({self._inpaint_mode.get()})",
                        (idx + 0.6) / total)
                    dilated = cv2.dilate(bin_mask, np.ones((11, 11), np.uint8), iterations=2)
                    working = self._inpaint_region(working, dilated)
                    bg_path = os.path.join(out_dir, f"{stem}_bg_{i+1}.png")
                    working.save(bg_path)
                    results.append(('background', bg_path))

            self._overlay_hide()
            self._update_status(
                f"{len(results)} arquivo(s) exportados!\n{self._short_path(out_dir)}")
            self.after(0, lambda: self._show_results(results))

        except Exception as ex:
            _log('PIPELINE', type(ex), ex, ex.__traceback__)
            self._overlay_hide()
            self._update_status(f"Erro: {ex}")

    # ── Misc ───────────────────────────────────────────────────────────────────

    def _update_status(self, text):
        self.after(0, lambda: self.status_label.configure(text=text))


if __name__ == "__main__":
    app = ParallaxStudio()
    app.mainloop()
