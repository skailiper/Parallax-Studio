import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import threading
import traceback
import os
import sys
import json
import time

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_DIR       = os.path.dirname(os.path.abspath(__file__))
SAM2_CONFIG   = 'configs/sam2.1/sam2.1_hiera_s'
SAM2_CKPT     = os.path.join(APP_DIR, 'sam2_hiera_small.pt')
SETTINGS_FILE = os.path.join(APP_DIR, 'settings.json')
CRASH_LOG     = os.path.join(APP_DIR, 'crash.log')

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
    _log('MAIN', t, v, tb)
    _orig_hook(t, v, tb)
sys.excepthook = _excepthook

def _thread_hook(args):
    _log(f'THREAD:{getattr(args.thread, "name", "?")}',
         args.exc_type, args.exc_value, args.exc_tb)
threading.excepthook = _thread_hook

# ── DnD imports ────────────────────────────────────────────────────────────────

try:
    import tkinterdnd2
    _HAS_TKDND = True
except ImportError:
    _HAS_TKDND = False

try:
    import windnd
    _HAS_WINDND = True
except ImportError:
    _HAS_WINDND = False

# ── Helpers ────────────────────────────────────────────────────────────────────

def _checkerboard(w, h, ts=10):
    arr = np.full((h, w, 4), 40, dtype=np.uint8)
    arr[:, :, 3] = 255
    rows = np.arange(h) // ts
    cols = np.arange(w) // ts
    light = (rows[:, None] + cols[None, :]) % 2 == 0
    arr[light] = [62, 62, 62, 255]
    return arr


def _feather_mask(mask, radius=5):
    """Smooth mask edges: interior stays fully opaque, boundary gets a soft gradient."""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    k_size = radius * 2 + 1
    k = np.ones((k_size, k_size), np.uint8)
    inner = cv2.erode(binary, k)
    edge  = cv2.subtract(binary, inner)
    blur_k = max(3, radius * 4 + 1) | 1   # must be odd
    blurred = cv2.GaussianBlur(edge.astype(np.float32), (blur_k, blur_k), radius * 0.5)
    result = np.maximum(inner.astype(np.float32), blurred)
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Main application ───────────────────────────────────────────────────────────

class ParallaxStudio(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Parallax Studio")
        self.geometry("1500x940")
        self.minsize(1100, 680)

        # Core state
        self.image_path   = None
        self.orig_image   = None
        self._base_cache  = {}
        self.tk_image     = None
        self.num_layers   = 3
        self.active_layer = 0
        self.brush_size   = 30
        self.tool         = "brush"
        self.zoom         = 1.0
        self.is_painting  = False
        self.last_x       = None
        self.last_y       = None
        self.masks        = [None] * 8
        self.output_dir   = None
        self._brush_oval  = None
        self._thumb_refs  = []

        # SAM2
        self._sam2_model     = None
        self._sam2_predictor = None
        self._sam2_auto_gen  = None
        self._auto_masks     = None
        self._sam2_lock      = threading.Lock()

        # Depth
        self._depth_model = None
        self._depth_map   = None   # float32 H×W, 1=near 0=far
        self._depth_lock  = threading.Lock()

        # SDXL
        self._sdxl_pipe = None
        self._sdxl_lock = threading.Lock()

        # Preprocessing gate — blocks canvas interaction while AI is running
        self._preprocessing_active = False

        self._load_settings()
        self._build_ui()
        self._setup_dnd()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_settings(self):
        try:
            with open(SETTINGS_FILE, encoding='utf-8') as f:
                d = json.load(f)
            self.output_dir = d.get('output_dir')
        except Exception:
            pass

    def _save_settings(self):
        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump({'output_dir': self.output_dir}, f)
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
        sb = ctk.CTkFrame(self, width=245, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)
        sb.grid_rowconfigure(99, weight=1)
        r = 0

        ctk.CTkLabel(sb, text="PARALLAX STUDIO",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=r, column=0, padx=16, pady=(18, 2), sticky="w"); r += 1
        ctk.CTkLabel(sb, text="Pinte. A IA recorta.",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=r, column=0, padx=16, pady=(0, 10), sticky="w"); r += 1

        ctk.CTkButton(sb, text="📁  Carregar Imagem",
                      command=self._browse_image).grid(
            row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        self._sep(sb, r); r += 1

        ctk.CTkLabel(sb, text="Camadas:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        self._layer_seg = ctk.CTkSegmentedButton(
            sb, values=["2", "3", "4", "5", "6", "7", "8"],
            command=self._on_layer_count_change)
        self._layer_seg.set("3")
        self._layer_seg.grid(row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        ctk.CTkLabel(sb, text="Layer ativa:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1
        self.layer_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self.layer_frame.grid(row=r, column=0, padx=12, pady=2, sticky="ew"); r += 1
        self.layer_buttons = []
        self._rebuild_layer_buttons()

        self._sep(sb, r); r += 1

        ctk.CTkLabel(sb, text="Ferramenta:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        self.tool_var = ctk.StringVar(value="brush")
        tf = ctk.CTkFrame(sb, fg_color="transparent")
        tf.grid(row=r, column=0, padx=16, pady=3, sticky="w"); r += 1
        for val, label in [("brush",  "✦ Pincel"),
                            ("eraser", "◻ Borracha"),
                            ("magic",  "🪄 Seleção por Objeto")]:
            ctk.CTkRadioButton(tf, text=label, variable=self.tool_var, value=val,
                               command=lambda v=val: setattr(self, 'tool', v)
                               ).pack(anchor="w", pady=1)

        ctk.CTkLabel(sb, text="Tamanho do pincel:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1
        self.brush_slider = ctk.CTkSlider(sb, from_=4, to=300,
                                          command=self._on_brush_change)
        self.brush_slider.set(30)
        self.brush_slider.grid(row=r, column=0, padx=12, pady=2, sticky="ew"); r += 1
        self.brush_label = ctk.CTkLabel(sb, text="30px", font=ctk.CTkFont(size=11))
        self.brush_label.grid(row=r, column=0, padx=16, sticky="w"); r += 1

        ctk.CTkLabel(sb, text="Resolução SAM2:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1
        self.proc_size_var = ctk.StringVar(value="1024")
        ctk.CTkSegmentedButton(sb, values=["512", "1024", "2048"],
                               variable=self.proc_size_var).grid(
            row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        ctk.CTkLabel(sb, text="Suavização de bordas:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1
        self.feather_var = ctk.StringVar(value="Média")
        ctk.CTkSegmentedButton(sb, values=["Fina", "Média", "Suave"],
                               variable=self.feather_var).grid(
            row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        self._sep(sb, r); r += 1
        ctk.CTkLabel(sb, text="Inpainting (Layer 2+):",
                     font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        self._inpaint_mode = ctk.StringVar(value="OpenCV")
        ctk.CTkSegmentedButton(sb, values=["SDXL", "OpenCV", "Desligado"],
                               variable=self._inpaint_mode).grid(
            row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        self._sep(sb, r); r += 1
        ctk.CTkLabel(sb, text="Zoom:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        zf = ctk.CTkFrame(sb, fg_color="transparent")
        zf.grid(row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1
        ctk.CTkButton(zf, text="−", width=34,
                      command=lambda: self._set_zoom(self.zoom - 0.1)).pack(side="left", padx=1)
        self.zoom_label = ctk.CTkLabel(zf, text="100%", width=46)
        self.zoom_label.pack(side="left", padx=2)
        ctk.CTkButton(zf, text="+", width=34,
                      command=lambda: self._set_zoom(self.zoom + 0.1)).pack(side="left", padx=1)
        ctk.CTkButton(zf, text="⊡  Fit", width=60,
                      command=self._zoom_to_fit).pack(side="left", padx=(4, 0))

        ctk.CTkButton(sb, text="🗑  Limpar Layer",
                      fg_color="transparent", border_width=1,
                      command=self._clear_active_layer).grid(
            row=r, column=0, padx=12, pady=(10, 3), sticky="ew"); r += 1

        self._sep(sb, r); r += 1
        ctk.CTkLabel(sb, text="Salvar em:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        od = ctk.CTkFrame(sb, fg_color="transparent")
        od.grid(row=r, column=0, padx=12, pady=2, sticky="ew"); r += 1
        od.grid_columnconfigure(0, weight=1)
        self._out_label = ctk.CTkLabel(
            od, text=self._short_path(self.output_dir or "Mesma pasta da imagem"),
            font=ctk.CTkFont(size=10), text_color="gray", wraplength=155, anchor="w")
        self._out_label.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(od, text="...", width=34,
                      command=self._choose_output_dir).grid(row=0, column=1, padx=(4, 0))

        self._sep(sb, r); r += 1
        ctk.CTkButton(sb, text="⚡  Processar com IA",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      fg_color="#5eead4", text_color="#000", hover_color="#2dd4bf",
                      command=self._process).grid(
            row=r, column=0, padx=12, pady=(10, 3), sticky="ew"); r += 1
        ctk.CTkButton(sb, text="🔬  Análise Automática",
                      font=ctk.CTkFont(size=12),
                      fg_color="#7c3aed", hover_color="#6d28d9",
                      command=self._auto_analyze).grid(
            row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        self.status_label = ctk.CTkLabel(
            sb, text="Carregue uma imagem\nou arraste aqui",
            font=ctk.CTkFont(size=11), text_color="gray", wraplength=225)
        self.status_label.grid(row=99, column=0, padx=12, pady=10, sticky="sw")

    def _sep(self, parent, row):
        ctk.CTkFrame(parent, height=1, fg_color="#222233").grid(
            row=row, column=0, padx=12, pady=4, sticky="ew")

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
            text="📁  Arraste uma imagem aqui\nou use o botão Carregar Imagem",
            fg="#3a3a5c", bg="#08080f", font=("Segoe UI", 16), justify="center")
        self._drop_hint.place(relx=0.5, rely=0.5, anchor="center")

        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_down)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Enter>",           self._on_canvas_enter)
        self.canvas.bind("<Motion>",          self._on_canvas_motion)
        self.canvas.bind("<Leave>",           self._on_canvas_leave)

        # ── Loading overlay — full-canvas blocker + centered card ───────────
        # _ov_bg: covers the entire canvas area, blocks all mouse events
        self._ov_bg = tk.Frame(self._canvas_frame, bg="#000000")
        self._ov_bg.bind("<ButtonPress-1>",   lambda e: "break")
        self._ov_bg.bind("<B1-Motion>",       lambda e: "break")
        self._ov_bg.bind("<ButtonRelease-1>", lambda e: "break")

        # _ov: the visible info card, centered on top of the blocker
        self._ov = ctk.CTkFrame(self._canvas_frame, corner_radius=18,
                                fg_color="#0b0b1e", border_width=1,
                                border_color="#2e2e55")
        self._ov_icon  = ctk.CTkLabel(self._ov, text="🔬",
                                      font=ctk.CTkFont(size=42))
        self._ov_icon.pack(pady=(26, 4))
        self._ov_title = ctk.CTkLabel(self._ov, text="Analisando…",
                                      font=ctk.CTkFont(size=16, weight="bold"),
                                      text_color="#5eead4")
        self._ov_title.pack(pady=(0, 6))
        self._ov_body  = ctk.CTkLabel(self._ov, text="",
                                      font=ctk.CTkFont(size=12),
                                      text_color="#aaa", wraplength=380, justify="center")
        self._ov_body.pack(pady=(0, 14), padx=30)
        self._ov_bar   = ctk.CTkProgressBar(self._ov, width=340)
        self._ov_bar.pack(pady=(0, 26), padx=30)
        self._ov_bar.set(0)

    def _build_results_panel(self):
        frame = ctk.CTkFrame(self, width=265, corner_radius=0, fg_color="#0b0b18")
        frame.grid(row=0, column=2, sticky="nsew")
        frame.grid_propagate(False)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="RESULTADOS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#444466").grid(
            row=0, column=0, padx=14, pady=(14, 6), sticky="w")

        self._results_scroll = ctk.CTkScrollableFrame(frame, fg_color="transparent")
        self._results_scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._results_scroll.grid_columnconfigure(0, weight=1)

    # ── Loading overlay ────────────────────────────────────────────────────────

    def _overlay_show(self, icon, title, body, progress):
        def _do():
            self._ov_icon.configure(text=icon)
            self._ov_title.configure(text=title)
            self._ov_body.configure(text=body)
            self._ov_bar.set(max(0.0, min(1.0, progress)))
            # Full-coverage semi-transparent blocker
            self._ov_bg.place(x=0, y=0, relwidth=1, relheight=1)
            self._ov_bg.lift()
            # Info card centered on top of blocker
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

    # ── Layer UI ───────────────────────────────────────────────────────────────

    def _rebuild_layer_buttons(self):
        for w in self.layer_frame.winfo_children():
            w.destroy()
        self.layer_buttons = []
        for i in range(self.num_layers):
            rv, gv, bv, name = LAYER_COLORS[i]
            hex_c = f"#{rv:02x}{gv:02x}{bv:02x}"
            btn = ctk.CTkButton(
                self.layer_frame, text=name,
                fg_color=hex_c if i == self.active_layer else "transparent",
                text_color="#000" if i == self.active_layer else "#fff",
                border_color=hex_c, border_width=2,
                command=lambda idx=i: self._set_active_layer(idx))
            btn.pack(fill="x", pady=1)
            self.layer_buttons.append(btn)

    def _set_active_layer(self, idx):
        self.active_layer = idx
        self._rebuild_layer_buttons()

    def _on_layer_count_change(self, val):
        self.num_layers = int(val)
        self._rebuild_layer_buttons()

    # ── Drag-and-drop ──────────────────────────────────────────────────────────

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
        """All heavy IO in background; only touches UI via self.after()."""
        path = path.strip().strip('"').strip("'")
        if not os.path.isfile(path):
            return
        self.after(0, lambda: self._update_status("⏳ Carregando imagem…"))
        try:
            image = Image.open(path).convert("RGBA")
            W, H  = image.size
            masks = [np.zeros((H, W), dtype=np.uint8) for _ in range(8)]
        except Exception as e:
            self.after(0, lambda err=e: self._update_status(f"❌ Erro ao abrir:\n{err}"))
            return

        def _apply():
            self.image_path  = path
            self.orig_image  = image
            self.masks       = masks
            self._base_cache = {}
            self._auto_masks = None
            self._depth_map  = None
            self._drop_hint.place_forget()
            self._update_status(f"✅ {os.path.basename(path)}\n{W}×{H}px")
            self._zoom_to_fit()
            threading.Thread(target=self._preprocess_image, daemon=True).start()

        self.after(0, _apply)

    # ── Zoom / fit ─────────────────────────────────────────────────────────────

    def _zoom_to_fit(self):
        if not self.orig_image:
            return
        self.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        W, H = self.orig_image.size
        z = min(cw / W, ch / H, 1.0)
        self._set_zoom(round(z, 2))

    def _set_zoom(self, z):
        self.zoom = max(0.05, min(10.0, round(z, 2)))
        self.zoom_label.configure(text=f"{int(self.zoom * 100)}%")
        if self.orig_image:
            self._render_canvas()

    # ── Canvas rendering ───────────────────────────────────────────────────────

    def _render_canvas(self, fast=False):
        if not self.orig_image:
            return
        W, H = self.orig_image.size
        dw = max(1, int(W * self.zoom))
        dh = max(1, int(H * self.zoom))

        z_key = self.zoom
        if z_key not in self._base_cache:
            interp   = Image.BILINEAR if fast else Image.LANCZOS
            base_pil = self.orig_image.resize((dw, dh), interp)
            if not fast:
                self._base_cache = {z_key: base_pil}
        else:
            base_pil = self._base_cache[z_key]

        arr = np.array(base_pil.convert("RGB"), dtype=np.float32)
        for i in range(self.num_layers):
            if self.masks[i] is None or self.masks[i].max() == 0:
                continue
            mr = cv2.resize(self.masks[i], (dw, dh),
                            interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
            rv, gv, bv, _ = LAYER_COLORS[i]
            alpha = mr * 0.55
            arr[:, :, 0] = arr[:, :, 0] * (1 - alpha) + rv * alpha
            arr[:, :, 1] = arr[:, :, 1] * (1 - alpha) + gv * alpha
            arr[:, :, 2] = arr[:, :, 2] * (1 - alpha) + bv * alpha

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(arr.astype(np.uint8)))
        self.canvas.delete("img")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image, tags="img")
        self.canvas.configure(scrollregion=(0, 0, dw, dh))
        self.canvas.tag_raise("cursor")

    # ── Brush cursor ───────────────────────────────────────────────────────────

    def _on_canvas_motion(self, e):
        cx = self.canvas.canvasx(e.x)
        cy = self.canvas.canvasy(e.y)
        if self.tool == "magic":
            r, outline, dash = 14, "#a855f7", ()
        else:
            r, outline, dash = self.brush_size, "white", (3, 3)
        if self._brush_oval:
            self.canvas.coords(self._brush_oval, cx - r, cy - r, cx + r, cy + r)
            self.canvas.itemconfigure(self._brush_oval, outline=outline, dash=dash)
        else:
            self._brush_oval = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=outline, width=1, dash=dash, tags="cursor")

    def _on_canvas_leave(self, e):
        if self._brush_oval:
            self.canvas.delete(self._brush_oval)
            self._brush_oval = None

    # ── Painting ───────────────────────────────────────────────────────────────

    def _canvas_to_orig(self, ex, ey):
        x = int(self.canvas.canvasx(ex) / self.zoom)
        y = int(self.canvas.canvasy(ey) / self.zoom)
        if self.orig_image:
            W, H = self.orig_image.size
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
        return x, y

    def _paint_at(self, x, y):
        if self.orig_image is None:
            return
        mask = self.masks[self.active_layer]
        r = max(1, int(self.brush_size / self.zoom))
        if self.tool == "brush":
            cv2.circle(mask, (x, y), r, 255, -1)
        elif self.tool == "eraser":
            cv2.circle(mask, (x, y), int(r * 1.6), 0, -1)

    def _on_mouse_down(self, e):
        if self.orig_image is None or self._preprocessing_active:
            return
        x, y = self._canvas_to_orig(e.x, e.y)
        if self.tool == "magic":
            self._update_status("🪄 Identificando objeto…")
            threading.Thread(target=self._magic_select, args=(x, y),
                             daemon=True).start()
            return
        self.is_painting = True
        self.last_x, self.last_y = x, y
        self._paint_at(x, y)
        self._render_canvas(fast=True)

    def _on_mouse_drag(self, e):
        self._on_canvas_motion(e)
        if not self.is_painting:
            return
        x, y = self._canvas_to_orig(e.x, e.y)
        if self.last_x is not None:
            dx, dy = x - self.last_x, y - self.last_y
            dist = max(1, int((dx**2 + dy**2) ** 0.5))
            step = max(1, int(self.brush_size / self.zoom) // 6)
            for i in range(0, dist, step):
                t = i / dist
                self._paint_at(int(self.last_x + dx * t), int(self.last_y + dy * t))
        self.last_x, self.last_y = x, y
        self._render_canvas(fast=True)

    def _on_mouse_up(self, e):
        if self.is_painting:
            self.is_painting = False
            self.last_x = self.last_y = None
            self._render_canvas()

    def _on_canvas_enter(self, e):
        if e.state & 0x100:
            self.is_painting = True

    def _on_brush_change(self, val):
        self.brush_size = int(val)
        self.brush_label.configure(text=f"{self.brush_size}px")

    def _clear_active_layer(self):
        if self.masks[self.active_layer] is not None:
            self.masks[self.active_layer][:] = 0
            self._render_canvas()

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
        return "…/" + "/".join(parts[-2:]) if len(parts) > 2 else p

    # ── SAM2 ───────────────────────────────────────────────────────────────────

    def _load_sam2(self):
        with self._sam2_lock:
            if self._sam2_predictor is not None:
                return self._sam2_predictor
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                if not os.path.exists(SAM2_CKPT):
                    raise FileNotFoundError(f"Checkpoint não encontrado: {SAM2_CKPT}")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._sam2_model     = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)
                self._sam2_predictor = SAM2ImagePredictor(self._sam2_model)
                return self._sam2_predictor
            except Exception as e:
                self._update_status(f"⚠️ SAM2 indisponível:\n{e}")
                return None

    def _load_sam2_auto(self):
        if self._sam2_auto_gen is not None:
            return self._sam2_auto_gen
        if self._load_sam2() is None:
            return None
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            self._sam2_auto_gen = SAM2AutomaticMaskGenerator(
                self._sam2_model,
                points_per_side=24,         # fewer points → bigger, whole-object masks
                pred_iou_thresh=0.78,        # slightly relaxed → keeps full objects
                stability_score_thresh=0.85,
                min_mask_region_area=1200,   # drop tiny fragments (skin patches, leaves)
                box_nms_thresh=0.65,         # merge near-duplicate boxes
                crop_n_layers=1,             # multi-scale: finds small AND large objects
                crop_overlap_ratio=0.4,
            )
            return self._sam2_auto_gen
        except Exception as e:
            self._update_status(f"⚠️ AutoMask: {e}")
            return None

    def _merge_object_masks(self, masks):
        """
        Absorb small masks that are mostly contained inside a larger mask.
        This turns over-segmented body-parts / leaf clusters into whole objects.
        """
        if len(masks) < 2:
            return masks

        # Sort largest first so parents come before children
        masks = sorted(masks, key=lambda m: m['area'], reverse=True)
        absorbed = [False] * len(masks)

        for i in range(len(masks)):
            if absorbed[i]:
                continue
            parent_seg = masks[i]['segmentation']
            for j in range(i + 1, len(masks)):
                if absorbed[j]:
                    continue
                child_seg  = masks[j]['segmentation']
                child_area = masks[j]['area']
                if child_area == 0:
                    continue
                # How much of child is inside parent?
                overlap = int(np.sum((parent_seg > 0) & (child_seg > 0)))
                if overlap / child_area > 0.50:          # >50% contained → merge
                    masks[i]['segmentation'] = np.maximum(parent_seg, child_seg)
                    masks[i]['area']         = int(np.sum(masks[i]['segmentation'] > 0))
                    parent_seg               = masks[i]['segmentation']
                    absorbed[j]              = True

        result = []
        for i, m in enumerate(masks):
            if not absorbed[i]:
                # Recompute centroid after merging
                ys, xs = np.nonzero(m['segmentation'] > 0)
                m['cx'] = int(xs.mean()) if len(xs) > 0 else m['cx']
                m['cy'] = int(ys.mean()) if len(ys) > 0 else m['cy']
                result.append(m)
        return result

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

    # ── Pre-processing (parallel SAM2 + depth) ─────────────────────────────────

    def _preprocess_image(self):
        if self.orig_image is None:
            return

        self._overlay_show("🔬", "Analisando imagem…",
                           "Inicializando modelos de IA…", 0.03)

        # ── Thread 1: SAM2 auto-segmentation ──────────────────────────────────
        def run_sam2():
            self._overlay_show("🔍", "Identificando objetos…",
                               "SAM2: mapeando segmentos semânticos da imagem…", 0.10)
            gen = self._load_sam2_auto()
            if gen is None or self.orig_image is None:
                return
            try:
                img_rgb = np.array(self.orig_image.convert("RGB"))
                H, W    = img_rgb.shape[:2]
                ps      = int(self.proc_size_var.get())
                scale   = min(1.0, ps / max(W, H))
                small   = cv2.resize(img_rgb, (int(W * scale), int(H * scale))) \
                          if scale < 1.0 else img_rgb

                self._overlay_show("🔍", "SAM2: segmentando objetos…",
                                   f"Resolução de análise: {small.shape[1]}×{small.shape[0]}px", 0.20)

                with torch.inference_mode():
                    raw = gen.generate(small)

                full = []
                for m in raw:
                    seg = m['segmentation'].astype(np.uint8) * 255
                    if scale < 1.0:
                        seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
                    area = int(np.sum(seg > 0))
                    ys, xs = np.nonzero(seg > 0)
                    cx = int(xs.mean()) if len(xs) > 0 else W // 2
                    cy = int(ys.mean()) if len(ys) > 0 else H // 2
                    full.append({'segmentation': seg, 'area': area, 'cx': cx, 'cy': cy})

                # ── Merge over-segmented parts into whole objects ──────────
                # If a smaller mask is >50% contained inside a larger mask,
                # they are parts of the same object → absorb into the parent.
                self._overlay_show("🔗", "Unindo partes de objetos…",
                                   "Combinando segmentos relacionados em objetos inteiros…", 0.35)
                full = self._merge_object_masks(full)

                self._auto_masks = full
                self._overlay_show("🎯", "Objetos identificados",
                                   f"{len(full)} elementos encontrados na imagem", 0.45)
            except Exception as e:
                _log('SAM2_AUTO', type(e), e, e.__traceback__)

        # ── Thread 2: depth map ────────────────────────────────────────────────
        def run_depth():
            self._overlay_show("📏", "Calculando mapa de profundidade…",
                               "ZoeDepth: estimando distância de cada pixel da câmera…", 0.50)
            model = self._load_zoedepth()
            if model is None or self.orig_image is None:
                return
            try:
                pil = self.orig_image.convert("RGB")
                with torch.inference_mode():
                    depth = model.infer_pil(pil)
                if isinstance(depth, torch.Tensor):
                    depth = depth.squeeze().cpu().numpy()
                mn, mx = float(depth.min()), float(depth.max())
                # Normalize: 1 = near (front), 0 = far (back)
                norm = (depth - mn) / max(mx - mn, 1e-6)
                self._depth_map = norm.astype(np.float32)
                self._overlay_show("📏", "Mapa de profundidade completo",
                                   "Profundidade métrica calibrada por ZoeDepth", 0.80)
            except Exception as e:
                _log('DEPTH', type(e), e, e.__traceback__)
                self._depth_map = None

        t1 = threading.Thread(target=run_sam2,  daemon=True)
        t2 = threading.Thread(target=run_depth, daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        parts = []
        if self._auto_masks:
            parts.append(f"🎯 {len(self._auto_masks)} objetos")
        if self._depth_map is not None:
            parts.append("📏 Depth map real")

        if parts:
            self._overlay_show("✅", "Análise completa!",
                               " | ".join(parts) + "\nIA pronta para uso", 1.0)
            time.sleep(1.0)
        else:
            self._overlay_show("⚠️", "Análise parcial",
                               "SAM2 ou ZoeDepth não disponíveis", 1.0)
            time.sleep(1.5)

        self._overlay_hide()
        self._update_status(" | ".join(parts) if parts else "⚠️ Análise incompleta")

    # ── Depth helpers ──────────────────────────────────────────────────────────

    def _depth_to_layer(self, img_x, img_y):
        """Map pixel depth to suggested layer index (near=0, far=N-1)."""
        if self._depth_map is None:
            return self.active_layer
        H, W      = self._depth_map.shape
        y         = min(H - 1, max(0, img_y))
        x         = min(W - 1, max(0, img_x))
        depth_val = float(self._depth_map[y, x])   # 1=near, 0=far
        bucket    = int((1.0 - depth_val) * self.num_layers)
        return min(max(bucket, 0), self.num_layers - 1)

    def _mask_mean_depth(self, seg):
        """Return average depth inside a segmentation mask (1=near, 0=far)."""
        if self._depth_map is None:
            return 0.0
        H, W = self._depth_map.shape
        h, w = seg.shape
        if (h, w) != (H, W):
            seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
        vals = self._depth_map[seg > 127]
        return float(vals.mean()) if len(vals) > 0 else 0.0

    # ── Magic select (semantic object, SAM2-only) ──────────────────────────────

    def _magic_select(self, img_x, img_y):
        # ── Fast path: pre-computed SAM2 masks ──────────────────────────────
        if self._auto_masks:
            candidates = []
            for m in self._auto_masks:
                seg = m['segmentation']
                if img_y < seg.shape[0] and img_x < seg.shape[1] and seg[img_y, img_x] > 127:
                    candidates.append(m)

            if candidates:
                # Pick smallest mask that contains the click (= most specific object)
                best = min(candidates, key=lambda m: m['area'])
                seg  = best['segmentation']

                suggested = self._depth_to_layer(best['cx'], best['cy'])
                if suggested != self.active_layer:
                    self.active_layer = suggested
                    self.after(0, self._rebuild_layer_buttons)

                self.masks[self.active_layer] = np.maximum(
                    self.masks[self.active_layer], seg)
                self.after(0, self._render_canvas)
                depth_info = (f"  profundidade: {self._depth_map[best['cy'], best['cx']]:.2f}"
                              if self._depth_map is not None else "")
                self._update_status(
                    f"✅ Objeto selecionado\n"
                    f"Layer {self.active_layer + 1} | {best['area']:,}px{depth_info}")
                return

        # ── Fallback: live SAM2 point prediction ─────────────────────────────
        if self._auto_masks is not None:
            # auto masks loaded but click not inside any → warn rather than fall to Canny
            self._update_status("⚠️ Nenhum objeto encontrado\nneste ponto.\nTente outro local.")
            return

        # Auto masks still loading → run live predictor
        predictor = self._load_sam2()
        if predictor is None or self.orig_image is None:
            self._update_status("⚠️ SAM2 indisponível")
            return
        try:
            ps    = int(self.proc_size_var.get())
            W, H  = self.orig_image.size
            scale = min(1.0, ps / max(W, H))
            img   = np.array(self.orig_image.convert("RGB"))
            if scale < 1.0:
                img_s  = cv2.resize(img, (int(W * scale), int(H * scale)))
                px, py = int(img_x * scale), int(img_y * scale)
            else:
                img_s, px, py = img, img_x, img_y

            predictor.set_image(img_s)
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[px, py]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=True)

            best   = int(np.argmax(scores))
            result = (masks[best] > 0.5).astype(np.uint8) * 255
            if scale < 1.0:
                result = cv2.resize(result, (W, H), interpolation=cv2.INTER_NEAREST)

            suggested = self._depth_to_layer(img_x, img_y)
            if suggested != self.active_layer:
                self.active_layer = suggested
                self.after(0, self._rebuild_layer_buttons)

            self.masks[self.active_layer] = np.maximum(
                self.masks[self.active_layer], result)
            self.after(0, self._render_canvas)
            self._update_status(
                f"✅ Objeto selecionado (SAM2)\n"
                f"score {scores[best]:.2f} | layer {self.active_layer + 1}")
        except Exception as e:
            self._update_status(f"❌ SAM2 erro: {e}")

    # ── Auto analysis ──────────────────────────────────────────────────────────

    def _auto_analyze(self):
        if self.orig_image is None:
            self._update_status("⚠ Carregue uma imagem primeiro")
            return
        if self._auto_masks is None:
            self._update_status("⏳ Aguardando análise…")
            threading.Thread(target=self._run_and_apply_auto, daemon=True).start()
        else:
            threading.Thread(target=self._apply_auto_masks, daemon=True).start()

    def _run_and_apply_auto(self):
        self._preprocess_image()
        if self._auto_masks:
            self._apply_auto_masks()

    def _apply_auto_masks(self):
        """Assign auto-detected objects to layers, ordered by DEPTH (near=Layer1)."""
        if not self._auto_masks or self.orig_image is None:
            return
        W, H     = self.orig_image.size
        total_px = W * H

        valid = [m for m in self._auto_masks
                 if 0.003 * total_px < m['area'] < 0.72 * total_px]

        if self._depth_map is not None:
            # Sort by mean depth of each segment: highest depth value = nearest = Layer 1
            valid.sort(key=lambda m: self._mask_mean_depth(m['segmentation']), reverse=True)
        else:
            # Fallback: larger area = assume foreground
            valid.sort(key=lambda m: m['area'], reverse=True)

        n = min(len(valid), self.num_layers)
        for i in range(n):
            seg = valid[i]['segmentation']
            if seg.shape != (H, W):
                seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
            self.masks[i] = seg
        for i in range(n, self.num_layers):
            if self.masks[i] is not None:
                self.masks[i][:] = 0

        self.after(0, self._render_canvas)
        method = "depth" if self._depth_map is not None else "área"
        self._update_status(f"🎉 {n} objetos → {n} layers\n(ordenados por {method})")

    # ── SAM2 mask refinement ───────────────────────────────────────────────────

    def _extract_prompt_points(self, mask, n=10):
        ys, xs = np.nonzero(mask > 10)
        if len(xs) == 0:
            return None, None
        pts = [(int(xs.mean()), int(ys.mean()))]
        if len(xs) >= n:
            order = np.argsort(xs.astype(np.int64) + ys.astype(np.int64))
            for k in range(1, n):
                idx = order[int((k / n) * len(order))]
                pts.append((int(xs[idx]), int(ys[idx])))
        arr = np.array(pts[:n], dtype=np.float32)
        return arr, np.ones(len(arr), dtype=np.int32)

    def _sam2_refine_mask(self, mask):
        predictor = self._load_sam2()
        if predictor is None:
            return self._canny_refine_mask(mask)

        W, H  = self.orig_image.size
        ps    = int(self.proc_size_var.get())
        scale = min(1.0, ps / max(W, H))
        img   = np.array(self.orig_image.convert("RGB"))

        if scale < 1.0:
            img_s  = cv2.resize(img,  (int(W * scale), int(H * scale)))
            mask_s = cv2.resize(mask, (int(W * scale), int(H * scale)),
                                interpolation=cv2.INTER_NEAREST)
        else:
            img_s, mask_s = img, mask

        points, labels = self._extract_prompt_points(mask_s, n=10)
        if points is None:
            return self._canny_refine_mask(mask)

        try:
            predictor.set_image(img_s)
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=points, point_labels=labels, multimask_output=True)
            best = (masks[int(np.argmax(scores))] > 0.5).astype(np.uint8) * 255
            if scale < 1.0:
                best = cv2.resize(best, (W, H), interpolation=cv2.INTER_NEAREST)
            best = cv2.morphologyEx(best, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            return best
        except Exception:
            return self._canny_refine_mask(mask)

    def _canny_refine_mask(self, mask):
        gray      = cv2.cvtColor(np.array(self.orig_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        blurred   = cv2.GaussianBlur(mask, (21, 21), 0)
        _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        edges     = cv2.Canny(gray, 50, 150)
        edges_in  = cv2.bitwise_and(
            cv2.dilate(edges,  np.ones((5,  5),  np.uint8), iterations=2),
            cv2.dilate(binary, np.ones((20, 20), np.uint8), iterations=1))
        combined  = cv2.bitwise_or(binary, edges_in)
        combined  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _get_feather_radius(self):
        return {"Fina": 3, "Média": 6, "Suave": 12}.get(self.feather_var.get(), 6)

    def _apply_mask_to_image(self, mask):
        """Apply feathered mask as alpha channel on the original image."""
        # 1. Morphological closing to fill small holes inside the object
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        # 2. Smooth the boundary
        feathered = _feather_mask(closed, self._get_feather_radius())
        arr = np.array(self.orig_image.convert("RGBA"))
        arr[:, :, 3] = feathered
        return Image.fromarray(arr, "RGBA")

    # ── SDXL Inpainting ────────────────────────────────────────────────────────

    def _load_sdxl(self):
        with self._sdxl_lock:
            if self._sdxl_pipe is not None:
                return self._sdxl_pipe
            try:
                from diffusers import StableDiffusionXLInpaintPipeline
                self._update_status("🔄 Carregando SDXL…\n(primeira vez é lento)")
                pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
                pipe.enable_model_cpu_offload()
                self._sdxl_pipe = pipe
                return pipe
            except Exception as e:
                self._update_status(f"⚠️ SDXL falhou:\n{e}\n→ usando OpenCV")
                return None

    def _inpaint_background(self, working_pil, binary_mask_np):
        """
        Fill binary_mask_np region in working_pil.
        binary_mask_np: uint8 H×W, 255 = pixels to fill.
        Returns PIL RGB.
        """
        mode = self._inpaint_mode.get()

        # Ensure binary, single-channel
        if binary_mask_np.ndim > 2:
            binary_mask_np = binary_mask_np[:, :, 0]
        binary_mask_np = (binary_mask_np > 127).astype(np.uint8) * 255

        if mode == "SDXL":
            pipe = self._load_sdxl()
            if pipe is not None:
                try:
                    W, H   = working_pil.size
                    scale  = 1024 / max(W, H)
                    img_r  = working_pil.resize((max(1, int(W * scale)),
                                                 max(1, int(H * scale))), Image.LANCZOS)
                    mask_r = Image.fromarray(binary_mask_np).resize(
                        (max(1, int(W * scale)), max(1, int(H * scale))), Image.NEAREST)

                    # Build a context-aware prompt by sampling dominant colors at mask boundary
                    prompt = self._build_inpaint_prompt(working_pil, binary_mask_np)

                    self._update_status(f"🎨 SDXL inpainting...\n\"{prompt[:60]}...\"")
                    out = pipe(
                        prompt=prompt,
                        negative_prompt=(
                            "blurry, distorted, artifacts, watermark, text, "
                            "seam, border, edge, low quality, deformed"),
                        image=img_r,
                        mask_image=mask_r,
                        num_inference_steps=30,
                        strength=0.99,
                        guidance_scale=8.0,
                    ).images[0]
                    return out.resize((W, H), Image.LANCZOS).convert("RGB")
                except Exception as e:
                    self._update_status(f"⚠️ SDXL erro → OpenCV: {e}")

        # OpenCV multi-scale Navier-Stokes — handles large regions much better
        img_bgr   = cv2.cvtColor(np.array(working_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        hole_area = int(np.sum(binary_mask_np > 0))

        if hole_area > 40000:
            # Large hole: first do a coarse fill at half resolution, then refine at full res
            H_orig, W_orig = img_bgr.shape[:2]
            img_half  = cv2.resize(img_bgr,        (W_orig // 2, H_orig // 2))
            mask_half = cv2.resize(binary_mask_np, (W_orig // 2, H_orig // 2),
                                   interpolation=cv2.INTER_NEAREST)
            mask_half = (mask_half > 127).astype(np.uint8) * 255
            coarse    = cv2.inpaint(img_half, mask_half, 25, cv2.INPAINT_NS)
            # Upscale coarse fill and use it to pre-fill the hole in the original
            coarse_up = cv2.resize(coarse, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            prefilled = img_bgr.copy()
            prefilled[binary_mask_np > 127] = coarse_up[binary_mask_np > 127]
            # Final refinement pass at full resolution with smaller radius
            result_bgr = cv2.inpaint(prefilled, binary_mask_np, 12, cv2.INPAINT_NS)
        else:
            radius     = min(40, max(10, int(np.sqrt(hole_area) * 0.04)))
            result_bgr = cv2.inpaint(img_bgr, binary_mask_np, radius, cv2.INPAINT_NS)

        return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

    def _build_inpaint_prompt(self, img_pil, mask_np):
        """Sample colors just outside the mask and build a descriptive prompt."""
        try:
            # Dilate mask to get the "border context" region
            k       = np.ones((21, 21), np.uint8)
            context = cv2.dilate(mask_np, k, iterations=3)
            border  = cv2.subtract(context, mask_np)   # ring just outside mask

            arr = np.array(img_pil.convert("RGB"))
            if arr.shape[:2] != border.shape:
                border = cv2.resize(border, (arr.shape[1], arr.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

            pixels = arr[border > 0]
            if len(pixels) == 0:
                return "seamless photorealistic background continuation, high quality"

            mean_rgb = pixels.mean(axis=0).astype(int)
            r, g, b  = mean_rgb

            # Simple color-to-description heuristic
            dominant = "neutral"
            if b > r and b > g:
                dominant = "sky blue"
            elif g > r and g > b:
                dominant = "green vegetation"
            elif r > g and r > b:
                dominant = "warm tones"
            elif r > 150 and g > 150 and b > 150:
                dominant = "bright light background"
            elif r < 80 and g < 80 and b < 80:
                dominant = "dark background"

            return (f"seamless photorealistic {dominant} background continuation, "
                    f"matching surrounding texture and lighting, high quality, no artifacts")
        except Exception:
            return "seamless photorealistic background continuation, high quality"

    # ── Results panel ──────────────────────────────────────────────────────────

    def _show_results(self, results):
        """results: list of (kind, path) — kind='layer' or 'background'."""
        for w in self._results_scroll.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        for kind, path in results:
            try:
                img = Image.open(path)
                if kind == 'layer':
                    img = img.convert("RGBA")
                    thumb = img.copy()
                    thumb.thumbnail((230, 170), Image.LANCZOS)
                    cb   = Image.fromarray(_checkerboard(thumb.width, thumb.height), "RGBA")
                    flat = Image.alpha_composite(cb, thumb)
                else:
                    img = img.convert("RGB")
                    thumb = img.copy()
                    thumb.thumbnail((230, 170), Image.LANCZOS)
                    flat = thumb

                photo = ImageTk.PhotoImage(flat)
                self._thumb_refs.append(photo)

                card = ctk.CTkFrame(self._results_scroll,
                                    fg_color="#13131f", corner_radius=8)
                card.pack(fill="x", pady=5, padx=4)

                tag_text = "✂️ Recorte" if kind == 'layer' else "🎨 Fundo inpainted"
                ctk.CTkLabel(card, text=tag_text,
                             font=ctk.CTkFont(size=9, weight="bold"),
                             text_color="#5eead4" if kind == 'layer' else "#a78bfa"
                             ).pack(pady=(6, 0))
                tk.Label(card, image=photo, bg="#13131f").pack(pady=2)
                ctk.CTkLabel(card, text=os.path.basename(path),
                             font=ctk.CTkFont(size=9),
                             text_color="gray").pack(pady=(0, 2))
                p = path
                ctk.CTkButton(card, text="📂  Abrir", height=26,
                              font=ctk.CTkFont(size=10),
                              command=lambda pp=p: os.startfile(pp)).pack(pady=(0, 8))
            except Exception:
                pass

    # ── Processing pipeline ────────────────────────────────────────────────────

    def _process(self):
        if self.orig_image is None:
            self._update_status("⚠ Carregue uma imagem primeiro!")
            return
        self._update_status("⏳ Processando…")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        try:
            out_dir = self._get_output_dir()
            os.makedirs(out_dir, exist_ok=True)
            stem    = os.path.splitext(os.path.basename(self.image_path or "img"))[0]

            painted = [i for i in range(self.num_layers)
                       if self.masks[i] is not None and self.masks[i].max() > 0]
            if not painted:
                self._update_status("⚠ Nenhuma layer pintada.")
                return

            results = []
            # Working image for progressive inpainting: starts as original RGB
            working = self.orig_image.convert("RGB")

            for idx, i in enumerate(painted):
                self._update_status(
                    f"🔍 Refinando Layer {i+1} ({idx+1}/{len(painted)})…")

                refined = self._sam2_refine_mask(self.masks[i])
                # Clean binary mask for inpainting
                _, bin_refined = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY)

                # Export feathered cutout from ORIGINAL image
                cutout   = self._apply_mask_to_image(refined)
                out_path = os.path.join(out_dir, f"{stem}_layer_{i+1}.png")
                cutout.save(out_path, optimize=False)
                results.append(('layer', out_path))
                self._update_status(f"✅ Layer {i+1} exportada com bordas suaves")

                # Inpaint hole in working image for next layer
                inpaint_mode = self._inpaint_mode.get()
                if inpaint_mode != "Desligado" and idx < len(painted) - 1:
                    self._update_status(
                        f"🎨 Preenchendo fundo Layer {i+1} ({inpaint_mode})…")
                    # Dilate slightly so we fill past the object edge
                    kernel  = np.ones((11, 11), np.uint8)
                    dilated = cv2.dilate(bin_refined, kernel, iterations=2)
                    working = self._inpaint_background(working, dilated)
                    bg_path = os.path.join(out_dir, f"{stem}_bg_layer_{i+1}.png")
                    working.save(bg_path)
                    results.append(('background', bg_path))

            self._update_status(
                f"🎉 {len(results)} arquivo(s) exportados!\n{self._short_path(out_dir)}")
            self.after(0, lambda: self._show_results(results))

        except Exception as ex:
            _log('PIPELINE', type(ex), ex, ex.__traceback__)
            self._update_status(f"❌ Erro: {ex}")

    def _update_status(self, text):
        self.after(0, lambda: self.status_label.configure(text=text))


if __name__ == "__main__":
    app = ParallaxStudio()
    app.mainloop()
