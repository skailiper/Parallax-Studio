import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import threading
import os
import json

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SAM2_CONFIG   = 'configs/sam2.1/sam2.1_hiera_s'
SAM2_CKPT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_hiera_small.pt')
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')

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

try:
    import windnd
    _HAS_DND = True
except ImportError:
    _HAS_DND = False


def _checkerboard(w, h, ts=10):
    arr = np.full((h, w, 4), 40, dtype=np.uint8)
    arr[:, :, 3] = 255
    rows = np.arange(h) // ts
    cols = np.arange(w) // ts
    light = (rows[:, None] + cols[None, :]) % 2 == 0
    arr[light] = [62, 62, 62, 255]
    return arr


class ParallaxStudio(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Parallax Studio")
        self.geometry("1480x920")
        self.minsize(1100, 680)

        # ── Core state ─────────────────────────────────────────────────────────
        self.image_path   = None
        self.orig_image   = None   # PIL RGBA, original resolution
        self._base_cache  = {}     # zoom → PIL Image (base render cache)
        self.tk_image     = None
        self.num_layers   = 3
        self.active_layer = 0
        self.brush_size   = 30
        self.tool         = "brush"   # brush | eraser | magic
        self.zoom         = 1.0
        self.is_painting  = False
        self.last_x       = None
        self.last_y       = None
        self.masks        = [None] * 8
        self.output_dir   = None
        self._brush_oval  = None

        # ── SAM2 state (lazy) ──────────────────────────────────────────────────
        self._sam2_model      = None
        self._sam2_predictor  = None
        self._sam2_auto_gen   = None
        self._sam2_device     = None
        self._auto_masks      = None   # pre-computed on image load
        self._sam2_lock       = threading.Lock()

        # ── Result thumbnails (keep refs so GC doesn't collect) ────────────────
        self._thumb_refs = []

        self._load_settings()
        self._build_ui()

        if _HAS_DND:
            windnd.hook_dropfiles(self, func=self._on_windnd_drop)

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

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_canvas_area()
        self._build_results_panel()

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=240, corner_radius=0)
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

        # Layers
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

        # Tools
        ctk.CTkLabel(sb, text="Ferramenta:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(6, 0), sticky="w"); r += 1
        self.tool_var = ctk.StringVar(value="brush")
        tf = ctk.CTkFrame(sb, fg_color="transparent")
        tf.grid(row=r, column=0, padx=16, pady=3, sticky="w"); r += 1
        for val, label in [("brush", "✦ Pincel"), ("eraser", "◻ Borracha"), ("magic", "🪄 Seleção Mágica")]:
            ctk.CTkRadioButton(tf, text=label, variable=self.tool_var, value=val,
                               command=lambda v=val: setattr(self, 'tool', v)
                               ).pack(anchor="w", pady=1)

        # Brush size
        ctk.CTkLabel(sb, text="Tamanho do pincel:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1
        self.brush_slider = ctk.CTkSlider(sb, from_=4, to=300,
                                          command=self._on_brush_change)
        self.brush_slider.set(30)
        self.brush_slider.grid(row=r, column=0, padx=12, pady=2, sticky="ew"); r += 1
        self.brush_label = ctk.CTkLabel(sb, text="30px", font=ctk.CTkFont(size=11))
        self.brush_label.grid(row=r, column=0, padx=16, sticky="w"); r += 1

        # Proc resolution
        ctk.CTkLabel(sb, text="Resolução SAM2:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1
        self.proc_size_var = ctk.StringVar(value="1024")
        ctk.CTkSegmentedButton(sb, values=["512", "1024", "2048"],
                               variable=self.proc_size_var).grid(
            row=r, column=0, padx=12, pady=3, sticky="ew"); r += 1

        # Zoom
        ctk.CTkLabel(sb, text="Zoom:", font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1
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

        # Output dir
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
            font=ctk.CTkFont(size=11), text_color="gray", wraplength=220)
        self.status_label.grid(row=99, column=0, padx=12, pady=10, sticky="sw")

    def _sep(self, parent, row):
        ctk.CTkFrame(parent, height=1, fg_color="#222233").grid(
            row=row, column=0, padx=12, pady=4, sticky="ew")

    def _build_canvas_area(self):
        frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#08080f")
        frame.grid(row=0, column=1, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(frame, bg="#08080f",
                                cursor="crosshair", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.vscroll = ctk.CTkScrollbar(frame, command=self.canvas.yview)
        self.vscroll.grid(row=0, column=1, sticky="ns")
        self.hscroll = ctk.CTkScrollbar(frame, orientation="horizontal",
                                        command=self.canvas.xview)
        self.hscroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=self.vscroll.set,
                              xscrollcommand=self.hscroll.set)

        # Drop-zone hint
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

    def _build_results_panel(self):
        frame = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color="#0b0b18")
        frame.grid(row=0, column=2, sticky="nsew")
        frame.grid_propagate(False)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="RESULTADOS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#444466").grid(
            row=0, column=0, padx=14, pady=(14, 6), sticky="w")

        self._results_scroll = ctk.CTkScrollableFrame(
            frame, fg_color="transparent")
        self._results_scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._results_scroll.grid_columnconfigure(0, weight=1)

    # ── Layer UI ───────────────────────────────────────────────────────────────

    def _rebuild_layer_buttons(self):
        for w in self.layer_frame.winfo_children():
            w.destroy()
        self.layer_buttons = []
        for i in range(self.num_layers):
            r, g, b, name = LAYER_COLORS[i]
            hex_c = f"#{r:02x}{g:02x}{b:02x}"
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

    # ── Image loading ──────────────────────────────────────────────────────────

    def _browse_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if path:
            self._load_image(path)

    def _on_windnd_drop(self, files):
        if not files:
            return
        raw = files[0]
        path = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else str(raw)
        self.after(0, lambda: self._load_image(path.strip().strip('"')))

    def _load_image(self, path):
        path = path.strip().strip('"').strip("'")
        if not os.path.isfile(path):
            return
        self.image_path  = path
        self.orig_image  = Image.open(path).convert("RGBA")
        W, H             = self.orig_image.size
        self.masks       = [np.zeros((H, W), dtype=np.uint8) for _ in range(8)]
        self._base_cache = {}
        self._auto_masks = None

        self._drop_hint.place_forget()
        self.after(80, self._zoom_to_fit)

        name = os.path.basename(path)
        self._update_status(f"✅ {name}\n{W}×{H}px")
        threading.Thread(target=self._preprocess_image, daemon=True).start()

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

        # Cache base image per zoom level — avoid redundant LANCZOS resampling
        z_key = self.zoom
        if z_key not in self._base_cache:
            interp = Image.BILINEAR if fast else Image.LANCZOS
            base_pil = self.orig_image.resize((dw, dh), interp)
            if not fast:
                self._base_cache = {z_key: base_pil}  # only cache final quality
        else:
            base_pil = self._base_cache[z_key]

        # Composit mask overlays via numpy (fast, avoids PIL alpha_composite loop)
        arr = np.array(base_pil.convert("RGB"), dtype=np.float32)

        for i in range(self.num_layers):
            if self.masks[i] is None or self.masks[i].max() == 0:
                continue
            mr = cv2.resize(self.masks[i], (dw, dh),
                            interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
            rv, gv, bv, _ = LAYER_COLORS[i]
            alpha = mr * 0.55  # overlay strength
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
            r, outline, dash = 12, "#a855f7", ()
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
        if self.orig_image is None:
            return
        x, y = self._canvas_to_orig(e.x, e.y)
        if self.tool == "magic":
            self._update_status(f"🪄 Seleção mágica…")
            threading.Thread(target=self._magic_select, args=(x, y),
                             daemon=True).start()
            return
        self.is_painting = True
        self.last_x, self.last_y = x, y
        self._paint_at(x, y)
        self._render_canvas(fast=True)

    def _on_mouse_drag(self, e):
        self._on_canvas_motion(e)   # keep brush oval updated while dragging
        if not self.is_painting:
            return
        x, y = self._canvas_to_orig(e.x, e.y)
        if self.last_x is not None:
            dx, dy = x - self.last_x, y - self.last_y
            dist = max(1, int((dx**2 + dy**2) ** 0.5))
            step = max(1, int(self.brush_size / self.zoom) // 6)  # smooth: 6× denser
            for i in range(0, dist, step):
                t = i / dist
                self._paint_at(int(self.last_x + dx * t), int(self.last_y + dy * t))
        self.last_x, self.last_y = x, y
        self._render_canvas(fast=True)

    def _on_mouse_up(self, e):
        if self.is_painting:
            self.is_painting = False
            self.last_x = self.last_y = None
            self._render_canvas()   # final LANCZOS quality render

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

    # ── Output directory ───────────────────────────────────────────────────────

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

    # ── SAM2 loading ───────────────────────────────────────────────────────────

    def _load_sam2(self):
        with self._sam2_lock:
            if self._sam2_predictor is not None:
                return self._sam2_predictor
            self._update_status("🔄 Carregando SAM2…")
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                if not os.path.exists(SAM2_CKPT):
                    raise FileNotFoundError(f"Checkpoint não encontrado:\n{SAM2_CKPT}")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._sam2_model     = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)
                self._sam2_predictor = SAM2ImagePredictor(self._sam2_model)
                self._sam2_device    = device
                self._update_status(f"✅ SAM2 pronto ({device.upper()})")
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
                points_per_side=32,
                pred_iou_thresh=0.80,
                stability_score_thresh=0.88,
                min_mask_region_area=800,
            )
            return self._sam2_auto_gen
        except Exception as e:
            self._update_status(f"⚠️ AutoMask: {e}")
            return None

    # ── Pre-processing on image load ───────────────────────────────────────────

    def _preprocess_image(self):
        """Background thread: load SAM2 + run auto segmentation so Análise Automática is instant."""
        gen = self._load_sam2_auto()
        if gen is None or self.orig_image is None:
            return
        try:
            self._update_status("⏳ Pré-analisando imagem…")
            img_rgb  = np.array(self.orig_image.convert("RGB"))
            H, W     = img_rgb.shape[:2]
            ps       = int(self.proc_size_var.get())
            scale    = min(1.0, ps / max(W, H))
            small    = cv2.resize(img_rgb, (int(W * scale), int(H * scale))) if scale < 1.0 else img_rgb

            with torch.inference_mode():
                raw_masks = gen.generate(small)

            # Scale segmentations back to full resolution
            full = []
            for m in raw_masks:
                seg = (m['segmentation'].astype(np.uint8)) * 255
                if scale < 1.0:
                    seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
                full.append({'segmentation': seg, 'area': int(np.sum(seg > 0))})

            self._auto_masks = full
            n = len(full)
            self._update_status(f"✅ {n} objetos detectados\n→ 'Análise Automática' pronta")
        except Exception as e:
            self._update_status(f"⚠️ Pré-análise: {e}")

    # ── Magic select ───────────────────────────────────────────────────────────

    def _magic_select(self, img_x, img_y):
        predictor = self._load_sam2()
        if predictor is None or self.orig_image is None:
            return
        try:
            ps    = int(self.proc_size_var.get())
            W, H  = self.orig_image.size
            scale = min(1.0, ps / max(W, H))
            img   = np.array(self.orig_image.convert("RGB"))

            if scale < 1.0:
                img_s = cv2.resize(img, (int(W * scale), int(H * scale)))
                px, py = int(img_x * scale), int(img_y * scale)
            else:
                img_s, px, py = img, img_x, img_y

            predictor.set_image(img_s)
            pts    = np.array([[px, py]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)

            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=pts,
                    point_labels=labels,
                    multimask_output=True,
                )

            best   = int(np.argmax(scores))
            result = (masks[best] > 0.5).astype(np.uint8) * 255
            if scale < 1.0:
                result = cv2.resize(result, (W, H), interpolation=cv2.INTER_NEAREST)
            result = cv2.GaussianBlur(result, (3, 3), 0)

            self.masks[self.active_layer] = np.maximum(
                self.masks[self.active_layer], result)
            self.after(0, self._render_canvas)
            self._update_status(f"✅ Mágica: score {scores[best]:.2f}")
        except Exception as e:
            self._update_status(f"❌ Mágica: {e}")

    # ── Auto analysis ──────────────────────────────────────────────────────────

    def _auto_analyze(self):
        if self.orig_image is None:
            self._update_status("⚠ Carregue uma imagem primeiro")
            return
        if self._auto_masks is None:
            self._update_status("⏳ Aguarde a pré-análise…")
            threading.Thread(target=self._run_and_apply_auto, daemon=True).start()
        else:
            threading.Thread(target=self._apply_auto_masks, daemon=True).start()

    def _run_and_apply_auto(self):
        self._preprocess_image()
        if self._auto_masks:
            self._apply_auto_masks()

    def _apply_auto_masks(self):
        if not self._auto_masks or self.orig_image is None:
            return
        W, H      = self.orig_image.size
        total_px  = W * H

        valid = [(m['area'], m['segmentation']) for m in self._auto_masks
                 if 0.003 * total_px < m['area'] < 0.72 * total_px]
        valid.sort(key=lambda x: x[0], reverse=True)   # largest → front layers

        n = min(len(valid), self.num_layers)
        for i in range(n):
            seg = valid[i][1]
            if seg.shape != (H, W):
                seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
            self.masks[i] = seg
        for i in range(n, self.num_layers):
            if self.masks[i] is not None:
                self.masks[i][:] = 0

        self.after(0, self._render_canvas)
        self._update_status(f"🎉 {n} objetos → {n} layer(s)")

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
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,
                )
            best = (masks[int(np.argmax(scores))] > 0.5).astype(np.uint8) * 255
            if scale < 1.0:
                best = cv2.resize(best, (W, H), interpolation=cv2.INTER_NEAREST)
            # Light morphological cleanup + feather
            best = cv2.morphologyEx(best, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            return cv2.GaussianBlur(best, (5, 5), 0)
        except Exception as e:
            self._update_status(f"⚠️ SAM2 erro: {e}")
            return self._canny_refine_mask(mask)

    def _canny_refine_mask(self, mask):
        gray     = cv2.cvtColor(np.array(self.orig_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        blurred  = cv2.GaussianBlur(mask, (21, 21), 0)
        _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        edges    = cv2.Canny(gray, 50, 150)
        edges_in = cv2.bitwise_and(
            cv2.dilate(edges,  np.ones((5,  5),  np.uint8), iterations=2),
            cv2.dilate(binary, np.ones((20, 20), np.uint8), iterations=1))
        combined = cv2.bitwise_or(binary, edges_in)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
        return cv2.GaussianBlur(combined, (9, 9), 0)

    def _apply_mask_to_image(self, mask):
        arr = np.array(self.orig_image.convert("RGBA"))
        arr[:, :, 3] = mask
        return Image.fromarray(arr, "RGBA")

    # ── Results preview ────────────────────────────────────────────────────────

    def _show_results(self, paths):
        for w in self._results_scroll.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        for path in paths:
            try:
                img   = Image.open(path).convert("RGBA")
                thumb = img.copy()
                thumb.thumbnail((230, 170), Image.LANCZOS)

                cb  = Image.fromarray(_checkerboard(thumb.width, thumb.height), "RGBA")
                flat = Image.alpha_composite(cb, thumb)

                photo = ImageTk.PhotoImage(flat)
                self._thumb_refs.append(photo)

                card = ctk.CTkFrame(self._results_scroll,
                                    fg_color="#13131f", corner_radius=8)
                card.pack(fill="x", pady=5, padx=4)

                tk.Label(card, image=photo, bg="#13131f").pack(pady=(8, 2))
                ctk.CTkLabel(card, text=os.path.basename(path),
                             font=ctk.CTkFont(size=10),
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
            self.status_label.configure(text="⚠ Carregue uma imagem primeiro!")
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
            for idx, i in enumerate(painted):
                self._update_status(f"🔍 SAM2: Layer {i+1} ({idx+1}/{len(painted)})…")
                refined  = self._sam2_refine_mask(self.masks[i])
                cutout   = self._apply_mask_to_image(refined)
                out_path = os.path.join(out_dir, f"{stem}_layer_{i+1}.png")
                cutout.save(out_path)
                results.append(out_path)
                self._update_status(f"✅ Layer {i+1} salva")

            self._update_status(
                f"🎉 {len(results)} layer(s) exportadas!\n{self._short_path(out_dir)}")
            self.after(0, lambda: self._show_results(results))

        except Exception as ex:
            self._update_status(f"❌ Erro: {ex}")

    def _update_status(self, text):
        self.after(0, lambda: self.status_label.configure(text=text))


if __name__ == "__main__":
    app = ParallaxStudio()
    app.mainloop()
