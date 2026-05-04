import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import threading
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SAM2_CONFIG   = 'configs/sam2.1/sam2.1_hiera_s'
SAM2_CKPT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_hiera_small.pt')

LAYER_COLORS = [
    (255, 61,  90,  "Layer 1 — Vermelho"),
    (0,   229, 255, "Layer 2 — Ciano"),
    (170, 255, 0,   "Layer 3 — Verde"),
    (255, 149, 0,   "Layer 4 — Laranja"),
    (191, 95,  255, "Layer 5 — Roxo"),
    (255, 230, 0,   "Layer 6 — Amarelo"),
    (0,   255, 178, "Layer 7 — Menta"),
    (255, 107, 202, "Layer 8 — Rosa"),
]


class ParallaxStudio(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Parallax Studio")
        self.geometry("1200x800")
        self.minsize(900, 600)

        self.image_path    = None
        self.orig_image    = None
        self.tk_image      = None
        self.num_layers    = 3
        self.active_layer  = 0
        self.brush_size    = 30
        self.tool          = "brush"
        self.zoom          = 1.0
        self.is_painting   = False
        self.last_x        = None
        self.last_y        = None
        self.scale_x       = 1.0
        self.scale_y       = 1.0
        self.masks         = [None] * 8

        # SAM2 — loaded lazily on first process call
        self._sam2_predictor = None
        self._sam2_device    = None
        self._sam2_loading   = False

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(20, weight=1)

        ctk.CTkLabel(self.sidebar, text="PARALLAX STUDIO",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, padx=16, pady=(20, 4), sticky="w")
        ctk.CTkLabel(self.sidebar, text="Pinte. A IA recorta.",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=1, column=0, padx=16, pady=(0, 16), sticky="w")

        ctk.CTkButton(self.sidebar, text="📁  Carregar Imagem",
                      command=self._load_image).grid(
            row=2, column=0, padx=12, pady=4, sticky="ew")

        ctk.CTkLabel(self.sidebar, text="Layers:",
                     font=ctk.CTkFont(size=12)).grid(
            row=3, column=0, padx=16, pady=(12, 0), sticky="w")
        layer_seg = ctk.CTkSegmentedButton(
            self.sidebar, values=["2","3","4","5","6","7","8"],
            command=self._on_layer_count_change)
        layer_seg.set("3")
        layer_seg.grid(row=4, column=0, padx=12, pady=4, sticky="ew")

        ctk.CTkLabel(self.sidebar, text="Layer ativa:",
                     font=ctk.CTkFont(size=12)).grid(
            row=5, column=0, padx=16, pady=(12, 0), sticky="w")
        self.layer_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.layer_frame.grid(row=6, column=0, padx=12, pady=4, sticky="ew")
        self.layer_buttons = []
        self._rebuild_layer_buttons()

        ctk.CTkLabel(self.sidebar, text="Ferramenta:",
                     font=ctk.CTkFont(size=12)).grid(
            row=7, column=0, padx=16, pady=(12, 0), sticky="w")
        self.tool_var = ctk.StringVar(value="brush")
        tool_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        tool_frame.grid(row=8, column=0, padx=12, pady=4, sticky="ew")
        ctk.CTkRadioButton(tool_frame, text="✦ Pincel", variable=self.tool_var,
                           value="brush",
                           command=lambda: setattr(self, 'tool', 'brush')).pack(side="left", padx=4)
        ctk.CTkRadioButton(tool_frame, text="◻ Borracha", variable=self.tool_var,
                           value="eraser",
                           command=lambda: setattr(self, 'tool', 'eraser')).pack(side="left", padx=4)

        ctk.CTkLabel(self.sidebar, text="Tamanho do pincel:",
                     font=ctk.CTkFont(size=12)).grid(
            row=9, column=0, padx=16, pady=(12, 0), sticky="w")
        self.brush_slider = ctk.CTkSlider(self.sidebar, from_=4, to=120,
                                          command=self._on_brush_change)
        self.brush_slider.set(30)
        self.brush_slider.grid(row=10, column=0, padx=12, pady=4, sticky="ew")
        self.brush_label = ctk.CTkLabel(self.sidebar, text="30px",
                                        font=ctk.CTkFont(size=11))
        self.brush_label.grid(row=11, column=0, padx=16, sticky="w")

        ctk.CTkLabel(self.sidebar, text="Zoom:",
                     font=ctk.CTkFont(size=12)).grid(
            row=12, column=0, padx=16, pady=(12, 0), sticky="w")
        zoom_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        zoom_frame.grid(row=13, column=0, padx=12, pady=4, sticky="ew")
        ctk.CTkButton(zoom_frame, text="−", width=36,
                      command=lambda: self._set_zoom(self.zoom - 0.1)).pack(side="left", padx=2)
        self.zoom_label = ctk.CTkLabel(zoom_frame, text="100%", width=50)
        self.zoom_label.pack(side="left", padx=2)
        ctk.CTkButton(zoom_frame, text="+", width=36,
                      command=lambda: self._set_zoom(self.zoom + 0.1)).pack(side="left", padx=2)
        ctk.CTkButton(zoom_frame, text="⊡", width=36,
                      command=lambda: self._set_zoom(1.0)).pack(side="left", padx=2)

        ctk.CTkButton(self.sidebar, text="🗑  Limpar Layer",
                      fg_color="transparent", border_width=1,
                      command=self._clear_active_layer).grid(
            row=14, column=0, padx=12, pady=(12, 4), sticky="ew")

        ctk.CTkButton(self.sidebar, text="⚡  Processar com IA",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      fg_color="#5eead4", text_color="#000",
                      hover_color="#2dd4bf",
                      command=self._process).grid(
            row=19, column=0, padx=12, pady=(0, 8), sticky="ew")

        self.status_label = ctk.CTkLabel(self.sidebar, text="Carregue uma imagem",
                                         font=ctk.CTkFont(size=11), text_color="gray",
                                         wraplength=200)
        self.status_label.grid(row=20, column=0, padx=12, pady=8, sticky="sw")

        canvas_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#0a0a0f")
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        import tkinter as tk
        self.canvas = tk.Canvas(canvas_frame, bg="#0a0a0f",
                                cursor="crosshair", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.vscroll = ctk.CTkScrollbar(canvas_frame, command=self.canvas.yview)
        self.vscroll.grid(row=0, column=1, sticky="ns")
        self.hscroll = ctk.CTkScrollbar(canvas_frame, orientation="horizontal",
                                        command=self.canvas.xview)
        self.hscroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=self.vscroll.set,
                              xscrollcommand=self.hscroll.set)

        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_down)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Enter>",           self._on_canvas_enter)

    # ── Layer buttons ─────────────────────────────────────────────────────────

    def _rebuild_layer_buttons(self):
        for w in self.layer_frame.winfo_children():
            w.destroy()
        self.layer_buttons = []
        for i in range(self.num_layers):
            r, g, b, name = LAYER_COLORS[i]
            color = f"#{r:02x}{g:02x}{b:02x}"
            btn = ctk.CTkButton(
                self.layer_frame, text=name,
                fg_color=color if i == self.active_layer else "transparent",
                text_color="#000" if i == self.active_layer else "#fff",
                border_color=color, border_width=2,
                command=lambda idx=i: self._set_active_layer(idx))
            btn.pack(fill="x", pady=2)
            self.layer_buttons.append(btn)

    def _set_active_layer(self, idx):
        self.active_layer = idx
        self._rebuild_layer_buttons()

    def _on_layer_count_change(self, val):
        self.num_layers = int(val)
        self._rebuild_layer_buttons()

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if not path:
            return
        self.image_path = path
        self.orig_image = Image.open(path).convert("RGBA")
        W, H = self.orig_image.size
        self.masks = [np.zeros((H, W), dtype=np.uint8) for _ in range(8)]
        self._set_zoom(1.0)
        self.status_label.configure(text=f"{os.path.basename(path)}\n{W}×{H}px")

    def _set_zoom(self, z):
        self.zoom = max(0.1, min(5.0, round(z, 1)))
        self.zoom_label.configure(text=f"{int(self.zoom*100)}%")
        if self.orig_image:
            self._render_canvas()

    # ── Canvas render ─────────────────────────────────────────────────────────

    def _render_canvas(self):
        if not self.orig_image:
            return
        W, H = self.orig_image.size
        dw = max(1, int(W * self.zoom))
        dh = max(1, int(H * self.zoom))
        self.scale_x = dw / W
        self.scale_y = dh / H

        base = self.orig_image.resize((dw, dh), Image.LANCZOS).convert("RGBA")

        for i in range(self.num_layers):
            if self.masks[i] is None:
                continue
            mask_resized = cv2.resize(self.masks[i], (dw, dh),
                                      interpolation=cv2.INTER_NEAREST)
            if mask_resized.max() == 0:
                continue
            r, g, b, _ = LAYER_COLORS[i]
            overlay = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
            arr = np.array(overlay)
            alpha_vals = (mask_resized.astype(np.float32) / 255 * 160).astype(np.uint8)
            arr[:, :, 0] = np.where(mask_resized > 0, r, 0)
            arr[:, :, 1] = np.where(mask_resized > 0, g, 0)
            arr[:, :, 2] = np.where(mask_resized > 0, b, 0)
            arr[:, :, 3] = alpha_vals
            base = Image.alpha_composite(base, Image.fromarray(arr, "RGBA"))

        self.tk_image = ImageTk.PhotoImage(base)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.configure(scrollregion=(0, 0, dw, dh))

    # ── Painting ──────────────────────────────────────────────────────────────

    def _canvas_to_orig(self, cx, cy):
        x = int(self.canvas.canvasx(cx) / self.scale_x)
        y = int(self.canvas.canvasy(cy) / self.scale_y)
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
        else:
            cv2.circle(mask, (x, y), int(r * 1.5), 0, -1)

    def _on_mouse_down(self, e):
        self.is_painting = True
        x, y = self._canvas_to_orig(e.x, e.y)
        self.last_x, self.last_y = x, y
        self._paint_at(x, y)
        self._render_canvas()

    def _on_mouse_move(self, e):
        if not self.is_painting:
            return
        x, y = self._canvas_to_orig(e.x, e.y)
        if self.last_x is not None:
            dx, dy = x - self.last_x, y - self.last_y
            dist = max(1, int((dx**2 + dy**2) ** 0.5))
            step = max(1, int(self.brush_size / self.zoom) // 3)
            for i in range(0, dist, step):
                t = i / dist
                self._paint_at(int(self.last_x + dx * t), int(self.last_y + dy * t))
        self.last_x, self.last_y = x, y
        self._render_canvas()

    def _on_mouse_up(self, e):
        self.is_painting = False
        self.last_x = self.last_y = None

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

    # ── SAM2 ─────────────────────────────────────────────────────────────────

    def _load_sam2(self):
        """Load SAM2 model on first call; returns predictor or None on failure."""
        if self._sam2_predictor is not None:
            return self._sam2_predictor

        self._update_status("🔄 Carregando SAM2...")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if not os.path.exists(SAM2_CKPT):
                raise FileNotFoundError(f"Checkpoint não encontrado: {SAM2_CKPT}")

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)
            self._sam2_predictor = SAM2ImagePredictor(model)
            self._sam2_device    = device
            self._update_status(f"✅ SAM2 carregado ({device.upper()})")
            return self._sam2_predictor
        except Exception as e:
            self._update_status(f"⚠️ SAM2 indisponível: {e}\n→ usando Canny")
            return None

    def _extract_prompt_points(self, mask, n=6):
        """Return (points, labels) sampled from the painted region."""
        ys, xs = np.nonzero(mask > 10)
        if len(xs) == 0:
            return None, None

        # Always include the centroid
        pts = [(int(xs.mean()), int(ys.mean()))]

        # Add spatially distributed samples if region is large enough
        if len(xs) >= n:
            # Sort pixels by x+y for deterministic diagonal spread
            order = np.argsort(xs + ys)
            for k in range(1, n):
                idx = order[int((k / n) * len(order))]
                pts.append((int(xs[idx]), int(ys[idx])))

        arr = np.array(pts[:n], dtype=np.float32)
        labels = np.ones(len(arr), dtype=np.int32)
        return arr, labels

    def _sam2_refine_mask(self, mask):
        """Use SAM2 point prompts to generate a pixel-precise mask."""
        predictor = self._load_sam2()
        if predictor is None:
            return self._canny_refine_mask(mask)

        img_rgb = np.array(self.orig_image.convert("RGB"))
        points, labels = self._extract_prompt_points(mask)
        if points is None:
            return self._canny_refine_mask(mask)

        try:
            predictor.set_image(img_rgb)
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False,
                )
            # masks: (1, H, W) — bool or float32
            binary = (masks[0] > 0.5).astype(np.uint8) * 255
            # Gentle feather for anti-aliased edges
            return cv2.GaussianBlur(binary, (5, 5), 0)
        except Exception as e:
            self._update_status(f"⚠️ SAM2 erro: {e}\n→ usando Canny")
            return self._canny_refine_mask(mask)

    # ── Fallback: Canny-based refinement ─────────────────────────────────────

    def _canny_refine_mask(self, mask):
        """Original edge-based mask refinement — used as SAM2 fallback."""
        orig_cv = cv2.cvtColor(np.array(self.orig_image.convert("RGB")),
                               cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(mask, (21, 21), 0)
        _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

        edges = cv2.Canny(gray, 50, 150)
        edges_in_mask = cv2.bitwise_and(
            cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2),
            cv2.dilate(binary, np.ones((20, 20), np.uint8), iterations=1))

        combined = cv2.bitwise_or(binary, edges_in_mask)
        kernel = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.GaussianBlur(combined, (9, 9), 0)

    def _apply_mask_to_image(self, mask):
        arr = np.array(self.orig_image.convert("RGBA"))
        arr[:, :, 3] = mask
        return Image.fromarray(arr, "RGBA")

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _process(self):
        if self.orig_image is None:
            self.status_label.configure(text="⚠ Carregue uma imagem primeiro!")
            return
        self.status_label.configure(text="⏳ Processando...")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        try:
            results = []
            for i in range(self.num_layers):
                mask = self.masks[i]
                if mask is None or mask.max() == 0:
                    continue

                self._update_status(f"🔍 SAM2: Layer {i+1}…")
                refined = self._sam2_refine_mask(mask)
                cutout  = self._apply_mask_to_image(refined)

                out_dir  = os.path.dirname(self.image_path) if self.image_path else "."
                out_path = os.path.join(out_dir, f"parallax_layer_{i+1}.png")
                cutout.save(out_path)
                results.append(out_path)
                self._update_status(f"✅ Layer {i+1} salva!")

            if results:
                self._update_status(
                    f"🎉 {len(results)} layer(s) exportadas!\n" +
                    "\n".join(os.path.basename(r) for r in results))
            else:
                self._update_status("⚠ Nenhuma layer pintada para processar.")
        except Exception as ex:
            self._update_status(f"❌ Erro: {ex}")

    def _update_status(self, text):
        self.after(0, lambda: self.status_label.configure(text=text))


if __name__ == "__main__":
    app = ParallaxStudio()
    app.mainloop()
