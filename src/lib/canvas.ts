import type { LayerColor } from '../hooks/usePipeline';

export function canvasToJpeg(canvas: HTMLCanvasElement, quality = 0.85): string {
  return canvas.toDataURL('image/jpeg', quality).split(',')[1];
}

export function canvasToPng(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL('image/png').split(',')[1];
}

export function createCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = w; c.height = h; return c;
}

export function resizeToFit(img: HTMLImageElement, maxPx = 900): { canvas: HTMLCanvasElement; w: number; h: number; scale: number } {
  const scale = Math.min(1, maxPx / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.round(img.naturalWidth * scale), h = Math.round(img.naturalHeight * scale);
  const c = createCanvas(w, h);
  c.getContext('2d')!.drawImage(img, 0, 0, w, h);
  return { canvas: c, w, h, scale };
}

export function resizeToStability(img: HTMLImageElement): { canvas: HTMLCanvasElement; w: number; h: number } {
  const maxPx = 1024;
  const scale = Math.min(1, maxPx / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.round(img.naturalWidth * scale / 64) * 64;
  const h = Math.round(img.naturalHeight * scale / 64) * 64;
  const c = createCanvas(w, h);
  c.getContext('2d')!.drawImage(img, 0, 0, w, h);
  return { canvas: c, w, h };
}

export function buildSketchOverlay(
  img: HTMLImageElement,
  maskCanvases: HTMLCanvasElement[],
  numLayers: number,
  colors: LayerColor[],
  tw: number,
  th: number,
): HTMLCanvasElement {
  const c = createCanvas(tw, th);
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0, tw, th);
  for (let i = 0; i < numLayers; i++) {
    const tmp = createCanvas(tw, th);
    const tCtx = tmp.getContext('2d')!;
    tCtx.drawImage(maskCanvases[i], 0, 0, tw, th);
    const td = tCtx.getImageData(0, 0, tw, th);
    const [r, g, b] = colors[i].rgb;
    for (let p = 0; p < td.data.length; p += 4)
      if (td.data[p + 3] > 0) { td.data[p]=r; td.data[p+1]=g; td.data[p+2]=b; td.data[p+3]=200; }
    tCtx.putImageData(td, 0, 0);
    ctx.drawImage(tmp, 0, 0);
  }
  return c;
}

export function buildInpaintMask(maskCanvases: HTMLCanvasElement[], layerIdx: number, W: number, H: number): HTMLCanvasElement {
  const c = createCanvas(W, H);
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = 'black'; ctx.fillRect(0, 0, W, H);
  for (let j = 0; j < layerIdx; j++) {
    const mData = maskCanvases[j].getContext('2d')!.getImageData(0, 0, W, H);
    const iData = ctx.getImageData(0, 0, W, H);
    for (let p = 0; p < mData.data.length; p += 4)
      if (mData.data[p + 3] > 30) { iData.data[p]=255; iData.data[p+1]=255; iData.data[p+2]=255; iData.data[p+3]=255; }
    ctx.putImageData(iData, 0, 0);
  }
  const fc = createCanvas(W, H);
  const fCtx = fc.getContext('2d')!;
  fCtx.filter = 'blur(10px)'; fCtx.drawImage(c, 0, 0); fCtx.filter = 'none';
  const fd = fCtx.getImageData(0, 0, W, H);
  for (let p = 0; p < fd.data.length; p += 4) {
    const v = fd.data[p] > 15 ? 255 : 0;
    fd.data[p]=v; fd.data[p+1]=v; fd.data[p+2]=v; fd.data[p+3]=255;
  }
  fCtx.putImageData(fd, 0, 0);
  return fc;
}

export function buildCutout(img: HTMLImageElement, W: number, H: number, alphaMap: Float32Array): HTMLCanvasElement {
  const c = createCanvas(W, H);
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, W, H);
  for (let p = 0; p < d.data.length; p += 4)
    d.data[p + 3] = Math.round(Math.min(255, alphaMap[p / 4] * 400));
  ctx.putImageData(d, 0, 0);
  return c;
}

export function expandStrokeToAlphaMap(strokeData: ImageData, W: number, H: number, brushSize: number, expansionFactor = 2.2): Float32Array {
  const base = new Float32Array(W * H);
  for (let p = 3; p < strokeData.data.length; p += 4) base[(p - 3) / 4] = strokeData.data[p] / 255;
  const radius = brushSize * expansionFactor;
  const result = new Float32Array(W * H);
  const step = Math.max(1, Math.round(radius / 14));
  for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    let maxA = 0;
    const x0=Math.max(0,px-radius), x1=Math.min(W-1,px+radius);
    const y0=Math.max(0,py-radius), y1=Math.min(H-1,py+radius);
    for (let ny=y0; ny<=y1; ny+=step) for (let nx=x0; nx<=x1; nx+=step) {
      const d = Math.sqrt((nx-px)**2+(ny-py)**2);
      if (d<=radius) { const a=base[ny*W+nx]*Math.max(0,1-d/(radius*1.25)); if(a>maxA) maxA=a; }
    }
    result[py*W+px] = maxA;
  }
  return result;
}

export interface BBox {
  x1pct: number; y1pct: number; x2pct: number; y2pct: number;
  softness?: number; priority?: number;
}

export function applyBBoxesToAlphaMap(alphaMap: Float32Array, bboxes: BBox[], W: number, H: number): Float32Array {
  for (const obj of bboxes) {
    const x1=Math.round(obj.x1pct/100*W), y1=Math.round(obj.y1pct/100*H);
    const x2=Math.round(obj.x2pct/100*W), y2=Math.round(obj.y2pct/100*H);
    const soft = Math.max(4,(obj.softness||10)*3);
    for (let py=y1; py<=y2; py++) for (let px=x1; px<=x2; px++) {
      const minD = Math.min(py-y1,y2-py,px-x1,x2-px);
      const a = Math.min(1,minD/soft)*(obj.priority||1.0);
      if (a>alphaMap[py*W+px]) alphaMap[py*W+px]=a;
    }
  }
  return alphaMap;
}
