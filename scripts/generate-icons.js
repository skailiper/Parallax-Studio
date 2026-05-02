'use strict';
const zlib = require('zlib');
const fs   = require('fs');
const path = require('path');

function crc32(buf) {
  let crc = 0xFFFFFFFF;
  for (const byte of buf) {
    crc ^= byte;
    for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ (crc & 1 ? 0xEDB88320 : 0);
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

function chunk(type, data) {
  const t  = Buffer.from(type, 'ascii');
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const crcBuf = Buffer.alloc(4); crcBuf.writeUInt32BE(crc32(Buffer.concat([t, data])));
  return Buffer.concat([len, t, data, crcBuf]);
}

function makePng(size) {
  const W = size, H = size;
  const rgba = Buffer.alloc(W * H * 4);
  const cx = W / 2, cy = H / 2;

  // Background: #07070a
  for (let i = 0; i < W * H; i++) {
    rgba[i * 4]     = 7;
    rgba[i * 4 + 1] = 7;
    rgba[i * 4 + 2] = 10;
    rgba[i * 4 + 3] = 255;
  }

  // Draw 4 rounded squares (parallax logo) in teal #5eead4 = (94,234,212)
  const pad  = Math.round(size * 0.07);
  const half = Math.round(size * 0.39);
  const gap  = Math.round(size * 0.06);
  const rx   = Math.round(size * 0.07); // corner radius
  const blocks = [
    { x: pad,          y: pad,          alpha: 230 },
    { x: pad+half+gap, y: pad,          alpha: 140 },
    { x: pad,          y: pad+half+gap, alpha: 89  },
    { x: pad+half+gap, y: pad+half+gap, alpha: 38  },
  ];

  function drawRect(x0, y0, w, h, r, a) {
    for (let py = y0; py < y0 + h; py++) {
      for (let px = x0; px < x0 + w; px++) {
        if (px < 0 || px >= W || py < 0 || py >= H) continue;
        // round corners: check distance to nearest corner center
        const nearX = Math.max(x0 + r, Math.min(x0 + w - r, px));
        const nearY = Math.max(y0 + r, Math.min(y0 + h - r, py));
        const dx = px - nearX, dy = py - nearY;
        if (dx * dx + dy * dy > r * r) continue;
        const idx = (py * W + px) * 4;
        rgba[idx]     = Math.round(94  * a / 255 + 7  * (1 - a / 255));
        rgba[idx + 1] = Math.round(234 * a / 255 + 7  * (1 - a / 255));
        rgba[idx + 2] = Math.round(212 * a / 255 + 10 * (1 - a / 255));
        rgba[idx + 3] = 255;
      }
    }
  }

  for (const b of blocks) drawRect(b.x, b.y, half, half, rx, b.alpha);

  // Build raw PNG scanlines (filter byte 0 per row)
  const raw = Buffer.alloc(H * (1 + W * 4));
  for (let y = 0; y < H; y++) {
    raw[y * (1 + W * 4)] = 0;
    for (let x = 0; x < W; x++) {
      const s = (y * W + x) * 4;
      const d = y * (1 + W * 4) + 1 + x * 4;
      raw[d] = rgba[s]; raw[d+1] = rgba[s+1]; raw[d+2] = rgba[s+2]; raw[d+3] = rgba[s+3];
    }
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4);
  ihdr[8]=8; ihdr[9]=6; ihdr[10]=0; ihdr[11]=0; ihdr[12]=0; // 8-bit RGBA

  return Buffer.concat([
    Buffer.from([137,80,78,71,13,10,26,10]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const outDir = path.join(__dirname, '..', 'public');
fs.writeFileSync(path.join(outDir, 'icon-192.png'), makePng(192));
fs.writeFileSync(path.join(outDir, 'icon-512.png'), makePng(512));
console.log('✅ icon-192.png e icon-512.png gerados em public/');
