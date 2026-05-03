const ALLOWED_ORIGINS = [
  'https://parallax-studio-mu.vercel.app',
  'http://localhost:3000',
];

function setCors(req, res) {
  const origin = req.headers.origin || '';
  const allowed = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];
  res.setHeader('Access-Control-Allow-Origin', allowed);
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Vary', 'Origin');
}

const SAM2_VERSION = 'fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83';

// Unified prediction runner.
// version   → POST /v1/predictions         { version, input }
// modelPath → POST /v1/models/{path}/predictions  { input }
async function replicateRun({ version, modelPath, input }) {
  const key = process.env.REPLICATE_KEY;
  const url  = version
    ? 'https://api.replicate.com/v1/predictions'
    : `https://api.replicate.com/v1/models/${modelPath}/predictions`;
  const requestBody = version ? { version, input } : { input };

  console.log('[replicate] endpoint:', url);
  console.log('[replicate] input keys:', Object.keys(input));

  let pred;
  for (let attempt = 0; attempt < 4; attempt++) {
    console.log(`[replicate] POST attempt ${attempt + 1}`);

    const res = await fetch(url, {
      method: 'POST',
      headers: {
        Authorization: `Token ${key}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    const rawText = await res.text();
    console.log(`[replicate] create → ${res.status}: ${rawText.slice(0, 600)}`);

    if (res.status === 429) {
      let retryAfter = 15;
      try { retryAfter = JSON.parse(rawText).retry_after ?? 15; } catch {}
      if (attempt === 3) throw new Error(`Replicate 429 — adicione créditos (retry_after: ${retryAfter}s)`);
      await new Promise(r => setTimeout(r, (retryAfter + 1) * 1000));
      continue;
    }

    if (!res.ok) {
      // Extract the most useful error detail from Replicate's response
      let detail = rawText;
      try {
        const parsed = JSON.parse(rawText);
        detail = parsed.detail || parsed.error || rawText;
      } catch {}
      throw new Error(`Replicate ${res.status}: ${detail}`);
    }

    pred = JSON.parse(rawText);
    console.log('[replicate] created id:', pred.id, 'status:', pred.status);
    break;
  }

  const deadline = Date.now() + 270_000;
  let polls = 0;
  while ((pred.status === 'starting' || pred.status === 'processing') && Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 3000));
    polls++;
    const poll = await fetch(`https://api.replicate.com/v1/predictions/${pred.id}`, {
      headers: { Authorization: `Token ${key}` },
    });
    const pollText = await poll.text();
    console.log(`[replicate] poll #${polls} → ${poll.status}: ${pollText.slice(0, 300)}`);
    if (!poll.ok) break;
    pred = JSON.parse(pollText);
  }

  if (pred.status === 'failed') throw new Error(`Replicate failed: ${pred.error}`);
  if (pred.status !== 'succeeded') throw new Error(`Replicate timed out (${polls} polls, status: ${pred.status})`);

  console.log('[replicate] output:', JSON.stringify(pred.output).slice(0, 200));
  return pred.output;
}

async function urlToBase64(url) {
  console.log('[replicate] fetching output URL:', url?.slice(0, 100));
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Output fetch failed: ${res.status}`);
  return Buffer.from(await res.arrayBuffer()).toString('base64');
}

module.exports = async function handler(req, res) {
  setCors(req, res);
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  console.log('[replicate] type:', req.body?.type, '| key set:', !!process.env.REPLICATE_KEY);

  if (!process.env.REPLICATE_KEY)
    return res.status(500).json({ error: 'REPLICATE_KEY not configured' });

  const { type } = req.body || {};

  try {
    // ── SAM2 Segmentation ──────────────────────────────────────────────────────
    // Image arrives as JPEG base64 (≤512px), points as [[x,y],...] arrays
    if (type === 'sam2') {
      const { imageBase64, pointCoords, pointLabels } = req.body;
      if (!imageBase64 || !Array.isArray(pointCoords) || !pointCoords.length)
        return res.status(400).json({ error: 'imageBase64 and pointCoords required' });

      console.log('[replicate] sam2: image len =', imageBase64.length, '| points =', JSON.stringify(pointCoords));

      // meta/sam-2 expects input_points and input_labels as JSON *strings*,
      // with float coordinates: [[[x1.0, y1.0], [x2.0, y2.0]]]  (triple-nested for batch)
      const inputPoints  = JSON.stringify([pointCoords.map(([x, y]) => [+x, +y])]);
      const inputLabels  = JSON.stringify([pointLabels.map(Number)]);

      const output = await replicateRun({
        version: SAM2_VERSION,
        input: {
          image:            `data:image/jpeg;base64,${imageBase64}`,
          input_points:     inputPoints,
          input_labels:     inputLabels,
          multimask_output: false,
        },
      });

      // output may be an array of mask URLs or a single URL
      const maskUrl = Array.isArray(output) ? output.find(u => typeof u === 'string') ?? output[0] : output;
      if (!maskUrl || typeof maskUrl !== 'string') throw new Error(`SAM2 output unexpected: ${JSON.stringify(output)}`);
      return res.status(200).json({ maskBase64: await urlToBase64(maskUrl) });
    }

    // ── Flux Fill Pro Inpainting ───────────────────────────────────────────────
    if (type === 'inpaint') {
      const { imageBase64, maskBase64: maskB64, prompt } = req.body;
      if (!imageBase64 || !maskB64)
        return res.status(400).json({ error: 'imageBase64 and maskBase64 required' });

      console.log('[replicate] inpaint: image len =', imageBase64.length, '| mask len =', maskB64.length);

      const output = await replicateRun({
        modelPath: 'black-forest-labs/flux-fill-pro',
        input: {
          image:    `data:image/png;base64,${imageBase64}`,
          mask:     `data:image/png;base64,${maskB64}`,
          prompt:   prompt || 'seamless natural background, photorealistic, highly detailed',
          steps:    28,
          guidance: 30,
        },
      });

      const imgUrl = Array.isArray(output) ? output[0] : output;
      return res.status(200).json({ imageBase64: await urlToBase64(imgUrl) });
    }

    return res.status(400).json({ error: `Unknown type: ${type}` });

  } catch (err) {
    console.error('[replicate] ERROR:', err.message);
    return res.status(500).json({ error: err.message });
  }
};
