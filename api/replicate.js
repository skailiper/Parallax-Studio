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

// POST to /v1/models/{owner}/{name}/predictions then poll /v1/predictions/{id}
async function replicateRun(model, input) {
  const key = process.env.REPLICATE_KEY;

  let res = await fetch(`https://api.replicate.com/v1/models/${model}/predictions`, {
    method: 'POST',
    headers: {
      Authorization: `Token ${key}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ input }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Replicate create (${model}): ${res.status} — ${text}`);
  }

  let pred = await res.json();

  // Poll /v1/predictions/{id} until terminal state
  const deadline = Date.now() + 270_000;
  while ((pred.status === 'starting' || pred.status === 'processing') && Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 2500));
    res = await fetch(`https://api.replicate.com/v1/predictions/${pred.id}`, {
      headers: { Authorization: `Token ${key}` },
    });
    if (!res.ok) break;
    pred = await res.json();
  }

  if (pred.status === 'failed') throw new Error(pred.error || 'Replicate prediction failed');
  if (pred.status !== 'succeeded') throw new Error(`Replicate timed out (status: ${pred.status})`);
  return pred.output;
}

async function urlToBase64(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch result: ${res.status}`);
  return Buffer.from(await res.arrayBuffer()).toString('base64');
}

module.exports = async function handler(req, res) {
  setCors(req, res);
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  if (!process.env.REPLICATE_KEY) return res.status(500).json({ error: 'REPLICATE_KEY not configured' });

  const { type } = req.body || {};

  try {
    // ── SAM2 segmentation ──────────────────────────────────────────────────────
    if (type === 'segment') {
      const { imageBase64, points, pointLabels } = req.body;
      if (!imageBase64 || !Array.isArray(points) || points.length === 0)
        return res.status(400).json({ error: 'imageBase64 and points required' });

      const output = await replicateRun('facebook/sam-2', {
        image: `data:image/jpeg;base64,${imageBase64}`,
        points: JSON.stringify(points),
        point_labels: JSON.stringify(pointLabels ?? points.map(() => 1)),
        multimask_output: false,
      });

      const maskUrl = Array.isArray(output) ? output[0] : output;
      return res.status(200).json({ maskBase64: await urlToBase64(maskUrl) });
    }

    // ── SDXL inpainting ────────────────────────────────────────────────────────
    if (type === 'inpaint') {
      const { imageBase64, maskBase64: maskB64, prompt, negativePrompt, strength = 0.60, steps = 30 } = req.body;
      if (!imageBase64 || !maskB64)
        return res.status(400).json({ error: 'imageBase64 and maskBase64 required' });

      // 'stability-ai' is the model owner's handle on Replicate — not a product dependency
      const output = await replicateRun('stability-ai/stable-diffusion-inpainting', {
        image:      `data:image/jpeg;base64,${imageBase64}`,
        mask_image: `data:image/png;base64,${maskB64}`,
        prompt: prompt || 'seamless natural background, photorealistic, highly detailed',
        negative_prompt: negativePrompt || 'blurry, artifacts, low quality, watermark, text',
        num_inference_steps: Math.round(steps),
        guidance_scale: 7.5,
        strength: Math.min(0.85, Math.max(0.3, strength)),
      });

      const imgUrl = Array.isArray(output) ? output[0] : output;
      return res.status(200).json({ imageBase64: await urlToBase64(imgUrl) });
    }

    return res.status(400).json({ error: `Unknown type: ${type}` });
  } catch (err) {
    console.error('[replicate]', err.message);
    return res.status(500).json({ error: err.message });
  }
};
