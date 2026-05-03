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

async function replicateRun(model, input) {
  const key = process.env.REPLICATE_KEY;

  console.log('[replicate] REPLICATE_KEY present:', !!key);
  console.log('[replicate] REPLICATE_KEY prefix:', key ? key.slice(0, 8) + '...' : 'MISSING');
  console.log('[replicate] model:', model);
  console.log('[replicate] input keys:', Object.keys(input));

  const createUrl = `https://api.replicate.com/v1/predictions`;
  const createBody = {
    version: undefined, // will use model field instead
    model,
    input,
  };

  // Use /v1/predictions with model field (no version hash needed for official models)
  // Authorization must be: Token <key>  (not Bearer)
  let pred;
  for (let attempt = 0; attempt < 4; attempt++) {
    console.log(`[replicate] POST ${createUrl} (attempt ${attempt + 1})`);

    let rawText;
    let httpStatus;
    try {
      const res = await fetch(createUrl, {
        method: 'POST',
        headers: {
          Authorization: `Token ${key}`,
          'Content-Type': 'application/json',
          Prefer: 'wait=5',
        },
        body: JSON.stringify({ model, input }),
      });

      httpStatus = res.status;
      rawText = await res.text();
      console.log(`[replicate] create response ${httpStatus}:`, rawText.slice(0, 500));

      if (res.status === 429) {
        let retryAfter = 15;
        try { retryAfter = JSON.parse(rawText).retry_after ?? 15; } catch {}
        console.log(`[replicate] 429 rate-limited, retry_after=${retryAfter}s (attempt ${attempt + 1}/4)`);
        if (attempt === 3) throw new Error(`Replicate: taxa excedida — adicione créditos em replicate.com/account/billing (retry_after: ${retryAfter}s)`);
        await new Promise(r => setTimeout(r, (retryAfter + 1) * 1000));
        continue;
      }

      if (!res.ok) {
        throw new Error(`Replicate create HTTP ${httpStatus}: ${rawText}`);
      }

      pred = JSON.parse(rawText);
      console.log('[replicate] prediction created, id:', pred.id, 'status:', pred.status);
      break;
    } catch (err) {
      // Re-throw if not a retryable network error
      if (httpStatus && httpStatus !== 429) throw err;
      if (attempt === 3) throw err;
      console.error('[replicate] network error on create:', JSON.stringify(err, Object.getOwnPropertyNames(err)));
      await new Promise(r => setTimeout(r, 5000));
    }
  }

  // If already succeeded (Prefer: wait=5 may have resolved it)
  if (pred.status === 'succeeded') {
    console.log('[replicate] prediction already succeeded');
    return pred.output;
  }
  if (pred.status === 'failed') {
    console.error('[replicate] prediction failed immediately:', pred.error);
    throw new Error(pred.error || 'Replicate prediction failed');
  }

  // Poll /v1/predictions/{id} until terminal state
  const pollUrl = `https://api.replicate.com/v1/predictions/${pred.id}`;
  const deadline = Date.now() + 270_000;
  let pollCount = 0;

  while ((pred.status === 'starting' || pred.status === 'processing') && Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 2500));
    pollCount++;
    console.log(`[replicate] poll #${pollCount} ${pollUrl} status=${pred.status}`);

    const poll = await fetch(pollUrl, {
      headers: { Authorization: `Token ${key}` },
    });

    if (!poll.ok) {
      const pollErr = await poll.text();
      console.error('[replicate] poll error:', poll.status, pollErr);
      break;
    }

    pred = await poll.json();
    console.log(`[replicate] poll result: status=${pred.status}`);
  }

  if (pred.status === 'failed') {
    console.error('[replicate] prediction failed after polling:', pred.error);
    throw new Error(pred.error || 'Replicate prediction failed');
  }
  if (pred.status !== 'succeeded') {
    console.error('[replicate] prediction did not succeed:', JSON.stringify(pred));
    throw new Error(`Replicate timed out (status: ${pred.status}, polls: ${pollCount})`);
  }

  console.log('[replicate] succeeded, output type:', typeof pred.output, Array.isArray(pred.output) ? `array[${pred.output.length}]` : '');
  return pred.output;
}

async function urlToBase64(url) {
  console.log('[replicate] fetching output URL:', url?.slice(0, 80));
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch result URL: ${res.status} ${url?.slice(0, 80)}`);
  return Buffer.from(await res.arrayBuffer()).toString('base64');
}

module.exports = async function handler(req, res) {
  setCors(req, res);
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  console.log('[replicate] handler called, type:', req.body?.type);
  console.log('[replicate] REPLICATE_KEY configured:', !!process.env.REPLICATE_KEY);

  if (!process.env.REPLICATE_KEY) {
    console.error('[replicate] REPLICATE_KEY is not set in environment');
    return res.status(500).json({ error: 'REPLICATE_KEY not configured' });
  }

  const { type } = req.body || {};

  try {
    // ── SAM2 segmentation ──────────────────────────────────────────────────────
    if (type === 'segment') {
      const { imageBase64, points, pointLabels } = req.body;
      if (!imageBase64 || !Array.isArray(points) || points.length === 0)
        return res.status(400).json({ error: 'imageBase64 and points required' });

      console.log('[replicate] segment: points count =', points.length);

      const output = await replicateRun('facebook/sam-2', {
        image:        `data:image/jpeg;base64,${imageBase64}`,
        points:       JSON.stringify(points),
        point_labels: JSON.stringify(pointLabels ?? points.map(() => 1)),
        multimask_output: false,
      });

      const maskUrl = Array.isArray(output) ? output[0] : output;
      return res.status(200).json({ maskBase64: await urlToBase64(maskUrl) });
    }

    // ── SDXL Inpainting ────────────────────────────────────────────────────────
    if (type === 'inpaint') {
      const { imageBase64, maskBase64: maskB64, prompt, negativePrompt, strength = 0.60, steps = 30 } = req.body;
      if (!imageBase64 || !maskB64)
        return res.status(400).json({ error: 'imageBase64 and maskBase64 required' });

      console.log('[replicate] inpaint: imageBase64 length =', imageBase64.length, 'maskBase64 length =', maskB64.length);

      // stability-ai is the model owner's handle on Replicate
      const output = await replicateRun('stability-ai/stable-diffusion-inpainting', {
        image:           `data:image/jpeg;base64,${imageBase64}`,
        mask_image:      `data:image/png;base64,${maskB64}`,
        prompt:          prompt || 'seamless natural background, photorealistic, highly detailed',
        negative_prompt: negativePrompt || 'blurry, artifacts, low quality, watermark, text',
        num_inference_steps: Math.round(steps),
        guidance_scale:  7.5,
        strength:        Math.min(0.85, Math.max(0.3, strength)),
      });

      const imgUrl = Array.isArray(output) ? output[0] : output;
      return res.status(200).json({ imageBase64: await urlToBase64(imgUrl) });
    }

    console.error('[replicate] unknown type:', type);
    return res.status(400).json({ error: `Unknown type: ${type}` });

  } catch (err) {
    console.error('[replicate] FATAL ERROR:', JSON.stringify(err, Object.getOwnPropertyNames(err)));
    console.error('[replicate] message:', err.message);
    console.error('[replicate] stack:', err.stack);
    return res.status(500).json({ error: err.message });
  }
};
