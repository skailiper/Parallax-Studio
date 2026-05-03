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

const SDXL_VERSION = '95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3';

// POST to /v1/predictions with explicit version hash, then poll until done.
async function replicateRun(version, input) {
  const key = process.env.REPLICATE_KEY;

  console.log('[replicate] key present:', !!key, '| prefix:', key?.slice(0, 8));
  console.log('[replicate] version:', version);
  console.log('[replicate] input keys:', Object.keys(input));

  // Create prediction — retry up to 4× on 429
  let pred;
  for (let attempt = 0; attempt < 4; attempt++) {
    console.log(`[replicate] POST /v1/predictions attempt ${attempt + 1}`);

    const res = await fetch('https://api.replicate.com/v1/predictions', {
      method: 'POST',
      headers: {
        Authorization: `Token ${key}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ version, input }),
    });

    const rawText = await res.text();
    console.log(`[replicate] create → ${res.status}: ${rawText.slice(0, 400)}`);

    if (res.status === 429) {
      let retryAfter = 15;
      try { retryAfter = JSON.parse(rawText).retry_after ?? 15; } catch {}
      if (attempt === 3) throw new Error(`Replicate 429: rate-limited — adicione créditos em replicate.com/account/billing (retry_after: ${retryAfter}s)`);
      console.log(`[replicate] 429 – aguardando ${retryAfter + 1}s…`);
      await new Promise(r => setTimeout(r, (retryAfter + 1) * 1000));
      continue;
    }

    if (!res.ok) throw new Error(`Replicate create HTTP ${res.status}: ${rawText}`);

    pred = JSON.parse(rawText);
    console.log('[replicate] created id:', pred.id, 'status:', pred.status);
    break;
  }

  // Poll until succeeded / failed / timeout
  const deadline = Date.now() + 270_000;
  let polls = 0;
  while ((pred.status === 'starting' || pred.status === 'processing') && Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 3000));
    polls++;
    const poll = await fetch(`https://api.replicate.com/v1/predictions/${pred.id}`, {
      headers: { Authorization: `Token ${key}` },
    });
    const pollText = await poll.text();
    console.log(`[replicate] poll #${polls} → ${poll.status}: ${pollText.slice(0, 200)}`);
    if (!poll.ok) break;
    pred = JSON.parse(pollText);
  }

  if (pred.status === 'failed') throw new Error(`Replicate failed: ${pred.error}`);
  if (pred.status !== 'succeeded') throw new Error(`Replicate timed out after ${polls} polls (status: ${pred.status})`);

  console.log('[replicate] succeeded. output:', JSON.stringify(pred.output).slice(0, 100));
  return pred.output;
}

async function urlToBase64(url) {
  console.log('[replicate] fetching output:', url?.slice(0, 80));
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch output URL failed: ${res.status}`);
  return Buffer.from(await res.arrayBuffer()).toString('base64');
}

module.exports = async function handler(req, res) {
  setCors(req, res);
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  console.log('[replicate] handler type:', req.body?.type, '| key set:', !!process.env.REPLICATE_KEY);

  if (!process.env.REPLICATE_KEY) {
    console.error('[replicate] REPLICATE_KEY missing from environment');
    return res.status(500).json({ error: 'REPLICATE_KEY not configured' });
  }

  const { type } = req.body || {};

  try {
    // ── SDXL Inpainting ────────────────────────────────────────────────────────
    if (type === 'inpaint') {
      const { imageBase64, maskBase64: maskB64, prompt } = req.body;
      if (!imageBase64 || !maskB64)
        return res.status(400).json({ error: 'imageBase64 and maskBase64 required' });

      console.log('[replicate] inpaint: image len =', imageBase64.length, 'mask len =', maskB64.length);

      const output = await replicateRun(SDXL_VERSION, {
        image:              `data:image/jpeg;base64,${imageBase64}`,
        mask:               `data:image/png;base64,${maskB64}`,
        prompt:             prompt || 'seamless natural background, photorealistic, highly detailed',
        num_inference_steps: 25,
      });

      const imgUrl = Array.isArray(output) ? output[0] : output;
      return res.status(200).json({ imageBase64: await urlToBase64(imgUrl) });
    }

    console.error('[replicate] unknown type:', type);
    return res.status(400).json({ error: `Unknown type: ${type}` });

  } catch (err) {
    console.error('[replicate] ERROR:', JSON.stringify(err, Object.getOwnPropertyNames(err)));
    return res.status(500).json({ error: err.message });
  }
};
