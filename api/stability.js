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

module.exports = async function handler(req, res) {
  setCors(req, res);
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  if (!process.env.STABILITY_KEY) return res.status(500).json({ error: 'STABILITY_KEY not configured' });

  try {
    const { imageBase64, maskBase64, prompt, negativePrompt, steps = 40 } = req.body;
    const imageBuffer = Buffer.from(imageBase64, 'base64');
    const maskBuffer  = Buffer.from(maskBase64,  'base64');
    const boundary = '----ParallaxBoundary' + Math.random().toString(36).slice(2);

    function part(name, value, filename, contentType) {
      let header = `--${boundary}\r\nContent-Disposition: form-data; name="${name}"`;
      if (filename)    header += `; filename="${filename}"`;
      if (contentType) header += `\r\nContent-Type: ${contentType}`;
      header += '\r\n\r\n';
      return Buffer.concat([
        Buffer.from(header),
        typeof value === 'string' ? Buffer.from(value) : value,
        Buffer.from('\r\n'),
      ]);
    }

    const body = Buffer.concat([
      part('image',           imageBuffer, 'image.png', 'image/png'),
      part('mask',            maskBuffer,  'mask.png',  'image/png'),
      part('prompt',          prompt || 'seamless natural background, photorealistic, highly detailed'),
      part('negative_prompt', negativePrompt || 'blurry, artifacts, low quality, watermark, text'),
      part('output_format',   'png'),
      part('steps',           String(steps)),
      part('strength',        '0.99'),
      Buffer.from(`--${boundary}--\r\n`),
    ]);

    const response = await fetch('https://api.stability.ai/v2beta/stable-image/edit/inpaint', {
      method: 'POST',
      headers: {
        Authorization:    `Bearer ${process.env.STABILITY_KEY}`,
        Accept:           'image/*',
        'Content-Type':   `multipart/form-data; boundary=${boundary}`,
        'Content-Length': String(body.length),
      },
      body,
    });

    if (!response.ok) {
      const errText = await response.text();
      return res.status(response.status).json({ error: `Stability AI: ${response.status} — ${errText}` });
    }

    const arrayBuffer = await response.arrayBuffer();
    const base64 = Buffer.from(arrayBuffer).toString('base64');
    return res.status(200).json({ imageBase64: base64 });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
};
