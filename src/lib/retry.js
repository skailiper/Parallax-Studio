export async function withRetry(fn, { attempts = 3, baseDelayMs = 800, onRetry } = {}) {
  let lastErr = new Error('Unknown error');
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err instanceof Error ? err : new Error(String(err));
      if (i < attempts - 1) {
        onRetry?.(lastErr, i + 1);
        await new Promise(r => setTimeout(r, baseDelayMs * 2 ** i));
      }
    }
  }
  throw lastErr;
}
