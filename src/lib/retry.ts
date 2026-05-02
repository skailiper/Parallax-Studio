interface RetryOptions {
  attempts?: number;
  baseDelayMs?: number;
  onRetry?: (err: Error, attempt: number) => void;
}

export async function withRetry<T>(fn: () => Promise<T>, options: RetryOptions = {}): Promise<T> {
  const { attempts = 3, baseDelayMs = 800, onRetry } = options;
  let lastErr: Error = new Error('Unknown error');
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err instanceof Error ? err : new Error(String(err));
      if (i < attempts - 1) {
        onRetry?.(lastErr, i + 1);
        await new Promise<void>(r => setTimeout(r, baseDelayMs * 2 ** i));
      }
    }
  }
  throw lastErr;
}
