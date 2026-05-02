import { withRetry } from './retry';

test('resolves immediately on first success', async () => {
  const fn = jest.fn().mockResolvedValue('ok');
  await expect(withRetry(fn)).resolves.toBe('ok');
  expect(fn).toHaveBeenCalledTimes(1);
});

test('retries on failure and eventually succeeds', async () => {
  let calls = 0;
  const fn = jest.fn().mockImplementation(async () => {
    calls++;
    if (calls < 3) throw new Error('transient');
    return 'done';
  });
  await expect(withRetry(fn, { attempts: 3, baseDelayMs: 0 })).resolves.toBe('done');
  expect(fn).toHaveBeenCalledTimes(3);
});

test('throws after exhausting all attempts', async () => {
  const fn = jest.fn().mockRejectedValue(new Error('permanent'));
  await expect(withRetry(fn, { attempts: 3, baseDelayMs: 0 })).rejects.toThrow('permanent');
  expect(fn).toHaveBeenCalledTimes(3);
});

test('calls onRetry with error and attempt number', async () => {
  const onRetry = jest.fn();
  const fn = jest.fn()
    .mockRejectedValueOnce(new Error('fail1'))
    .mockResolvedValue('ok');
  await withRetry(fn, { attempts: 3, baseDelayMs: 0, onRetry });
  expect(onRetry).toHaveBeenCalledTimes(1);
  expect(onRetry).toHaveBeenCalledWith(expect.any(Error), 1);
});
