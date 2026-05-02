import { getSessionId } from './session';

const STORAGE_KEY = 'parallax_session_id';

beforeEach(() => localStorage.clear());

test('creates a session id on first call', () => {
  const id = getSessionId();
  expect(id).toMatch(/^sess_\d+_[a-z0-9]+$/);
  expect(localStorage.getItem(STORAGE_KEY)).toBe(id);
});

test('returns the same id on subsequent calls', () => {
  const first  = getSessionId();
  const second = getSessionId();
  expect(first).toBe(second);
});

test('restores id stored in localStorage', () => {
  localStorage.setItem(STORAGE_KEY, 'sess_existing_id');
  expect(getSessionId()).toBe('sess_existing_id');
});
