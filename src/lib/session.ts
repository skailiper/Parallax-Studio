export function getSessionId(): string {
  const key = 'parallax_session_id';
  let id = localStorage.getItem(key);
  if (!id) {
    id = 'sess_' + Date.now() + '_' + Math.random().toString(36).slice(2, 10);
    localStorage.setItem(key, id);
  }
  return id;
}
