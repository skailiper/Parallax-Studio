jest.mock('../lib/supabase', () => ({
  createProject: jest.fn(),
  updateProjectStatus: jest.fn(),
  saveProjectLayers: jest.fn(),
  logProcessingEvent: jest.fn(),
}));

import { LAYER_COLORS } from './usePipeline';

test('LAYER_COLORS has exactly 8 entries', () => {
  expect(LAYER_COLORS).toHaveLength(8);
});

test('every color has a valid hex, name, and rgb triple', () => {
  for (const c of LAYER_COLORS) {
    expect(c.hex).toMatch(/^#[0-9A-Fa-f]{6}$/);
    expect(typeof c.name).toBe('string');
    expect(c.rgb).toHaveLength(3);
    c.rgb.forEach(v => {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(255);
    });
  }
});

test('hex matches rgb values', () => {
  for (const c of LAYER_COLORS) {
    const [r, g, b] = c.rgb;
    const expected = '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0').toUpperCase()).join('');
    expect(c.hex.toUpperCase()).toBe(expected);
  }
});
