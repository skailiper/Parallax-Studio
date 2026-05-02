import { createClient } from '@supabase/supabase-js';

const supabaseUrl  = process.env.REACT_APP_SUPABASE_URL  ?? '';
const supabaseAnon = process.env.REACT_APP_SUPABASE_ANON_KEY ?? '';

export const supabase = createClient(supabaseUrl, supabaseAnon);

export async function createProject({ sessionId, numLayers, imageFilename, imageSizeBytes }) {
  const { data, error } = await supabase
    .from('projects')
    .insert({ session_id: sessionId, num_layers: numLayers, image_filename: imageFilename, image_size_bytes: imageSizeBytes, status: 'painting' })
    .select().single();
  if (error) throw error;
  return data;
}

export async function updateProjectStatus(id, status, meta = {}) {
  const { error } = await supabase.from('projects').update({ status, updated_at: new Date().toISOString(), ...meta }).eq('id', id);
  if (error) throw error;
}

export async function saveProjectLayers(projectId, layers) {
  const rows = layers.map(l => ({
    project_id: projectId, layer_index: l.index, elements: l.elements,
    has_inpaint: l.hasInpaint, cutout_data_url: l.cutoutDataURL, inpainted_data_url: l.inpaintedDataURL ?? null,
  }));
  const { error } = await supabase.from('layers').insert(rows);
  if (error) throw error;
}

export async function logProcessingEvent(projectId, event, details = {}) {
  await supabase.from('processing_logs').insert({ project_id: projectId, event, details });
}

export async function trackUsage({ sessionId, action, meta = {} }) {
  await supabase.from('usage_events').insert({ session_id: sessionId, action, meta });
}
