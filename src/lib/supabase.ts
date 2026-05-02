import { createClient } from '@supabase/supabase-js';

const supabaseUrl  = process.env.REACT_APP_SUPABASE_URL  ?? '';
const supabaseAnon = process.env.REACT_APP_SUPABASE_ANON_KEY ?? '';

export const supabase = createClient(supabaseUrl, supabaseAnon);

export interface Project {
  id: string;
  session_id: string;
  num_layers: number;
  image_filename: string;
  image_size_bytes: number;
  status: string;
}

export async function createProject(params: {
  sessionId: string;
  numLayers: number;
  imageFilename: string;
  imageSizeBytes: number;
}): Promise<Project> {
  const { data, error } = await supabase
    .from('projects')
    .insert({
      session_id: params.sessionId,
      num_layers: params.numLayers,
      image_filename: params.imageFilename,
      image_size_bytes: params.imageSizeBytes,
      status: 'painting',
    })
    .select()
    .single();
  if (error) throw error;
  return data as Project;
}

export async function updateProjectStatus(id: string, status: string, meta: Record<string, unknown> = {}): Promise<void> {
  const { error } = await supabase
    .from('projects')
    .update({ status, updated_at: new Date().toISOString(), ...meta })
    .eq('id', id);
  if (error) throw error;
}

export interface LayerResult {
  index: number;
  elements: string[];
  hasInpaint: boolean;
  cutoutDataURL: string;
  inpaintedDataURL: string | null;
}

export async function saveProjectLayers(projectId: string, layers: LayerResult[]): Promise<void> {
  const rows = layers.map(l => ({
    project_id: projectId,
    layer_index: l.index,
    elements: l.elements,
    has_inpaint: l.hasInpaint,
    cutout_data_url: l.cutoutDataURL,
    inpainted_data_url: l.inpaintedDataURL ?? null,
  }));
  const { error } = await supabase.from('layers').insert(rows);
  if (error) throw error;
}

export async function logProcessingEvent(projectId: string, event: string, details: Record<string, unknown> = {}): Promise<void> {
  await supabase.from('processing_logs').insert({ project_id: projectId, event, details });
}

export async function trackUsage(params: { sessionId: string; action: string; meta?: Record<string, unknown> }): Promise<void> {
  await supabase.from('usage_events').insert({ session_id: params.sessionId, action: params.action, meta: params.meta ?? {} });
}
