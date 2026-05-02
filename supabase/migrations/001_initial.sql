CREATE TABLE IF NOT EXISTS projects (
  id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id        TEXT        NOT NULL,
  num_layers        INT         NOT NULL DEFAULT 3,
  image_filename    TEXT,
  image_size_bytes  BIGINT,
  status            TEXT        NOT NULL DEFAULT 'painting'
                                CHECK (status IN ('painting','processing','inpainting','done','error')),
  layers_count      INT,
  error_message     TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS layers (
  id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id          UUID        NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  layer_index         INT         NOT NULL,
  elements            TEXT[]      DEFAULT '{}',
  has_inpaint         BOOLEAN     DEFAULT FALSE,
  cutout_data_url     TEXT,
  inpainted_data_url  TEXT,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS processing_logs (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id  UUID        REFERENCES projects(id) ON DELETE CASCADE,
  event       TEXT        NOT NULL,
  details     JSONB       DEFAULT '{}',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS usage_events (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id  TEXT        NOT NULL,
  action      TEXT        NOT NULL,
  meta        JSONB       DEFAULT '{}',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_projects_session ON projects (session_id);
CREATE INDEX IF NOT EXISTS idx_projects_status  ON projects (status);
CREATE INDEX IF NOT EXISTS idx_layers_project   ON layers   (project_id);
CREATE INDEX IF NOT EXISTS idx_usage_session    ON usage_events (session_id);
CREATE INDEX IF NOT EXISTS idx_usage_action     ON usage_events (action);

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_projects_updated_at
  BEFORE UPDATE ON projects
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

ALTER TABLE projects        ENABLE ROW LEVEL SECURITY;
ALTER TABLE layers          ENABLE ROW LEVEL SECURITY;
ALTER TABLE processing_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_events    ENABLE ROW LEVEL SECURITY;

CREATE POLICY "insert_projects"   ON projects        FOR INSERT WITH CHECK (true);
CREATE POLICY "insert_layers"     ON layers          FOR INSERT WITH CHECK (true);
CREATE POLICY "insert_logs"       ON processing_logs FOR INSERT WITH CHECK (true);
CREATE POLICY "insert_usage"      ON usage_events    FOR INSERT WITH CHECK (true);
CREATE POLICY "read_own_projects" ON projects        FOR SELECT USING (true);
CREATE POLICY "update_own_project"ON projects        FOR UPDATE USING (true);
CREATE POLICY "read_own_layers"   ON layers          FOR SELECT USING (true);
