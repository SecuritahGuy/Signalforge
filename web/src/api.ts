const API_BASE = "";

export type ScriptArg = {
  name: string;
  default?: unknown;
  type?: string;
  choices?: string[];
  required?: boolean;
};

export type ScriptMeta = {
  name: string;
  description: string;
  args: ScriptArg[];
};

export type ScriptCategory = {
  category: string;
  label: string;
  scripts: ScriptMeta[];
};

export type RunState = {
  run_id: string;
  script_name: string;
  status: "pending" | "running" | "completed" | "failed";
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  output: string[];
};

export type PaperSummary = {
  initial_capital?: number;
  cash?: number;
  equity?: number;
  realized_pnl?: number;
  unrealized_pnl?: number;
  open_positions?: number;
  closed_positions?: number;
  planned_orders?: number;
  skipped_orders?: number;
  error?: string;
};

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchScripts(): Promise<ScriptCategory[]> {
  const res = await fetch(`${API_BASE}/api/scripts`);
  if (!res.ok) return [];
  return res.json();
}

export async function triggerScript(name: string, args?: Record<string, unknown>): Promise<string | null> {
  const res = await fetch(`${API_BASE}/api/scripts/${encodeURIComponent(name)}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ args: args ?? {} }),
  });
  if (!res.ok) return null;
  const body = await res.json();
  return body.run_id as string;
}

export async function fetchRun(runId: string): Promise<RunState | null> {
  const res = await fetch(`${API_BASE}/api/runs/${runId}`);
  if (!res.ok) return null;
  return res.json();
}

export function streamRun(
  runId: string,
  onMessage: (line: string) => void,
  onDone: (status: string) => void,
): () => void {
  const es = new EventSource(`${API_BASE}/api/runs/${runId}/stream`);

  es.addEventListener("message", (e: MessageEvent) => {
    onMessage(e.data);
  });

  es.addEventListener("done", (e: MessageEvent) => {
    const data = JSON.parse(e.data);
    onDone(data.status);
    es.close();
  });

  es.onerror = () => {
    es.close();
    onDone("error");
  };

  return () => es.close();
}

export async function fetchPaperSummary(): Promise<PaperSummary | null> {
  try {
    const res = await fetch(`${API_BASE}/api/paper/summary`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchPaperPositions(): Promise<Record<string, unknown>[]> {
  try {
    const res = await fetch(`${API_BASE}/api/paper/positions`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}
