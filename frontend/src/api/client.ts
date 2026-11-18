import type {
  CompareResult,
  DatasetMeta,
  Experiment,
  ExperimentSummary,
  Filter,
  ModelInfo,
  PreprocessConfig,
  PreviewResult,
  Profile,
  SampleInfo,
} from './types'

const BASE = import.meta.env.VITE_API_URL ?? ''

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, init)
  } catch {
    throw new ApiError(0, 'Não foi possível conectar à API. Verifique se o backend está rodando na porta 8000.')
  }
  if (!res.ok) {
    let msg = `Erro ${res.status}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') msg = body.detail
      else if (Array.isArray(body.detail))
        msg = body.detail.map((d: { loc?: string[]; msg: string }) => `${d.loc?.slice(1).join('.')}: ${d.msg}`).join('; ')
    } catch {
      /* resposta sem JSON */
    }
    throw new ApiError(res.status, msg)
  }
  return res.json() as Promise<T>
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  health: () => request<{ status: string }>('/api/health'),
  models: () => request<ModelInfo[]>('/api/models'),

  datasets: () => request<DatasetMeta[]>('/api/datasets'),
  dataset: (id: string) => request<DatasetMeta>(`/api/datasets/${id}`),
  samples: () => request<SampleInfo[]>('/api/samples'),
  loadSample: (key: string) => request<DatasetMeta>(`/api/samples/${key}`, { method: 'POST' }),
  upload: (file: File, opts: { sep?: string; decimal?: string; name?: string }) => {
    const fd = new FormData()
    fd.append('file', file)
    if (opts.sep) fd.append('sep', opts.sep)
    if (opts.decimal) fd.append('decimal', opts.decimal)
    if (opts.name) fd.append('name', opts.name)
    return request<DatasetMeta>('/api/datasets/upload', { method: 'POST', body: fd })
  },
  deleteDataset: (id: string) => request<{ ok: boolean }>(`/api/datasets/${id}`, { method: 'DELETE' }),
  rows: (
    id: string,
    q: { offset: number; limit: number; sort_by?: string | null; ascending?: boolean; search?: string; filters?: Filter[] },
  ) =>
    request<{ total: number; offset: number; columns: string[]; rows: Record<string, unknown>[]; index: number[] }>(
      `/api/datasets/${id}/rows`,
      json(q),
    ),
  profile: (id: string) => request<Profile>(`/api/datasets/${id}/profile`),
  targetInfo: (id: string, target: string) =>
    request<{ task: 'classification' | 'regression'; default_features: string[]; n_unique: number; classes?: { label: string; count: number }[] }>(
      `/api/datasets/${id}/target-info?target=${encodeURIComponent(target)}`,
    ),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  eda: <T = any>(id: string, body: Record<string, unknown>) => request<T>(`/api/datasets/${id}/eda`, json(body)),

  preview: (dataset_id: string, preprocessing: PreprocessConfig) =>
    request<PreviewResult>('/api/preprocess/preview', json({ dataset_id, preprocessing })),

  train: (body: Record<string, unknown>) => request<Experiment>('/api/train', json(body)),
  experiments: () => request<ExperimentSummary[]>('/api/experiments'),
  experiment: (id: string) => request<Experiment>(`/api/experiments/${id}`),
  patchExperiment: (id: string, body: { name?: string; tags?: string[]; notes?: string }) =>
    request<Experiment>(`/api/experiments/${id}`, { ...json(body), method: 'PATCH' }),
  deleteExperiment: (id: string) => request<{ ok: boolean }>(`/api/experiments/${id}`, { method: 'DELETE' }),
  compare: (ids: string[], metric?: string) => request<CompareResult>('/api/experiments/compare', json({ ids, metric })),
  modelUrl: (id: string) => `${BASE}/api/experiments/${id}/model`,
}
