import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { DatasetMeta, ModelInfo, PreprocessConfig } from '../api/types'

export const defaultPreprocess = (target = ''): PreprocessConfig => ({
  target,
  task: 'auto',
  features: null,
  numeric_imputation: 'median',
  numeric_fill_value: 0,
  categorical_imputation: 'most_frequent',
  add_missing_indicator: false,
  outlier_treatment: 'none',
  scaling: 'standard',
  encoding: 'onehot',
  max_categories: 20,
  feature_selection: { method: 'none', k: 10, variance_threshold: 0 },
  split_strategy: 'stratified',
  time_column: null,
  test_size: 0.2,
  random_state: 42,
  positive_class: null,
})

interface AppState {
  datasets: DatasetMeta[]
  refreshDatasets: () => Promise<void>
  datasetId: string | null
  dataset: DatasetMeta | null
  setDatasetId: (id: string | null) => void
  models: ModelInfo[]
  theme: 'dark' | 'light'
  toggleTheme: () => void
  eduMode: boolean
  setEduMode: (v: boolean) => void
  preprocess: PreprocessConfig | null
  setPreprocess: (cfg: PreprocessConfig) => void
  apiOnline: boolean | null
}

const Ctx = createContext<AppState | null>(null)

function read<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return v === null ? fallback : (JSON.parse(v) as T)
  } catch {
    return fallback
  }
}
function write(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* armazenamento indisponível */
  }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([])
  const [datasetId, setDatasetIdState] = useState<string | null>(() => read('mllab-dataset', null))
  const [models, setModels] = useState<ModelInfo[]>([])
  const [theme, setTheme] = useState<'dark' | 'light'>(() =>
    document.documentElement.classList.contains('dark') ? 'dark' : 'light',
  )
  const [eduMode, setEduModeState] = useState<boolean>(() => read('mllab-edu', true))
  const [preprocessMap, setPreprocessMap] = useState<Record<string, PreprocessConfig>>(() => read('mllab-prep', {}))
  const [apiOnline, setApiOnline] = useState<boolean | null>(null)

  const refreshDatasets = useCallback(async () => {
    try {
      const list = await api.datasets()
      setDatasets(list)
      setApiOnline(true)
      setDatasetIdState((cur) => (cur && list.some((d) => d.id === cur) ? cur : (list[0]?.id ?? null)))
    } catch {
      setApiOnline(false)
    }
  }, [])

  useEffect(() => {
    refreshDatasets()
    api.models().then(setModels).catch(() => setApiOnline(false))
  }, [refreshDatasets])

  useEffect(() => {
    write('mllab-dataset', datasetId)
  }, [datasetId])
  useEffect(() => {
    write('mllab-prep', preprocessMap)
  }, [preprocessMap])

  const value = useMemo<AppState>(
    () => ({
      datasets,
      refreshDatasets,
      datasetId,
      dataset: datasets.find((d) => d.id === datasetId) ?? null,
      setDatasetId: setDatasetIdState,
      models,
      theme,
      toggleTheme: () => {
        const next = theme === 'dark' ? 'light' : 'dark'
        document.documentElement.classList.toggle('dark', next === 'dark')
        try {
          localStorage.setItem('mllab-theme', next)
        } catch {
          /* ignore */
        }
        setTheme(next)
      },
      eduMode,
      setEduMode: (v) => {
        setEduModeState(v)
        write('mllab-edu', v)
      },
      preprocess: datasetId ? (preprocessMap[datasetId] ?? null) : null,
      setPreprocess: (cfg) => datasetId && setPreprocessMap((m) => ({ ...m, [datasetId]: cfg })),
      apiOnline,
    }),
    [datasets, refreshDatasets, datasetId, models, theme, eduMode, preprocessMap, apiOnline],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useApp fora do AppProvider')
  return v
}
