import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { PreprocessConfig, PreviewResult, Profile } from '../api/types'
import { Plot } from '../components/Plot'
import {
  Badge,
  Button,
  cx,
  Edu,
  Empty,
  ErrorBox,
  Field,
  Loading,
  NumberInput,
  Notice,
  PageHeader,
  Panel,
  Select,
  Table,
  Toggle,
} from '../components/ui'
import { defaultPreprocess, useApp } from '../context/AppContext'
import { featureLabel, fmt, fmtInt, fmtPct } from '../lib/format'
import { useAsync } from '../lib/useAsync'
import { useProfile } from '../lib/useProfile'

type Cfg = PreprocessConfig

/** Diagrama do pipeline montado: mostra a ordem real das etapas. */
export function PipelineDiagram({ cfg, nNum, nCat, model }: { cfg: Cfg; nNum: number; nCat: number; model?: string }) {
  const numSteps = [
    cfg.outlier_treatment === 'clip_iqr' && 'Limitar outliers (IQR)',
    `Imputar (${{ mean: 'média', median: 'mediana', most_frequent: 'moda', constant: `constante ${cfg.numeric_fill_value}`, drop_rows: 'linhas removidas' }[cfg.numeric_imputation]})`,
    cfg.scaling !== 'none' && { standard: 'StandardScaler', minmax: 'MinMaxScaler', robust: 'RobustScaler' }[cfg.scaling],
  ].filter(Boolean) as string[]
  const catSteps = [
    `Imputar (${{ most_frequent: 'moda', constant: '"(ausente)"', drop_rows: 'linhas removidas' }[cfg.categorical_imputation]})`,
    cfg.encoding === 'onehot' ? `One-hot (máx. ${cfg.max_categories})` : 'Ordinal',
  ]
  const sel = { none: null, kbest_f: `SelectKBest F (k=${cfg.feature_selection.k})`, kbest_mi: `SelectKBest MI (k=${cfg.feature_selection.k})`, variance: `Variância > ${cfg.feature_selection.variance_threshold}` }[cfg.feature_selection.method]
  const Box = ({ children, muted }: { children: React.ReactNode; muted?: boolean }) => (
    <span className={cx('rounded border px-2 py-1 text-[12.5px] whitespace-nowrap', muted ? 'border-dashed border-line text-muted' : 'border-line bg-bg')}>{children}</span>
  )
  const Arrow = () => <span className="text-muted" aria-hidden>›</span>
  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="flex flex-col gap-1.5 rounded border border-line/70 p-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="w-24 text-[12px] text-muted">Numéricas ({nNum})</span>
          {nNum ? numSteps.map((s, i) => (
            <span key={s} className="flex items-center gap-1.5">
              {i > 0 && <Arrow />}
              <Box>{s}</Box>
            </span>
          )) : <Box muted>nenhuma</Box>}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="w-24 text-[12px] text-muted">Categóricas ({nCat})</span>
          {nCat ? catSteps.map((s, i) => (
            <span key={s} className="flex items-center gap-1.5">
              {i > 0 && <Arrow />}
              <Box>{s}</Box>
            </span>
          )) : <Box muted>nenhuma</Box>}
        </div>
      </div>
      <Arrow />
      <Box muted={!sel}>{sel ?? 'sem seleção'}</Box>
      <Arrow />
      <span className="rounded border border-accent bg-accent/10 px-2 py-1 text-[12.5px] font-semibold text-accent">{model ?? 'Modelo'}</span>
    </div>
  )
}

function FeaturePicker({
  profile,
  cfg,
  defaults,
  onChange,
  leaky,
}: {
  profile: Profile
  cfg: Cfg
  defaults: string[]
  onChange: (f: string[]) => void
  leaky: string[]
}) {
  const selected = cfg.features ?? defaults
  const cols = profile.columns.filter((c) => c.name !== cfg.target && c.name !== cfg.time_column)
  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-2">
        <Button size="sm" variant="ghost" onClick={() => onChange(defaults)}>
          Padrão sugerido
        </Button>
        <Button size="sm" variant="ghost" onClick={() => onChange([])}>
          Limpar
        </Button>
        <span className="num ml-auto self-center text-[13px] text-muted">{selected.length} selecionados</span>
      </div>
      <div className="grid max-h-72 grid-cols-1 gap-1 overflow-y-auto pr-1 sm:grid-cols-2">
        {cols.map((c) => {
          const usable = !['texto_identificador', 'data_hora', 'constante'].includes(c.semantic_type)
          const isLeaky = leaky.includes(c.name)
          return (
            <label
              key={c.name}
              className={cx(
                'flex items-center gap-2 rounded border px-2 py-1 text-[13px]',
                isLeaky ? 'border-alarm/60 bg-alarm/6' : 'border-line',
                !usable && 'opacity-60',
              )}
              title={!usable ? 'Tipo não utilizável diretamente como atributo' : undefined}
            >
              <input
                type="checkbox"
                disabled={!usable}
                checked={selected.includes(c.name)}
                onChange={(e) => onChange(e.target.checked ? [...selected, c.name] : selected.filter((x) => x !== c.name))}
              />
              <span className="min-w-0 flex-1 truncate">{c.name}</span>
              {isLeaky && <Badge tone="alarm">vazamento?</Badge>}
              <span className="text-[11.5px] text-muted">{c.semantic_label}</span>
            </label>
          )
        })}
      </div>
    </div>
  )
}

function PreviewView({ p }: { p: PreviewResult }) {
  return (
    <div className="flex flex-col gap-4">
      {p.warnings.map((w) => (
        <Notice key={w} level={w.startsWith('Possível vazamento') ? 'alarm' : 'warn'}>
          {w}
        </Notice>
      ))}
      {p.notes.map((n) => (
        <Notice key={n}>{n}</Notice>
      ))}
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Separação treino/teste">
          <div className="num mb-3 flex gap-6 text-[14px]">
            <span>
              <span className="text-muted">Treino </span>
              <b>{fmtInt(p.n_train)}</b>
            </span>
            <span>
              <span className="text-muted">Teste </span>
              <b>{fmtInt(p.n_test)}</b>
            </span>
            <span>
              <span className="text-muted">Estratégia </span>
              {{ random: 'aleatória', stratified: 'estratificada', temporal: 'temporal' }[p.split_strategy] ?? p.split_strategy}
            </span>
          </div>
          {p.class_distribution && (
            <Plot
              deps={[p]}
              height={Math.max(160, p.class_distribution.train.length * 46 + 70)}
              build={(c) => ({
                data: (['train', 'test'] as const).map((k, i) => ({
                  type: 'bar',
                  orientation: 'h',
                  name: k === 'train' ? 'Treino' : 'Teste',
                  y: p.class_distribution![k].map((d) => d.label),
                  x: p.class_distribution![k].map((d) => d.pct),
                  text: p.class_distribution![k].map((d) => `${fmtPct(d.pct)} (${fmtInt(d.count)})`),
                  textposition: 'outside',
                  marker: { color: i ? c.series[1] : c.series[0] },
                  hovertemplate: '%{y}: %{x:.2f}%<extra></extra>',
                })),
                layout: { barmode: 'group', xaxis: { title: '% das linhas', range: [0, 115] }, margin: { l: 110, r: 20, t: 10, b: 60 } },
              })}
            />
          )}
          {p.target_summary && (
            <Table
              head={['Alvo', 'Média', 'Desvio padrão']}
              align={['left', 'right', 'right']}
              rows={[
                ['Treino', fmt(p.target_summary.train.mean), fmt(p.target_summary.train.std)],
                ['Teste', fmt(p.target_summary.test.mean), fmt(p.target_summary.test.std)],
              ]}
            />
          )}
          <div className="mt-3">
            <Edu concept="split" />
          </div>
        </Panel>
        <Panel title="Atributos">
          <div className="num mb-3 flex flex-wrap gap-x-6 text-[14px]">
            <span>
              <span className="text-muted">Originais </span>
              {p.n_features_in}
            </span>
            <span>
              <span className="text-muted">Após codificação </span>
              {p.n_features_after_encoding}
            </span>
            <span>
              <span className="text-muted">Após seleção </span>
              <b>{p.n_features_selected}</b>
            </span>
          </div>
          {p.selection_scores ? (
            <Plot
              deps={[p]}
              height={Math.min(520, Math.max(200, p.selection_scores.length * 20 + 50))}
              build={(c) => ({
                data: [
                  {
                    type: 'bar',
                    orientation: 'h',
                    y: p.selection_scores!.map((s) => featureLabel(s.feature)),
                    x: p.selection_scores!.map((s) => s.score),
                    marker: { color: p.selection_scores!.map((s) => (s.selected ? c.accent : c.line)) },
                    hovertemplate: '%{y}: %{x:.4g}<extra></extra>',
                  },
                ],
                layout: { yaxis: { autorange: 'reversed' }, xaxis: { title: 'Escore (calculado só no treino)' }, margin: { l: 170, r: 16, t: 6, b: 44 } },
              })}
            />
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {p.selected_features.map((f) => (
                <span key={f} className="rounded border border-line bg-bg px-1.5 py-0.5 text-[12.5px]">
                  {featureLabel(f)}
                </span>
              ))}
            </div>
          )}
          {p.ignored.length > 0 && <p className="mt-2 text-[13px] text-muted">Ignoradas: {p.ignored.join(', ')}</p>}
        </Panel>
      </div>

      <Panel title="Parâmetros aprendidos no conjunto de treino" subtitle="Estes valores são reaplicados, sem reajuste, ao teste e a dados novos">
        <Table
          head={['Coluna', 'Tipo', 'Valor de imputação', 'Ausentes (treino/teste)', 'Escala: centro', 'Escala: fator', 'Limites de outlier / categorias']}
          align={['left', 'left', 'right', 'right', 'right', 'right', 'left']}
          rows={p.learned_params.map((l) => [
            l.column,
            l.kind,
            typeof l.impute_value === 'number' ? fmt(l.impute_value) : (l.impute_value ?? '—'),
            `${fmtInt(l.train_missing)} / ${fmtInt(l.test_missing)}`,
            l.center !== undefined ? fmt(l.center) : '—',
            l.scale !== undefined ? fmt(l.scale) : '—',
            l.clip_lower !== undefined
              ? `[${fmt(l.clip_lower)}; ${fmt(l.clip_upper)}]`
              : l.categories
                ? `${l.n_categories}: ${l.categories.slice(0, 6).join(', ')}${(l.n_categories ?? 0) > 6 ? '…' : ''}`
                : '—',
          ])}
        />
        <div className="mt-3">
          <Edu concept="leakage" />
        </div>
      </Panel>

      <Panel title="Amostra do treino transformado" subtitle={`Primeiras ${p.transformed_sample.rows.length} linhas do que o modelo recebe. Teste transformado: ${p.test_transformed_shape.join(' × ')}`}>
        <div className="max-h-80 overflow-auto rounded border border-line">
          <table className="num w-full border-collapse text-[12.5px]">
            <thead className="sticky top-0">
              <tr className="bg-panel-2">
                {p.transformed_sample.columns.map((c) => (
                  <th key={c} className="border-b border-line px-2 py-1 text-right font-semibold whitespace-nowrap">
                    {featureLabel(c)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {p.transformed_sample.rows.map((r, i) => (
                <tr key={i} className="border-b border-line/50">
                  {r.map((v, j) => (
                    <td key={j} className="px-2 py-0.5 text-right">
                      {fmt(v, 4)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  )
}

export default function PreprocessPage() {
  const { datasetId, dataset, preprocess, setPreprocess } = useApp()
  const profile = useProfile(datasetId)
  const nav = useNavigate()
  const [cfg, setCfg] = useState<Cfg | null>(null)
  const [preview, setPreview] = useState<PreviewResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Inicializa a configuração quando o dataset muda
  useEffect(() => {
    setPreview(null)
    setError(null)
    if (!profile.data) return setCfg(null)
    if (preprocess && profile.data.columns.some((c) => c.name === preprocess.target)) setCfg(preprocess)
    else {
      const cols = profile.data.columns
      const target = cols.at(-1)!.name
      const time = cols.find((c) => c.semantic_type === 'data_hora')?.name ?? null
      setCfg({ ...defaultPreprocess(target), time_column: time })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profile.data, datasetId])

  const info = useAsync(
    () => (datasetId && cfg?.target ? api.targetInfo(datasetId, cfg.target) : Promise.resolve(null)),
    [datasetId, cfg?.target],
    !!cfg?.target,
  )
  const task = cfg?.task === 'auto' ? info.data?.task : cfg?.task
  const defaults = useMemo(
    () => (info.data?.default_features ?? []).filter((f) => f !== cfg?.time_column),
    [info.data, cfg?.time_column],
  )

  if (!datasetId || !dataset)
    return (
      <>
        <PageHeader title="Pré-processamento" />
        <Empty title="Nenhum dataset ativo" action={<Link className="text-accent underline" to="/datasets">Carregar um dataset</Link>} />
      </>
    )
  if (profile.loading || !cfg || !profile.data) return <Loading />

  const set = <K extends keyof Cfg>(k: K, v: Cfg[K]) => {
    setCfg({ ...cfg, [k]: v })
    setPreview(null)
  }
  const features = cfg.features ?? defaults
  const kinds = profile.data.columns.filter((c) => features.includes(c.name))
  const nNum = kinds.filter((c) => c.stats.mean !== undefined && c.semantic_type !== 'categorica').length
  const nCat = kinds.length - nNum
  const leaky = preview?.leakage.map((l) => l.feature) ?? []

  async function runPreview() {
    if (!cfg || !datasetId) return
    setBusy(true)
    setError(null)
    try {
      const body = { ...cfg, features: cfg.features ?? defaults }
      const r = await api.preview(datasetId, body)
      setPreview(r)
      setPreprocess(body)
      setCfg(body)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const allCols = profile.data.columns.map((c) => ({ value: c.name, label: c.name }))

  return (
    <>
      <PageHeader
        title="Pré-processamento"
        actions={
          <div className="flex gap-2">
            <Button variant="primary" loading={busy} onClick={runPreview}>
              Aplicar e visualizar
            </Button>
            <Button
              disabled={!preview && !preprocess}
              onClick={() => nav('/train')}
              title={!preview && !preprocess ? 'Visualize o pré-processamento antes de treinar' : undefined}
            >
              Ir para treinamento
            </Button>
          </div>
        }
      >
        Monte o pipeline. Ele é ajustado somente no conjunto de treino; na validação cruzada é refeito dentro de cada fold.
      </PageHeader>

      <div className="mb-4 rounded-md border border-line bg-panel px-4 py-3">
        <PipelineDiagram cfg={cfg} nNum={nNum} nCat={nCat} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Alvo e tarefa">
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Variável alvo (o que prever)">
              <Select value={cfg.target} onChange={(v) => setCfg({ ...cfg, target: v, features: null, positive_class: null })} options={allCols} />
            </Field>
            <Field label="Tarefa" hint={info.data ? `Inferida: ${info.data.task === 'classification' ? 'classificação' : 'regressão'} (${info.data.n_unique} valores distintos)` : undefined}>
              <Select
                value={cfg.task}
                onChange={(v) => set('task', v)}
                options={[
                  { value: 'auto', label: 'Automática' },
                  { value: 'classification', label: 'Classificação' },
                  { value: 'regression', label: 'Regressão' },
                ]}
              />
            </Field>
            {task === 'classification' && info.data?.classes?.length === 2 && (
              <Field label="Classe positiva" hint="Precision/recall/F1 são calculados para ela. Em falhas, a positiva é o evento raro.">
                <Select
                  value={cfg.positive_class ?? ''}
                  onChange={(v) => set('positive_class', v || null)}
                  options={[{ value: '', label: 'Automática (1 / true / minoritária)' }, ...info.data.classes.map((c) => ({ value: c.label, label: `${c.label} (${fmtInt(c.count)})` }))]}
                />
              </Field>
            )}
          </div>
        </Panel>

        <Panel title="Separação treino/teste">
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Estratégia">
              <Select
                value={cfg.split_strategy}
                onChange={(v) => set('split_strategy', v)}
                options={[
                  { value: 'stratified', label: 'Estratificada (mantém proporção das classes)' },
                  { value: 'random', label: 'Aleatória' },
                  { value: 'temporal', label: 'Temporal (teste = período mais recente)' },
                ]}
              />
            </Field>
            {cfg.split_strategy === 'temporal' && (
              <Field label="Coluna de tempo" hint="Excluída dos atributos; usada só para ordenar.">
                <Select value={cfg.time_column ?? ''} onChange={(v) => set('time_column', v || null)} options={[{ value: '', label: 'Ordem das linhas' }, ...allCols]} />
              </Field>
            )}
            <Field label={`Tamanho do teste: ${Math.round(cfg.test_size * 100)}%`}>
              <input type="range" min={10} max={50} step={5} value={cfg.test_size * 100} onChange={(e) => set('test_size', +e.target.value / 100)} className="mt-2 w-full" />
            </Field>
            <Field label="Semente aleatória" hint="Mesma semente = mesma separação e mesmos folds, o que permite comparar modelos de forma pareada.">
              <NumberInput value={cfg.random_state} onChange={(v) => set('random_state', v ?? 42)} step={1} />
            </Field>
          </div>
          {cfg.split_strategy === 'temporal' && (
            <div className="mt-3">
              <Edu title="Por que separar no tempo">
                Em dados de sensores, linhas vizinhas são muito parecidas. Uma separação aleatória coloca no teste leituras de minutos antes
                ou depois de leituras do treino e infla o resultado. Com a separação temporal o modelo é avaliado como seria usado: prevendo
                o futuro. A validação cruzada passa a usar TimeSeriesSplit.
              </Edu>
            </div>
          )}
        </Panel>

        <Panel title="Atributos de entrada" className="xl:col-span-2">
          <FeaturePicker profile={profile.data} cfg={cfg} defaults={defaults} onChange={(f) => set('features', f)} leaky={leaky} />
          {leaky.length > 0 && (
            <div className="mt-3">
              <Button size="sm" variant="danger" onClick={() => set('features', features.filter((f) => !leaky.includes(f)))}>
                Remover atributos suspeitos de vazamento ({leaky.join(', ')})
              </Button>
            </div>
          )}
        </Panel>

        <Panel title="Valores ausentes e outliers">
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Numéricas">
              <Select
                value={cfg.numeric_imputation}
                onChange={(v) => set('numeric_imputation', v)}
                options={[
                  { value: 'median', label: 'Mediana (robusta a outliers)' },
                  { value: 'mean', label: 'Média' },
                  { value: 'most_frequent', label: 'Moda' },
                  { value: 'constant', label: 'Valor constante' },
                  { value: 'drop_rows', label: 'Remover linhas com ausentes' },
                ]}
              />
            </Field>
            {cfg.numeric_imputation === 'constant' && (
              <Field label="Valor constante">
                <NumberInput value={cfg.numeric_fill_value} onChange={(v) => set('numeric_fill_value', v ?? 0)} />
              </Field>
            )}
            <Field label="Categóricas">
              <Select
                value={cfg.categorical_imputation}
                onChange={(v) => set('categorical_imputation', v)}
                options={[
                  { value: 'most_frequent', label: 'Categoria mais frequente' },
                  { value: 'constant', label: 'Nova categoria "(ausente)"' },
                  { value: 'drop_rows', label: 'Remover linhas com ausentes' },
                ]}
              />
            </Field>
            <Field label="Outliers numéricos">
              <Select
                value={cfg.outlier_treatment}
                onChange={(v) => set('outlier_treatment', v)}
                options={[
                  { value: 'none', label: 'Manter como estão' },
                  { value: 'clip_iqr', label: 'Limitar a 1,5 × IQR (limites do treino)' },
                ]}
              />
            </Field>
            <div className="sm:col-span-2">
              <Toggle
                checked={cfg.add_missing_indicator}
                onChange={(v) => set('add_missing_indicator', v)}
                label="Adicionar indicador de ausência"
                hint="Cria uma coluna 0/1 para cada variável com ausentes. Útil quando a falta de leitura é informativa (sensor travado antes de uma falha)."
              />
            </div>
          </div>
        </Panel>

        <Panel title="Escala, codificação e seleção">
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Escala das numéricas" hint="Árvores e XGBoost não precisam; modelos lineares sim.">
              <Select
                value={cfg.scaling}
                onChange={(v) => set('scaling', v)}
                options={[
                  { value: 'standard', label: 'Padronização (z-score)' },
                  { value: 'minmax', label: 'Normalização min-max [0, 1]' },
                  { value: 'robust', label: 'Robusta (mediana e IQR)' },
                  { value: 'none', label: 'Nenhuma' },
                ]}
              />
            </Field>
            <Field label="Codificação das categóricas">
              <Select
                value={cfg.encoding}
                onChange={(v) => set('encoding', v)}
                options={[
                  { value: 'onehot', label: 'One-hot' },
                  { value: 'ordinal', label: 'Ordinal (inteiros)' },
                ]}
              />
            </Field>
            {cfg.encoding === 'onehot' && (
              <Field label="Máx. categorias por coluna" hint="As mais raras viram “infrequent”.">
                <NumberInput value={cfg.max_categories} min={2} step={1} onChange={(v) => set('max_categories', Math.max(2, v ?? 20))} />
              </Field>
            )}
            <Field label="Seleção de atributos">
              <Select
                value={cfg.feature_selection.method}
                onChange={(v) => set('feature_selection', { ...cfg.feature_selection, method: v })}
                options={[
                  { value: 'none', label: 'Nenhuma' },
                  { value: 'kbest_f', label: 'K melhores — teste F (ANOVA / regressão)' },
                  { value: 'kbest_mi', label: 'K melhores — informação mútua' },
                  { value: 'variance', label: 'Limiar de variância' },
                ]}
              />
            </Field>
            {cfg.feature_selection.method.startsWith('kbest') && (
              <Field label="k (atributos mantidos)">
                <NumberInput value={cfg.feature_selection.k} min={1} step={1} onChange={(v) => set('feature_selection', { ...cfg.feature_selection, k: Math.max(1, v ?? 10) })} />
              </Field>
            )}
            {cfg.feature_selection.method === 'variance' && (
              <Field label="Variância mínima">
                <NumberInput value={cfg.feature_selection.variance_threshold} min={0} onChange={(v) => set('feature_selection', { ...cfg.feature_selection, variance_threshold: v ?? 0 })} />
              </Field>
            )}
          </div>
          <div className="mt-3">
            <Edu title="Seleção também é aprendizado">
              Escolher os k melhores atributos olhando o dataset inteiro já vaza informação do teste. Aqui a seleção é uma etapa do pipeline:
              os escores são calculados só no treino (e em cada fold da validação cruzada).
            </Edu>
          </div>
        </Panel>
      </div>

      <div className="mt-6">
        <ErrorBox error={error} />
        {busy && <Loading label="Ajustando o pipeline no treino…" />}
        {preview && (
          <>
            <h2 className="mb-3 font-display text-[22px] font-semibold">Resultado do pré-processamento</h2>
            <PreviewView p={preview} />
          </>
        )}
        {!preview && !busy && (
          <p className="text-sm text-muted">Clique em “Aplicar e visualizar” para ajustar o pipeline e ver o que foi aprendido no treino.</p>
        )}
      </div>
    </>
  )
}
