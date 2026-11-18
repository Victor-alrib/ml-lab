import { Fragment, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { ColumnProfile, Profile } from '../api/types'
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
  Notice,
  PageHeader,
  Panel,
  Segmented,
  Select,
  Table,
  TextInput,
} from '../components/ui'
import { useApp } from '../context/AppContext'
import { fmt, fmtBytes, fmtDate, fmtInt, fmtPct } from '../lib/format'
import { useAsync } from '../lib/useAsync'
import { useProfile } from '../lib/useProfile'

function AddDataset() {
  const { refreshDatasets, setDatasetId } = useApp()
  const samples = useAsync(() => api.samples(), [])
  const [file, setFile] = useState<File | null>(null)
  const [sep, setSep] = useState('')
  const [decimal, setDecimal] = useState('')
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [drag, setDrag] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  async function upload() {
    if (!file) return
    setBusy('upload')
    setError(null)
    try {
      const ds = await api.upload(file, { sep, decimal })
      await refreshDatasets()
      setDatasetId(ds.id)
      setFile(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  async function loadSample(key: string) {
    setBusy(key)
    setError(null)
    try {
      const ds = await api.loadSample(key)
      await refreshDatasets()
      setDatasetId(ds.id)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <Panel title="Enviar CSV">
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setDrag(true)
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDrag(false)
            const f = e.dataTransfer.files[0]
            if (f) setFile(f)
          }}
          onClick={() => input.current?.click()}
          className={cx(
            'flex cursor-pointer flex-col items-center justify-center rounded border-2 border-dashed px-4 py-7 text-center transition-colors',
            drag ? 'border-accent bg-accent/5' : 'border-line hover:border-muted',
          )}
        >
          <input
            ref={input}
            type="file"
            accept=".csv,.txt,.tsv"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          {file ? (
            <>
              <div className="font-semibold">{file.name}</div>
              <div className="text-[13px] text-muted">{fmtBytes(file.size)}</div>
            </>
          ) : (
            <>
              <div className="font-medium">Arraste um arquivo .csv ou clique para escolher</div>
              <div className="mt-1 text-[13px] text-muted">Separador e decimal são detectados automaticamente (inclusive ; e vírgula)</div>
            </>
          )}
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3">
          <Field label="Separador">
            <Select
              value={sep}
              onChange={setSep}
              options={[
                { value: '', label: 'Detectar' },
                { value: ',', label: 'Vírgula ( , )' },
                { value: ';', label: 'Ponto e vírgula ( ; )' },
                { value: '\t', label: 'Tabulação' },
              ]}
            />
          </Field>
          <Field label="Decimal">
            <Select
              value={decimal}
              onChange={setDecimal}
              options={[
                { value: '', label: 'Detectar' },
                { value: '.', label: 'Ponto ( . )' },
                { value: ',', label: 'Vírgula ( , )' },
              ]}
            />
          </Field>
        </div>
        <Button variant="primary" className="mt-3 w-full" disabled={!file} loading={busy === 'upload'} onClick={upload}>
          Enviar dataset
        </Button>
      </Panel>

      <Panel title="Datasets de exemplo" subtitle="Para começar sem precisar de arquivo">
        {samples.loading && <Loading label="Carregando lista…" />}
        <ul className="divide-y divide-line">
          {samples.data?.map((s) => (
            <li key={s.key} className="flex flex-wrap items-start justify-between gap-3 py-2.5 first:pt-0 last:pb-0">
              <div className="min-w-0 flex-1">
                <div className="font-semibold">{s.name}</div>
                <div className="text-[13px] text-muted">{s.task_hint}</div>
              </div>
              <Button size="sm" loading={busy === s.key} onClick={() => loadSample(s.key)}>
                Carregar
              </Button>
            </li>
          ))}
        </ul>
      </Panel>
      <div className="lg:col-span-2">
        <ErrorBox error={error} />
      </div>
    </div>
  )
}

function DataTable({ datasetId }: { datasetId: string }) {
  const [offset, setOffset] = useState(0)
  const [sortBy, setSortBy] = useState<string | null>(null)
  const [asc, setAsc] = useState(true)
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const limit = 50
  const res = useAsync(
    () => api.rows(datasetId, { offset, limit, sort_by: sortBy, ascending: asc, search: query || undefined }),
    [datasetId, offset, sortBy, asc, query],
  )
  const d = res.data
  return (
    <div className="flex flex-col gap-3">
      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault()
          setOffset(0)
          setQuery(search)
        }}
      >
        <TextInput
          aria-label="Buscar nas linhas"
          placeholder="Buscar valor em qualquer coluna"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-xs"
        />
        <Button type="submit" size="sm">
          Buscar
        </Button>
        {d && (
          <span className="num ml-auto text-[13px] text-muted">
            {fmtInt(d.total)} linhas · exibindo {fmtInt(d.total ? offset + 1 : 0)}–{fmtInt(Math.min(offset + limit, d.total))}
          </span>
        )}
      </form>
      <ErrorBox error={res.error} />
      {d && (
        <div className="max-h-[560px] overflow-auto rounded border border-line">
          <table className="num w-full border-collapse text-[13px]">
            <thead className="sticky top-0 z-10">
              <tr className="bg-panel-2">
                <th className="border-b border-line px-2 py-1.5 text-right font-normal text-muted">#</th>
                {d.columns.map((c) => (
                  <th key={c} className="border-b border-line px-2 py-1.5 text-left font-semibold whitespace-nowrap">
                    <button
                      className="hover:text-accent"
                      onClick={() => {
                        if (sortBy === c) setAsc(!asc)
                        else {
                          setSortBy(c)
                          setAsc(true)
                        }
                        setOffset(0)
                      }}
                    >
                      {c}
                      {sortBy === c ? (asc ? ' ▲' : ' ▼') : ''}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {d.rows.map((r, i) => (
                <tr key={i} className="border-b border-line/50 hover:bg-panel-2/60">
                  <td className="px-2 py-1 text-right text-muted">{d.index[i]}</td>
                  {d.columns.map((c) => (
                    <td key={c} className="px-2 py-1 whitespace-nowrap">
                      {r[c] === null ? <span className="text-warn italic">ausente</span> : String(r[c])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {d && d.total > limit && (
        <div className="flex gap-2">
          <Button size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
            Anterior
          </Button>
          <Button size="sm" disabled={offset + limit >= d.total} onClick={() => setOffset(offset + limit)}>
            Próxima
          </Button>
        </div>
      )}
    </div>
  )
}

const TYPE_TONE: Record<string, 'accent' | 'neutral' | 'warn'> = {
  numerica_continua: 'accent',
  numerica_discreta: 'accent',
  binaria: 'neutral',
  categorica: 'neutral',
  data_hora: 'neutral',
  texto_identificador: 'warn',
  constante: 'warn',
}

function ColumnDetail({ c }: { c: ColumnProfile }) {
  const s = c.stats
  if (s.mean !== undefined) {
    return (
      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-[13px] sm:grid-cols-4">
        {(
          [
            ['Média', s.mean],
            ['Desvio padrão', s.std],
            ['Mínimo', s.min],
            ['Máximo', s.max],
            ['Q1 (25%)', s.q1],
            ['Mediana', s.median],
            ['Q3 (75%)', s.q3],
            ['Assimetria', s.skewness],
            ['Curtose (excesso)', s.kurtosis],
            ['Coef. de variação', s.cv_pct != null ? fmtPct(s.cv_pct) : null],
          ] as [string, unknown][]
        ).map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2 border-b border-line/50 py-0.5">
            <span className="text-muted">{k}</span>
            <span className="num">{typeof v === 'string' ? v : fmt(v)}</span>
          </div>
        ))}
        {c.outliers && (
          <div className="col-span-2 mt-1 text-muted sm:col-span-4">
            Outliers por IQR: <span className="num text-ink">{fmtInt(c.outliers.iqr_count)}</span> ({fmtPct(c.outliers.iqr_pct, 2)}) fora de [
            <span className="num">{fmt(c.outliers.lower_bound)}</span>; <span className="num">{fmt(c.outliers.upper_bound)}</span>] · por z-score (|z|&gt;3):{' '}
            <span className="num text-ink">{fmtInt(c.outliers.z_count)}</span>
          </div>
        )}
      </div>
    )
  }
  return (
    <div className="flex flex-wrap gap-1.5 text-[13px]">
      {s.top_values?.map((t) => (
        <span key={t.value} className="rounded border border-line bg-bg px-2 py-0.5">
          {t.value} <span className="num text-muted">{fmtInt(t.count)} · {fmtPct(t.pct)}</span>
        </span>
      ))}
    </div>
  )
}

function Variables({ profile }: { profile: Profile }) {
  const [open, setOpen] = useState<string | null>(null)
  return (
    <div className="flex flex-col gap-3">
      <Edu title="Tipos de variáveis">
        O tipo é inferido dos valores: numéricas contínuas recebem imputação/escala; categóricas e binárias de texto recebem
        codificação; data/hora e identificadores (um valor por linha) ficam fora dos atributos por padrão — um ID não generaliza
        para máquinas novas.
      </Edu>
      <div className="overflow-x-auto rounded border border-line">
        <table className="w-full border-collapse text-[13.5px]">
          <thead>
            <tr className="bg-panel-2 text-left text-muted">
              <th className="border-b border-line px-3 py-1.5 font-semibold">Coluna</th>
              <th className="border-b border-line px-3 py-1.5 font-semibold">Tipo</th>
              <th className="border-b border-line px-3 py-1.5 text-right font-semibold">Únicos</th>
              <th className="border-b border-line px-3 py-1.5 text-right font-semibold">Ausentes</th>
              <th className="border-b border-line px-3 py-1.5 text-right font-semibold">Média / moda</th>
              <th className="border-b border-line px-3 py-1.5 text-right font-semibold">Outliers (IQR)</th>
            </tr>
          </thead>
          <tbody className="num">
            {profile.columns.map((c) => (
              <Fragment key={c.name}>
                <tr
                  className={cx('cursor-pointer border-b border-line/60 hover:bg-panel-2/60', open === c.name && 'bg-panel-2/60')}
                  onClick={() => setOpen(open === c.name ? null : c.name)}
                >
                  <td className="px-3 py-1.5 font-medium">
                    <span className="mr-1.5 inline-block w-3 text-muted">{open === c.name ? '−' : '+'}</span>
                    {c.name}
                    {c.possible_id && (
                      <span className="ml-2">
                        <Badge tone="warn">possível ID</Badge>
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-1.5">
                    <Badge tone={TYPE_TONE[c.semantic_type]}>{c.semantic_label}</Badge>
                    <span className="ml-1.5 font-mono text-[11.5px] text-muted">{c.dtype}</span>
                  </td>
                  <td className="px-3 py-1.5 text-right">{fmtInt(c.unique)}</td>
                  <td className={cx('px-3 py-1.5 text-right', c.missing > 0 && 'text-warn')}>
                    {c.missing ? `${fmtInt(c.missing)} (${fmtPct(c.missing_pct)})` : '0'}
                  </td>
                  <td className="px-3 py-1.5 text-right">{c.stats.mean !== undefined ? fmt(c.stats.mean) : (c.stats.top ?? '—')}</td>
                  <td className="px-3 py-1.5 text-right">
                    {c.outliers ? (c.outliers.iqr_count ? `${fmtInt(c.outliers.iqr_count)} (${fmtPct(c.outliers.iqr_pct)})` : '0') : '—'}
                  </td>
                </tr>
                {open === c.name && (
                  <tr className="border-b border-line/60 bg-panel-2/30">
                    <td colSpan={6} className="px-4 py-3">
                      <ColumnDetail c={c} />
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Quality({ profile }: { profile: Profile }) {
  const missing = profile.columns.filter((c) => c.missing > 0).sort((a, b) => b.missing_pct - a.missing_pct)
  const outl = profile.columns.filter((c) => c.outliers && c.outliers.iqr_count > 0).sort((a, b) => b.outliers!.iqr_pct - a.outliers!.iqr_pct)
  return (
    <div className="flex flex-col gap-4">
      {profile.warnings.map((w) => (
        <Notice key={w} level="warn">
          {w}
        </Notice>
      ))}
      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Valores ausentes por coluna" subtitle={`${fmtInt(profile.rows_with_missing)} linhas (${fmtPct(profile.rows_with_missing_pct)}) têm ao menos um valor ausente`}>
          {missing.length ? (
            <Plot
              deps={[profile]}
              height={Math.max(180, missing.length * 30 + 60)}
              build={(c) => ({
                data: [
                  {
                    type: 'bar',
                    orientation: 'h',
                    y: missing.map((m) => m.name),
                    x: missing.map((m) => m.missing_pct),
                    text: missing.map((m) => `${fmtInt(m.missing)} (${fmtPct(m.missing_pct)})`),
                    textposition: 'outside',
                    marker: { color: c.warn },
                    hovertemplate: '%{y}: %{x:.2f}%<extra></extra>',
                  },
                ],
                layout: { xaxis: { title: '% ausente', rangemode: 'tozero' }, yaxis: { autorange: 'reversed' }, margin: { l: 150, r: 80, t: 10, b: 44 } },
              })}
            />
          ) : (
            <p className="text-sm text-muted">Nenhum valor ausente.</p>
          )}
        </Panel>
        <Panel title="Possíveis outliers (regra de Tukey, 1,5 × IQR)">
          {outl.length ? (
            <Table
              head={['Coluna', 'Fora do intervalo', '%', 'Limite inferior', 'Limite superior', '|z| > 3']}
              align={['left', 'right', 'right', 'right', 'right', 'right']}
              rows={outl.map((c) => [
                c.name,
                fmtInt(c.outliers!.iqr_count),
                fmtPct(c.outliers!.iqr_pct, 2),
                fmt(c.outliers!.lower_bound),
                fmt(c.outliers!.upper_bound),
                fmtInt(c.outliers!.z_count),
              ])}
            />
          ) : (
            <p className="text-sm text-muted">Nenhuma coluna numérica com valores fora de 1,5 × IQR.</p>
          )}
          <div className="mt-3">
            <Edu title="Outlier não é erro automaticamente">
              Em dados industriais, um valor extremo pode ser um pico espúrio de sensor ou exatamente o evento de falha que você quer
              prever. Investigue antes de remover; no pré-processamento você pode limitar (winsorizar) os valores com limites aprendidos
              só no treino.
            </Edu>
          </div>
        </Panel>
      </div>
    </div>
  )
}

export default function DatasetsPage() {
  const { datasets, dataset, datasetId, refreshDatasets, setDatasetId } = useApp()
  const [tab, setTab] = useState<'table' | 'vars' | 'quality'>('vars')
  const [showAdd, setShowAdd] = useState(false)
  const profile = useProfile(datasetId)
  const nav = useNavigate()

  async function remove(id: string) {
    if (!confirm('Excluir este dataset? Os experimentos já salvos continuam no histórico.')) return
    await api.deleteDataset(id)
    await refreshDatasets()
  }

  return (
    <>
      <PageHeader
        title="Datasets"
        actions={
          datasets.length > 0 && (
            <Button onClick={() => setShowAdd((s) => !s)}>{showAdd ? 'Fechar' : 'Adicionar dataset'}</Button>
          )
        }
      >
        Envie um CSV ou carregue um exemplo. O perfil abaixo é calculado sobre todas as linhas do arquivo.
      </PageHeader>

      {(datasets.length === 0 || showAdd) && (
        <div className="mb-6">
          <AddDataset />
        </div>
      )}

      {datasets.length > 0 && (
        <Panel title="Datasets carregados" pad={false} className="mb-6">
          <ul className="divide-y divide-line">
            {datasets.map((d) => (
              <li
                key={d.id}
                className={cx('flex flex-wrap items-center gap-3 px-4 py-2', d.id === datasetId && 'bg-panel-2')}
              >
                <button className="min-w-0 flex-1 text-left" onClick={() => setDatasetId(d.id)}>
                  <span className={cx('font-medium', d.id === datasetId && 'text-accent')}>{d.name}</span>
                  <span className="num ml-3 text-[13px] text-muted">
                    {fmtInt(d.n_rows)} linhas × {d.n_cols} colunas · {fmtBytes(d.size_bytes)} · {fmtDate(d.created_at)}
                  </span>
                </button>
                {d.id === datasetId ? <Badge tone="accent">ativo</Badge> : <Button size="sm" variant="ghost" onClick={() => setDatasetId(d.id)}>Selecionar</Button>}
                <Button size="sm" variant="danger" onClick={() => remove(d.id)}>
                  Excluir
                </Button>
              </li>
            ))}
          </ul>
        </Panel>
      )}

      {dataset && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div className="max-w-3xl">
              <h2 className="font-display text-[22px] font-semibold">{dataset.name}</h2>
              {dataset.description && <p className="text-[14px] text-muted">{dataset.description}</p>}
              {dataset.parse_info?.sep && (
                <p className="text-[12.5px] text-muted">
                  Lido com separador “{dataset.parse_info.sep === '\t' ? 'tab' : dataset.parse_info.sep}”, decimal “{dataset.parse_info.decimal}”,
                  codificação {dataset.parse_info.encoding}.
                </p>
              )}
            </div>
            <Button variant="primary" onClick={() => nav('/eda')}>
              Explorar dados
            </Button>
          </div>
          {profile.data && (
            <div className="num flex flex-wrap gap-x-8 gap-y-1 text-[14px]">
              <span>
                <span className="text-muted">Linhas </span>
                {fmtInt(profile.data.n_rows)}
              </span>
              <span>
                <span className="text-muted">Colunas </span>
                {profile.data.n_cols}
              </span>
              {Object.entries(profile.data.type_counts).map(([k, v]) => (
                <span key={k}>
                  <span className="text-muted">{k} </span>
                  {v}
                </span>
              ))}
              <span className={profile.data.total_missing ? 'text-warn' : ''}>
                <span className="text-muted">Valores ausentes </span>
                {fmtInt(profile.data.total_missing)}
              </span>
              <span>
                <span className="text-muted">Linhas duplicadas </span>
                {fmtInt(profile.data.duplicated_rows)}
              </span>
            </div>
          )}
          <Segmented
            value={tab}
            onChange={setTab}
            options={[
              { value: 'vars', label: 'Variáveis e estatísticas' },
              { value: 'quality', label: 'Qualidade dos dados' },
              { value: 'table', label: 'Tabela' },
            ]}
          />
          {profile.loading && <Loading label="Calculando perfil do dataset…" />}
          <ErrorBox error={profile.error} />
          {tab === 'table' && <DataTable datasetId={dataset.id} />}
          {tab === 'vars' && profile.data && <Variables profile={profile.data} />}
          {tab === 'quality' && profile.data && <Quality profile={profile.data} />}
        </div>
      )}
      {!dataset && datasets.length > 0 && <Empty title="Selecione um dataset" />}
    </>
  )
}
