import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { Filter, Profile } from '../api/types'
import { FilterBar } from '../components/FilterBar'
import { Plot } from '../components/Plot'
import { Edu, Empty, ErrorBox, Field, Loading, Notice, PageHeader, Panel, Segmented, Select, Table, Toggle } from '../components/ui'
import { useApp } from '../context/AppContext'
import { fmt, fmtInt } from '../lib/format'
import { useAsync } from '../lib/useAsync'
import { groupableColumns, numericColumns, useProfile } from '../lib/useProfile'

type Tab = 'hist' | 'box' | 'corr' | 'scatter' | 'dist'

/* eslint-disable @typescript-eslint/no-explicit-any */
function useEda(datasetId: string, body: Record<string, unknown> | null, filters: Filter[]) {
  return useAsync<any>(
    () => (body ? api.eda(datasetId, { ...body, filters }) : Promise.resolve(null)),
    [datasetId, JSON.stringify(body), JSON.stringify(filters)],
    !!body,
  )
}

const colOpts = (cols: string[], none?: string) => [
  ...(none !== undefined ? [{ value: '', label: none }] : []),
  ...cols.map((c) => ({ value: c, label: c })),
]

function Histogram({ id, profile, filters, onRows }: TabProps) {
  const all = profile.columns.filter((c) => c.semantic_type !== 'texto_identificador' && c.semantic_type !== 'data_hora').map((c) => c.name)
  const num = numericColumns(profile, false)
  const [column, setColumn] = useState(num[0] ?? all[0])
  const [bins, setBins] = useState(30)
  const [colorBy, setColorBy] = useState('')
  const [pct, setPct] = useState(false)
  const r = useEda(id, { kind: 'histogram', column, bins, color_by: colorBy || null }, filters)
  useEffect(() => {
    if (r.data) onRows(r.data.rows_filtered)
  }, [r.data, onRows])
  const d = r.data
  return (
    <Panel title="Histograma">
      <div className="mb-3 grid gap-3 sm:grid-cols-4">
        <Field label="Coluna">
          <Select value={column} onChange={setColumn} options={colOpts(all)} />
        </Field>
        <Field label={`Intervalos (bins): ${bins}`}>
          <input type="range" min={5} max={100} value={bins} onChange={(e) => setBins(+e.target.value)} className="mt-2 w-full" />
        </Field>
        <Field label="Separar por">
          <Select value={colorBy} onChange={setColorBy} options={colOpts(groupableColumns(profile).filter((c) => c !== column), 'Nenhum')} />
        </Field>
        <div className="pt-6">
          <Toggle checked={pct} onChange={setPct} label="Proporção por grupo" hint="Compara formatos com tamanhos diferentes" />
        </div>
      </div>
      <ErrorBox error={r.error} />
      {r.loading && !d && <Loading />}
      {d?.kind === 'numeric' && (
        <Plot
          deps={[d, pct]}
          height={380}
          build={() => {
            const centers = d.edges.slice(0, -1).map((e: number, i: number) => (e + d.edges[i + 1]) / 2)
            const width = d.edges[1] - d.edges[0]
            return {
              data: d.series.map((s: any) => ({
                type: 'bar',
                name: `${s.name} (n=${fmtInt(s.n)})`,
                x: centers,
                y: pct ? s.counts.map((v: number) => (s.n ? (v / s.n) * 100 : 0)) : s.counts,
                width,
                opacity: d.series.length > 1 ? 0.6 : 0.9,
                hovertemplate: `%{x:.3~f}: %{y:.2~f}${pct ? '%' : ''}<extra>${s.name}</extra>`,
              })),
              layout: {
                barmode: 'overlay',
                bargap: 0.04,
                xaxis: { title: column },
                yaxis: { title: pct ? '% do grupo' : 'Frequência' },
                showlegend: d.series.length > 1,
              },
            }
          }}
        />
      )}
      {d?.kind === 'categorical' && (
        <Plot
          deps={[d]}
          build={(c) => ({
            data: [{ type: 'bar', x: d.categories, y: d.counts, marker: { color: c.accent }, hovertemplate: '%{x}: %{y}<extra></extra>' }],
            layout: { xaxis: { title: column, type: 'category' }, yaxis: { title: 'Contagem' } },
          })}
        />
      )}
      <div className="mt-2">
        <Edu title="Lendo o histograma">
          Procure o formato (simétrico, cauda à direita, bimodal), valores fora do padrão e, ao separar pelo alvo, se as distribuições das
          classes se afastam: quanto menos sobreposição, mais o atributo ajuda a discriminar.
        </Edu>
      </div>
    </Panel>
  )
}

function Boxplots({ id, profile, filters, onRows }: TabProps) {
  const num = numericColumns(profile, false)
  const [cols, setCols] = useState<string[]>(num.slice(0, 4))
  const [groupBy, setGroupBy] = useState('')
  const r = useEda(id, cols.length ? { kind: 'boxplot', columns: cols, color_by: groupBy || null } : null, filters)
  useEffect(() => {
    if (r.data) onRows(r.data.rows_filtered)
  }, [r.data, onRows])
  const byCol = useMemo(() => {
    const m: Record<string, any[]> = {}
    r.data?.boxes.forEach((b: any) => (m[b.column] ||= []).push(b))
    return m
  }, [r.data])
  return (
    <Panel title="Boxplots" subtitle="Quartis calculados sobre todas as linhas filtradas; cada coluna tem sua própria escala">
      <div className="mb-3 flex flex-wrap items-end gap-4">
        <Field label="Separar por" className="w-56">
          <Select value={groupBy} onChange={setGroupBy} options={colOpts(groupableColumns(profile), 'Nenhum')} />
        </Field>
        <div className="flex max-w-3xl flex-wrap gap-1.5">
          {num.map((c) => (
            <label key={c} className="flex items-center gap-1 rounded border border-line px-2 py-0.5 text-[13px]">
              <input type="checkbox" checked={cols.includes(c)} onChange={(e) => setCols(e.target.checked ? [...cols, c] : cols.filter((x) => x !== c))} />
              {c}
            </label>
          ))}
        </div>
      </div>
      <ErrorBox error={r.error} />
      {r.loading && !r.data && <Loading />}
      <div className="grid gap-4 md:grid-cols-2">
        {cols.filter((c) => byCol[c]).map((col) => (
          <div key={col} className="rounded border border-line bg-bg p-2">
            <div className="px-1 text-[13.5px] font-semibold">{col}</div>
            <Plot
              deps={[byCol[col]]}
              height={260}
              build={(c) => {
                const boxes = byCol[col]
                return {
                  data: [
                    ...boxes.map((b: any, i: number) => ({
                      type: 'box',
                      name: b.group,
                      x: [b.group],
                      q1: [b.q1],
                      median: [b.median],
                      q3: [b.q3],
                      lowerfence: [b.lowerfence],
                      upperfence: [b.upperfence],
                      mean: [b.mean],
                      boxpoints: false,
                      marker: { color: c.series[i % 6] },
                      line: { width: 1.5 },
                      hoverinfo: 'y',
                    })),
                    ...boxes.map((b: any) => ({
                      type: 'scatter',
                      mode: 'markers',
                      x: b.outliers.map(() => b.group),
                      y: b.outliers,
                      marker: { color: c.warn, size: 4, opacity: 0.6 },
                      name: `outliers ${b.group}`,
                      hovertemplate: `%{y}<extra>outlier (${fmtInt(b.n_outliers)} no total)</extra>`,
                    })),
                  ],
                  layout: { showlegend: false, margin: { l: 50, r: 10, t: 8, b: 30 }, xaxis: { type: 'category' } },
                }
              }}
            />
          </div>
        ))}
      </div>
      <div className="mt-3">
        <Edu title="Anatomia do boxplot">
          A caixa vai do 1º ao 3º quartil (50% centrais dos dados), a linha interna é a mediana e as hastes chegam ao último valor dentro de
          1,5 × IQR. Pontos âmbar estão além disso — candidatos a outlier.
        </Edu>
      </div>
    </Panel>
  )
}

function Correlation({ id, filters, onRows }: TabProps) {
  const [method, setMethod] = useState<'pearson' | 'spearman' | 'kendall'>('pearson')
  const r = useEda(id, { kind: 'correlation', method }, filters)
  useEffect(() => {
    if (r.data) onRows(r.data.rows_filtered)
  }, [r.data, onRows])
  const d = r.data
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <Panel
        title="Matriz de correlação"
        actions={
          <Segmented
            value={method}
            onChange={setMethod}
            options={[
              { value: 'pearson', label: 'Pearson' },
              { value: 'spearman', label: 'Spearman' },
              { value: 'kendall', label: 'Kendall' },
            ]}
          />
        }
      >
        <ErrorBox error={r.error} />
        {r.loading && !d && <Loading />}
        {d && (
          <Plot
            deps={[d]}
            height={Math.max(380, d.columns.length * 34 + 140)}
            build={(c) => ({
              data: [
                {
                  type: 'heatmap',
                  z: d.matrix,
                  x: d.columns,
                  y: d.columns,
                  zmin: -1,
                  zmax: 1,
                  colorscale: [
                    [0, c.alarm],
                    [0.5, c.panel],
                    [1, c.accent],
                  ],
                  text: d.columns.length <= 16 ? d.matrix.map((row: number[]) => row.map((v) => (v == null ? '' : v.toFixed(2)))) : undefined,
                  texttemplate: d.columns.length <= 16 ? '%{text}' : undefined,
                  textfont: { size: 11, color: c.ink },
                  hovertemplate: '%{y} × %{x}: %{z:.3f}<extra></extra>',
                  colorbar: { thickness: 10, tickfont: { color: c.muted } },
                },
              ],
              layout: { margin: { l: 150, r: 20, t: 10, b: 130 }, yaxis: { autorange: 'reversed' }, xaxis: { tickangle: -40 } },
            })}
          />
        )}
      </Panel>
      <div className="flex flex-col gap-4">
        <Panel title="Pares mais correlacionados">
          {d && (
            <Table
              head={['Variável A', 'Variável B', 'r']}
              align={['left', 'left', 'right']}
              rows={d.top_pairs.map((p: any) => [p.a, p.b, <span className={Math.abs(p.r) > 0.8 ? 'font-semibold text-warn' : ''}>{fmt(p.r)}</span>])}
            />
          )}
        </Panel>
        <Edu title="Qual coeficiente usar">
          Pearson mede relação linear e é sensível a outliers. Spearman e Kendall usam postos: captam qualquer relação monotônica (ex.:
          desgaste que acelera com o tempo). Correlação alta entre dois atributos (multicolinearidade) deixa coeficientes de modelos lineares
          instáveis; correlação com o alvo perto de ±1 merece suspeita de vazamento.
        </Edu>
      </div>
    </div>
  )
}

function Scatter({ id, profile, filters, onRows }: TabProps) {
  const num = numericColumns(profile)
  const [x, setX] = useState(num[0])
  const [y, setY] = useState(num[1] ?? num[0])
  const [colorBy, setColorBy] = useState('')
  const [trend, setTrend] = useState(true)
  const r = useEda(id, x && y ? { kind: 'scatter', x, y, color_by: colorBy || null } : null, filters)
  useEffect(() => {
    if (r.data) onRows(r.data.rows_filtered)
  }, [r.data, onRows])
  const d = r.data
  return (
    <Panel title="Gráfico de dispersão">
      <div className="mb-3 grid gap-3 sm:grid-cols-4">
        <Field label="Eixo X">
          <Select value={x} onChange={setX} options={colOpts(num)} />
        </Field>
        <Field label="Eixo Y">
          <Select value={y} onChange={setY} options={colOpts(num)} />
        </Field>
        <Field label="Cor por">
          <Select value={colorBy} onChange={setColorBy} options={colOpts(groupableColumns(profile), 'Nenhum')} />
        </Field>
        <div className="pt-6">
          <Toggle checked={trend} onChange={setTrend} label="Reta de tendência (MQO)" />
        </div>
      </div>
      <ErrorBox error={r.error} />
      {r.loading && !d && <Loading />}
      {d && (
        <>
          <div className="num mb-2 flex flex-wrap gap-x-6 text-[13.5px]">
            <span>
              <span className="text-muted">Pearson r </span>
              {fmt(d.pearson_r)} <span className="text-muted">(p = {fmt(d.pearson_p)})</span>
            </span>
            <span>
              <span className="text-muted">Spearman ρ </span>
              {fmt(d.spearman_rho)}
            </span>
            {d.trendline && (
              <span>
                <span className="text-muted">Reta </span>y = {fmt(d.trendline.slope)}·x + {fmt(d.trendline.intercept)}
              </span>
            )}
            <span className="text-muted">
              {d.sampled ? `${fmtInt(d.n_plotted)} de ${fmtInt(d.n_total)} pontos desenhados (amostra); estatísticas usam todos` : `${fmtInt(d.n_total)} pontos`}
            </span>
          </div>
          <Plot
            deps={[d, trend]}
            height={440}
            build={(c) => ({
              data: [
                ...d.series.map((s: any) => ({
                  type: 'scattergl',
                  mode: 'markers',
                  name: s.name,
                  x: s.x,
                  y: s.y,
                  marker: { size: 5, opacity: 0.55 },
                })),
                ...(trend && d.trendline
                  ? [{ type: 'scatter', mode: 'lines', name: 'Tendência', x: d.trendline.x, y: d.trendline.y, line: { color: c.ink, width: 1.5, dash: 'dash' } }]
                  : []),
              ],
              layout: { xaxis: { title: x }, yaxis: { title: y }, showlegend: d.series.length > 1 || trend },
            })}
          />
        </>
      )}
    </Panel>
  )
}

function Distribution({ id, profile, filters, onRows }: TabProps) {
  const num = numericColumns(profile, false)
  const [column, setColumn] = useState(num[0])
  const r = useEda(id, column ? { kind: 'distribution', column } : null, filters)
  const h = useEda(id, column ? { kind: 'histogram', column, bins: 40 } : null, filters)
  useEffect(() => {
    if (r.data) onRows(r.data.rows_filtered)
  }, [r.data, onRows])
  const d = r.data
  return (
    <div className="flex flex-col gap-4">
      <div className="w-64">
        <Field label="Coluna">
          <Select value={column} onChange={setColumn} options={colOpts(num)} />
        </Field>
      </div>
      <ErrorBox error={r.error} />
      {(r.loading || h.loading) && !d && <Loading />}
      {d && h.data?.kind === 'numeric' && (
        <>
          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="Histograma com densidade (KDE)">
              <Plot
                deps={[d, h.data]}
                build={(c) => {
                  const e = h.data.edges
                  const w = e[1] - e[0]
                  const n = h.data.series[0].n
                  return {
                    data: [
                      {
                        type: 'bar',
                        x: e.slice(0, -1).map((v: number, i: number) => (v + e[i + 1]) / 2),
                        y: h.data.series[0].counts.map((v: number) => v / (n * w)),
                        width: w,
                        marker: { color: c.accent, opacity: 0.45 },
                        name: 'Histograma (densidade)',
                        hovertemplate: '%{x:.3~f}: %{y:.4f}<extra></extra>',
                      },
                      ...(d.kde ? [{ type: 'scatter', mode: 'lines', x: d.kde.x, y: d.kde.y, line: { color: c.warn, width: 2 }, name: 'KDE' }] : []),
                    ],
                    layout: { xaxis: { title: column }, yaxis: { title: 'Densidade' }, bargap: 0.03 },
                  }
                }}
              />
            </Panel>
            <Panel title="Q-Q plot contra a normal">
              <Plot
                deps={[d]}
                build={(c) => {
                  const q = d.qq
                  const lo = Math.min(q.theoretical[0], q.sample[0])
                  const hi = Math.max(q.theoretical.at(-1), q.sample.at(-1))
                  return {
                    data: [
                      { type: 'scatter', mode: 'markers', x: q.theoretical, y: q.sample, marker: { color: c.accent, size: 4 }, name: 'Quantis' },
                      { type: 'scatter', mode: 'lines', x: [lo, hi], y: [lo, hi], line: { color: c.muted, dash: 'dash', width: 1 }, name: 'Normal perfeita' },
                    ],
                    layout: { xaxis: { title: 'Quantis teóricos (normal)' }, yaxis: { title: 'Quantis da amostra' }, showlegend: false },
                  }
                }}
              />
            </Panel>
          </div>
          <Panel title="Resumo da distribuição">
            <div className="num mb-3 flex flex-wrap gap-x-8 gap-y-1 text-[14px]">
              <span>
                <span className="text-muted">n </span>
                {fmtInt(d.n)}
              </span>
              <span>
                <span className="text-muted">Assimetria </span>
                {fmt(d.skewness)}
              </span>
              <span>
                <span className="text-muted">Excesso de curtose </span>
                {fmt(d.excess_kurtosis)}
              </span>
              <span>
                <span className="text-muted">Formato </span>
                {d.shape}
              </span>
            </div>
            <Table
              head={['Teste de normalidade', 'Estatística', 'p-valor', 'Conclusão (α = 0,05)']}
              align={['left', 'right', 'right', 'left']}
              rows={d.normality_tests.map((t: any) => [
                t.name,
                fmt(t.statistic),
                fmt(t.p_value),
                t.p_value < 0.05 ? 'Rejeita normalidade' : 'Não rejeita normalidade',
              ])}
            />
            {d.suggest_log_transform && (
              <div className="mt-3">
                <Notice level="info">Assimetria forte à direita com valores positivos: uma transformação log pode aproximar a distribuição da normal.</Notice>
              </div>
            )}
            <div className="mt-3">
              <Edu title="Normalidade importa?">
                Árvores, Random Forest e XGBoost não assumem normalidade. Ela importa mais para modelos lineares (resíduos) e testes
                estatísticos. Com muitas amostras, os testes rejeitam normalidade por desvios pequenos — olhe o Q-Q plot: pontos sobre a reta
                indicam normal; curvatura nas pontas indica caudas pesadas.
              </Edu>
            </div>
          </Panel>
        </>
      )}
    </div>
  )
}

interface TabProps {
  id: string
  profile: Profile
  filters: Filter[]
  onRows: (n: number) => void
}

export default function EDAPage() {
  const { datasetId, dataset } = useApp()
  const profile = useProfile(datasetId)
  const [tab, setTab] = useState<Tab>('hist')
  const [filters, setFilters] = useState<Filter[]>([])
  const [rows, setRows] = useState<number | undefined>()
  useEffect(() => {
    setFilters([])
  }, [datasetId])

  if (!datasetId || !dataset)
    return (
      <>
        <PageHeader title="Análise exploratória" />
        <Empty title="Nenhum dataset ativo" action={<Link className="text-accent underline" to="/datasets">Carregar um dataset</Link>} />
      </>
    )

  return (
    <>
      <PageHeader title="Análise exploratória">
        Gráficos calculados no servidor sobre as linhas filtradas de <b className="text-ink">{dataset.name}</b>. Os filtros valem para todas as abas.
      </PageHeader>
      {profile.loading && <Loading />}
      <ErrorBox error={profile.error} />
      {profile.data && (
        <div className="flex flex-col gap-4">
          <FilterBar profile={profile.data} filters={filters} onChange={setFilters} rowsFiltered={rows} rowsTotal={profile.data.n_rows} />
          <Segmented
            value={tab}
            onChange={setTab}
            options={[
              { value: 'hist', label: 'Histograma' },
              { value: 'box', label: 'Boxplot' },
              { value: 'corr', label: 'Correlação' },
              { value: 'scatter', label: 'Dispersão' },
              { value: 'dist', label: 'Distribuição' },
            ]}
          />
          {(() => {
            const props = { id: datasetId, profile: profile.data, filters, onRows: setRows }
            switch (tab) {
              case 'hist':
                return <Histogram key={datasetId} {...props} />
              case 'box':
                return <Boxplots key={datasetId} {...props} />
              case 'corr':
                return <Correlation key={datasetId} {...props} />
              case 'scatter':
                return <Scatter key={datasetId} {...props} />
              case 'dist':
                return <Distribution key={datasetId} {...props} />
            }
          })()}
        </div>
      )}
    </>
  )
}
