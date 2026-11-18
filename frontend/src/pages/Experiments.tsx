import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { ExperimentSummary } from '../api/types'
import { RocPr } from '../components/EvalCharts'
import { Plot } from '../components/Plot'
import { Badge, Button, cx, Edu, Empty, ErrorBox, Loading, Notice, PageHeader, Panel, Select, Table } from '../components/ui'
import { METRICS } from '../lib/education'
import { fmt, fmtDate, fmtSeconds } from '../lib/format'
import { useAsync } from '../lib/useAsync'

const LOWER = new Set(['mae', 'rmse', 'mse', 'mape', 'log_loss'])
const CLS = ['f1', 'recall', 'precision', 'balanced_accuracy', 'accuracy', 'roc_auc', 'log_loss']
const REG = ['rmse', 'mae', 'r2', 'mape']

function Comparison({ ids, summaries }: { ids: string[]; summaries: ExperimentSummary[] }) {
  const sel = summaries.filter((s) => ids.includes(s.id))
  const sameTask = new Set(sel.map((s) => s.task)).size === 1
  const task = sel[0]?.task
  const metricOpts = (task === 'classification' ? CLS : REG).filter((m) => sel.some((s) => s.metrics.test[m] !== undefined))
  const [metric, setMetric] = useState(metricOpts[0])
  useEffect(() => {
    if (!metricOpts.includes(metric)) setMetric(metricOpts[0])
  }, [metricOpts, metric])
  const cmp = useAsync(() => api.compare(ids, metric), [ids.join(), metric], sameTask && !!metric)

  if (!sameTask) return <Notice level="warn">Selecione experimentos da mesma tarefa (só classificação ou só regressão) para comparar métricas.</Notice>

  const best = (k: string) => {
    const vals = sel.map((s) => s.metrics.test[k]).filter((v) => v !== undefined)
    return LOWER.has(k) ? Math.min(...vals) : Math.max(...vals)
  }
  const allParams = Array.from(new Set(sel.flatMap((s) => Object.keys(s.params))))
  const exps = cmp.data?.experiments ?? []
  const mName = METRICS[metric]?.name ?? metric

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="font-display text-[22px] font-semibold">Comparação de {sel.length} experimentos</h2>
        <div className="ml-auto w-56">
          <Select ariaLabel="Métrica" value={metric} onChange={setMetric} options={metricOpts.map((m) => ({ value: m, label: METRICS[m]?.name ?? m }))} />
        </div>
      </div>
      {new Set(sel.map((s) => `${s.dataset_id}:${s.target}`)).size > 1 && (
        <Notice level="warn">Os experimentos usam datasets ou alvos diferentes: as métricas não são diretamente comparáveis.</Notice>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title={`${mName}: teste e validação cruzada`} subtitle={LOWER.has(metric) ? 'Menor é melhor' : 'Maior é melhor'}>
          <Plot
            deps={[ids.join(), metric]}
            height={Math.max(260, sel.length * 56 + 90)}
            build={(c) => ({
              data: [
                {
                  type: 'bar',
                  orientation: 'h',
                  name: 'Validação cruzada (média ± dp)',
                  y: sel.map((s) => s.name),
                  x: sel.map((s) => s.cv?.metrics[metric]?.mean ?? null),
                  error_x: { type: 'data', array: sel.map((s) => s.cv?.metrics[metric]?.std ?? 0), color: c.muted, thickness: 1.2 },
                  marker: { color: c.series[1] },
                },
                { type: 'bar', orientation: 'h', name: 'Teste', y: sel.map((s) => s.name), x: sel.map((s) => s.metrics.test[metric] ?? null), marker: { color: c.series[0] } },
                {
                  type: 'scatter',
                  mode: 'markers',
                  name: 'Baseline',
                  y: sel.map((s) => s.name),
                  x: sel.map((s) => s.metrics.baseline[metric] ?? null),
                  marker: { color: c.ink, symbol: 'line-ns', size: 18, line: { width: 2, color: c.ink } },
                },
              ],
              layout: { barmode: 'group', yaxis: { autorange: 'reversed' }, xaxis: { title: mName }, margin: { l: 200, r: 20, t: 10, b: 70 } },
            })}
          />
        </Panel>
        <Panel title="Tempo de treinamento" subtitle="Ajuste final no treino completo e validação cruzada">
          <Plot
            deps={[ids.join()]}
            height={Math.max(260, sel.length * 56 + 90)}
            build={(c) => ({
              data: [
                { type: 'bar', orientation: 'h', name: 'Ajuste final', y: sel.map((s) => s.name), x: sel.map((s) => s.timing.fit_s), marker: { color: c.series[0] } },
                { type: 'bar', orientation: 'h', name: 'Validação cruzada', y: sel.map((s) => s.name), x: sel.map((s) => s.timing.cv_s), marker: { color: c.series[3] } },
              ],
              layout: { barmode: 'group', yaxis: { autorange: 'reversed' }, xaxis: { title: 'segundos' }, margin: { l: 200, r: 20, t: 10, b: 70 } },
            })}
          />
        </Panel>
      </div>

      <Panel title="Métricas no teste" subtitle="Melhor valor de cada linha em destaque">
        <Table
          head={['Métrica', ...sel.map((s) => s.name)]}
          align={['left', ...sel.map(() => 'right' as const)]}
          rows={[
            ...(task === 'classification' ? CLS : REG)
              .filter((k) => sel.some((s) => s.metrics.test[k] !== undefined))
              .map((k) => [
                METRICS[k]?.name ?? k,
                ...sel.map((s) => {
                  const v = s.metrics.test[k]
                  return <span className={cx(v === best(k) && sel.length > 1 && 'font-semibold text-ok')}>{fmt(v)}</span>
                }),
              ]),
            ['Algoritmo', ...sel.map((s) => s.model_name)],
            ['Atributos', ...sel.map((s) => String(s.n_features))],
            ['Treino / teste', ...sel.map((s) => `${s.n_train} / ${s.n_test}`)],
            ['Tempo total', ...sel.map((s) => fmtSeconds(s.timing.total_s))],
          ]}
        />
      </Panel>

      <Panel title="Hiperparâmetros" subtitle="Em âmbar: parâmetros compartilhados com valores diferentes. “—” = não se aplica ao algoritmo">
        <Table
          head={['Parâmetro', ...sel.map((s) => s.name)]}
          rows={allParams.map((p) => {
            const vals = sel.map((s) => (p in s.params ? String(s.params[p] ?? 'None') : '—'))
            const present = vals.filter((v) => v !== '—')
            const differs = present.length > 1 && new Set(present).size > 1
            return [<span className={differs ? 'text-warn' : ''}>{p}</span>, ...vals]
          })}
        />
      </Panel>

      {cmp.loading && <Loading label="Carregando detalhes e testes estatísticos…" />}
      <ErrorBox error={cmp.error} />
      {cmp.data && (
        <>
          <Panel title={`Teste de significância pareado (${mName}, folds da validação cruzada)`}>
            <Table
              head={['Experimento A', 'Experimento B', 'Diferença média (A − B)', 't corrigido', 'p corrigido', 'p (t-test comum)', 'Conclusão']}
              align={['left', 'left', 'right', 'right', 'right', 'right', 'left']}
              rows={cmp.data.pairs.map((p) =>
                p.available
                  ? [
                      p.a_name,
                      p.b_name,
                      fmt(p.mean_diff, 4),
                      fmt(p.t_corrected),
                      <b className={p.significant_5pct ? 'text-ok' : ''}>{fmt(p.p_corrected)}</b>,
                      <span className="text-muted">{fmt(p.p_standard)}</span>,
                      p.significant_5pct
                        ? `Diferença significativa: ${p.better === p.a ? 'A' : 'B'} é melhor`
                        : 'Sem evidência de diferença (α = 0,05)',
                    ]
                  : [p.a_name, p.b_name, '—', '—', '—', '—', <span className="text-muted">{p.reason}</span>],
              )}
            />
            {cmp.data.pairs.some((p) => p.note) && <p className="mt-2 text-[13px] text-muted">{cmp.data.pairs.find((p) => p.note)?.note}</p>}
            <div className="mt-3">
              <Edu concept="significance" />
            </div>
          </Panel>

          {exps.every((e) => e.cv?.metrics[metric]?.folds) && (
            <Panel title={`${mName} fold a fold`} subtitle="Mesmos folds para todos: as linhas são diretamente comparáveis">
              <Plot
                deps={[ids.join(), metric]}
                height={300}
                build={() => ({
                  data: exps.map((e) => ({
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: e.name,
                    x: e.cv!.metrics[metric].folds!.map((_, i) => `Fold ${i + 1}`),
                    y: e.cv!.metrics[metric].folds,
                  })),
                  layout: { yaxis: { title: mName } },
                })}
              />
            </Panel>
          )}
          {task === 'classification' && <RocPr exps={exps} />}
          {task === 'regression' && (
            <Panel title="Distribuição dos resíduos no teste">
              <Plot
                deps={[ids.join()]}
                height={320}
                build={() => ({
                  data: exps.map((e) => ({ type: 'box', name: e.name, y: e.results.residuals?.values, boxpoints: false })),
                  layout: { yaxis: { title: 'Resíduo (real − previsto)' }, showlegend: false },
                })}
              />
            </Panel>
          )}
        </>
      )}
    </div>
  )
}

export default function ExperimentsPage() {
  const list = useAsync(() => api.experiments(), [])
  const [params, setParams] = useSearchParams()
  const [selected, setSelected] = useState<string[]>(() => params.get('compare')?.split(',').filter(Boolean) ?? [])
  const [dsFilter, setDsFilter] = useState('')
  const [taskFilter, setTaskFilter] = useState('')

  useEffect(() => {
    if (selected.length) setParams({ compare: selected.join(',') }, { replace: true })
    else setParams({}, { replace: true })
  }, [selected, setParams])

  const datasets = useMemo(() => Array.from(new Map((list.data ?? []).map((e) => [e.dataset_id, e.dataset_name])).entries()), [list.data])
  const rows = (list.data ?? []).filter((e) => (!dsFilter || e.dataset_id === dsFilter) && (!taskFilter || e.task === taskFilter))

  async function rename(e: ExperimentSummary) {
    const name = prompt('Novo nome do experimento', e.name)
    if (!name) return
    await api.patchExperiment(e.id, { name })
    list.reload()
  }
  async function remove(e: ExperimentSummary) {
    if (!confirm(`Excluir "${e.name}"?`)) return
    await api.deleteExperiment(e.id)
    setSelected((s) => s.filter((x) => x !== e.id))
    list.reload()
  }

  return (
    <>
      <PageHeader title="Experimentos">
        Histórico de todas as execuções. Selecione dois ou mais para comparar métricas, hiperparâmetros, tempo e gráficos.
      </PageHeader>
      <ErrorBox error={list.error} />
      {list.loading && !list.data && <Loading />}
      {list.data && list.data.length === 0 && (
        <Empty title="Nenhum experimento salvo" action={<Link className="text-accent underline" to="/train">Treinar um modelo</Link>} />
      )}
      {list.data && list.data.length > 0 && (
        <div className="flex flex-col gap-6">
          <Panel
            pad={false}
            title={`${rows.length} experimento(s)`}
            actions={
              <>
                <div className="w-52">
                  <Select ariaLabel="Filtrar dataset" value={dsFilter} onChange={setDsFilter} options={[{ value: '', label: 'Todos os datasets' }, ...datasets.map(([v, l]) => ({ value: v, label: l }))]} />
                </div>
                <div className="w-44">
                  <Select
                    ariaLabel="Filtrar tarefa"
                    value={taskFilter}
                    onChange={setTaskFilter}
                    options={[
                      { value: '', label: 'Todas as tarefas' },
                      { value: 'classification', label: 'Classificação' },
                      { value: 'regression', label: 'Regressão' },
                    ]}
                  />
                </div>
                {selected.length > 0 && (
                  <Button size="sm" variant="ghost" onClick={() => setSelected([])}>
                    Limpar seleção
                  </Button>
                )}
              </>
            }
          >
            <div className="overflow-x-auto">
              <table className="num w-full border-collapse text-[13.5px]">
                <thead>
                  <tr className="bg-panel-2 text-left text-muted">
                    <th className="border-b border-line px-3 py-1.5" />
                    <th className="border-b border-line px-3 py-1.5 font-semibold">Experimento</th>
                    <th className="border-b border-line px-3 py-1.5 font-semibold">Dataset · alvo</th>
                    <th className="border-b border-line px-3 py-1.5 text-right font-semibold">Métrica principal (teste)</th>
                    <th className="border-b border-line px-3 py-1.5 text-right font-semibold">Validação cruzada</th>
                    <th className="border-b border-line px-3 py-1.5 text-right font-semibold">Ajuste</th>
                    <th className="border-b border-line px-3 py-1.5 font-semibold">Criado</th>
                    <th className="border-b border-line px-3 py-1.5" />
                  </tr>
                </thead>
                <tbody>
                  {rows.map((e) => {
                    const pm = e.primary_metric
                    const cv = e.cv?.metrics[pm]
                    const checked = selected.includes(e.id)
                    return (
                      <tr key={e.id} className={cx('border-b border-line/60', checked && 'bg-accent/6')}>
                        <td className="px-3 py-1.5">
                          <input
                            type="checkbox"
                            aria-label={`Selecionar ${e.name}`}
                            checked={checked}
                            onChange={(ev) => setSelected(ev.target.checked ? [...selected, e.id] : selected.filter((x) => x !== e.id))}
                          />
                        </td>
                        <td className="px-3 py-1.5">
                          <Link to={`/evaluate/${e.id}`} className="font-medium hover:text-accent">
                            {e.name}
                          </Link>
                          <div className="flex gap-1.5 pt-0.5">
                            <Badge>{e.model_name}</Badge>
                            <Badge>{e.task === 'classification' ? 'classificação' : 'regressão'}</Badge>
                          </div>
                        </td>
                        <td className="px-3 py-1.5 text-muted">
                          {e.dataset_name} · <span className="text-ink">{e.target}</span>
                        </td>
                        <td className="px-3 py-1.5 text-right">
                          <span className="text-muted">{METRICS[pm]?.name} </span>
                          <b>{fmt(e.metrics.test[pm])}</b>
                        </td>
                        <td className="px-3 py-1.5 text-right">{cv ? `${fmt(cv.mean)} ± ${fmt(cv.std)}` : '—'}</td>
                        <td className="px-3 py-1.5 text-right">{fmtSeconds(e.timing.fit_s)}</td>
                        <td className="px-3 py-1.5 whitespace-nowrap text-muted">{fmtDate(e.created_at)}</td>
                        <td className="px-3 py-1.5 whitespace-nowrap">
                          <Button size="sm" variant="ghost" onClick={() => rename(e)}>
                            Renomear
                          </Button>
                          <Button size="sm" variant="ghost" onClick={() => remove(e)}>
                            Excluir
                          </Button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Panel>
          {selected.length >= 2 ? (
            <Comparison ids={selected} summaries={list.data} />
          ) : (
            <p className="text-sm text-muted">Selecione pelo menos dois experimentos para ver a comparação.</p>
          )}
        </div>
      )}
    </>
  )
}
