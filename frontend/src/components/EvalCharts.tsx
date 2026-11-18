import { useMemo, useState } from 'react'
import type { Experiment } from '../api/types'
import { METRICS } from '../lib/education'
import { featureLabel, fmt, fmtInt, fmtPct } from '../lib/format'
import { bestF1, metricsAt, sweep } from '../lib/threshold'
import { Plot } from './Plot'
import { Edu, Notice, Panel, Readout, Segmented, Table } from './ui'

/* eslint-disable @typescript-eslint/no-explicit-any */

export function ConfusionMatrix({ exp }: { exp: Experiment }) {
  const [norm, setNorm] = useState<'count' | 'row'>('count')
  const cm = exp.results.confusion_matrix!
  const labels = exp.results.class_labels!
  const rowSums = cm.map((r) => r.reduce((a, b) => a + b, 0))
  const z = norm === 'row' ? cm.map((r, i) => r.map((v) => (rowSums[i] ? v / rowSums[i] : 0))) : cm
  const binary = labels.length === 2
  return (
    <Panel
      title="Matriz de confusão (teste)"
      actions={
        <Segmented
          value={norm}
          onChange={setNorm}
          options={[
            { value: 'count', label: 'Contagens' },
            { value: 'row', label: '% por classe real' },
          ]}
        />
      }
    >
      <Plot
        deps={[exp.id, norm]}
        height={Math.max(300, labels.length * 70 + 120)}
        build={(c) => ({
          data: [
            {
              type: 'heatmap',
              z,
              x: labels.map((l) => `prev. ${l}`),
              y: labels.map((l) => `real ${l}`),
              colorscale: [
                [0, c.panel],
                [1, c.accent],
              ],
              showscale: false,
              text: cm.map((r, i) => r.map((v) => `${fmtInt(v)}<br>${fmtPct(rowSums[i] ? (v / rowSums[i]) * 100 : 0)}`)),
              texttemplate: '%{text}',
              textfont: { size: 14, color: c.ink },
              hovertemplate: '%{y} → %{x}<br>%{text}<extra></extra>',
              xgap: 2,
              ygap: 2,
            },
          ],
          layout: { yaxis: { autorange: 'reversed' }, xaxis: { side: 'top' }, margin: { l: 110, r: 10, t: 40, b: 20 } },
        })}
      />
      {binary && (
        <div className="num mt-2 grid grid-cols-2 gap-2 text-[13px] sm:grid-cols-4">
          <span>
            <span className="text-muted">VN </span>
            {fmtInt(cm[0][0])}
          </span>
          <span>
            <span className="text-muted">FP (falso alarme) </span>
            {fmtInt(cm[0][1])}
          </span>
          <span>
            <span className="text-muted">FN (não detectado) </span>
            <span className={cm[1][0] ? 'text-alarm' : ''}>{fmtInt(cm[1][0])}</span>
          </span>
          <span>
            <span className="text-muted">VP </span>
            {fmtInt(cm[1][1])}
          </span>
        </div>
      )}
      <div className="mt-3">
        <Edu title="Como ler">
          Linhas são a classe real, colunas a prevista. A diagonal são os acertos. Em manutenção, o canto “real falha → previsto normal”
          (falso negativo) costuma ser o erro mais caro.
        </Edu>
      </div>
    </Panel>
  )
}

export function RocPr({ exps }: { exps: Experiment[] }) {
  const withRoc = exps.filter((e) => e.results.roc_curve)
  if (!withRoc.length) return null
  const multi = withRoc.length > 1
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Panel title="Curva ROC (teste)">
        <Plot
          height={multi ? 340 + withRoc.length * 22 : 340}
          deps={[withRoc.map((e) => e.id).join()]}
          build={(c) => ({
            data: [
              ...withRoc.map((e, i) => ({
                type: 'scatter',
                mode: 'lines',
                x: e.results.roc_curve!.fpr,
                y: e.results.roc_curve!.tpr,
                name: `${multi ? e.name : 'Modelo'} (AUC ${fmt(e.results.roc_curve!.auc)})`,
                line: { color: c.series[i % 6], width: 2, shape: 'hv' },
              })),
              { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], name: 'Aleatório', line: { color: c.muted, dash: 'dash', width: 1 } },
            ],
            layout: { xaxis: { title: 'Taxa de falsos positivos', range: [0, 1] }, yaxis: { title: 'Taxa de verdadeiros positivos (recall)', range: [0, 1.02] }, legend: { orientation: 'h', y: -0.25 } },
          })}
        />
      </Panel>
      <Panel title="Curva Precision-Recall (teste)">
        <Plot
          height={multi ? 340 + withRoc.length * 22 : 340}
          deps={[withRoc.map((e) => e.id).join()]}
          build={(c) => ({
            data: [
              ...withRoc.map((e, i) => ({
                type: 'scatter',
                mode: 'lines',
                x: e.results.pr_curve!.recall,
                y: e.results.pr_curve!.precision,
                name: multi ? e.name : 'Modelo',
                line: { color: c.series[i % 6], width: 2, shape: 'hv' },
              })),
              {
                type: 'scatter',
                mode: 'lines',
                x: [0, 1],
                y: [withRoc[0].results.pr_curve!.prevalence, withRoc[0].results.pr_curve!.prevalence],
                name: `Aleatório (prevalência ${fmtPct(withRoc[0].results.pr_curve!.prevalence * 100)})`,
                line: { color: c.muted, dash: 'dash', width: 1 },
              },
            ],
            layout: { xaxis: { title: 'Recall', range: [0, 1] }, yaxis: { title: 'Precision', range: [0, 1.02] }, legend: { orientation: 'h', y: -0.25 } },
          })}
        />
        {!multi && (
          <div className="mt-2">
            <Edu title="Por que olhar a curva PR">
              Com classes raras a ROC pode parecer boa mesmo com muitos falsos alarmes, porque os negativos são muitos. A curva PR foca na
              classe positiva: a linha tracejada é o desempenho de um palpite aleatório.
            </Edu>
          </div>
        )}
      </Panel>
    </div>
  )
}

export function ThresholdExplorer({ exp }: { exp: Experiment }) {
  const td = exp.results.threshold_data!
  const [source, setSource] = useState<'oof' | 'test'>(td.oof ? 'oof' : 'test')
  const [t, setT] = useState(0.5)
  const data = source === 'oof' && td.oof ? td.oof : td.test
  const curve = useMemo(() => sweep(data.y, data.p), [data])
  const best = useMemo(() => bestF1(curve), [curve])
  const m = metricsAt(data.y, data.p, t)
  const bestOnTest = td.oof ? metricsAt(td.test.y, td.test.p, bestF1(sweep(td.oof.y, td.oof.p)).t) : null
  const default05 = metricsAt(td.test.y, td.test.p, 0.5)

  return (
    <Panel
      title="Limiar de decisão"
      subtitle="Calculado no navegador a partir das probabilidades reais do modelo"
      actions={
        <Segmented
          value={source}
          onChange={setSource}
          options={[
            ...(td.oof ? [{ value: 'oof' as const, label: 'Validação (out-of-fold)' }] : []),
            { value: 'test' as const, label: 'Teste' },
          ]}
        />
      }
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Plot
          deps={[exp.id, source, t]}
          height={300}
          build={(c) => ({
            data: [
              { type: 'scatter', mode: 'lines', name: 'Precision', x: curve.map((r) => r.t), y: curve.map((r) => r.precision), line: { color: c.series[0], width: 2 } },
              { type: 'scatter', mode: 'lines', name: 'Recall', x: curve.map((r) => r.t), y: curve.map((r) => r.recall), line: { color: c.series[1], width: 2 } },
              { type: 'scatter', mode: 'lines', name: 'F1', x: curve.map((r) => r.t), y: curve.map((r) => r.f1), line: { color: c.ink, width: 2, dash: 'dot' } },
            ],
            layout: {
              xaxis: { title: 'Limiar', range: [0, 1] },
              yaxis: { range: [0, 1.02] },
              shapes: [{ type: 'line', x0: t, x1: t, y0: 0, y1: 1, line: { color: c.warn, width: 1.5 } }],
            },
          })}
        />
        <div className="flex flex-col gap-3">
          <label className="text-[13px] text-muted">
            Limiar: <b className="num text-ink">{t.toFixed(2)}</b>
            <input type="range" min={0.01} max={0.99} step={0.01} value={t} onChange={(e) => setT(+e.target.value)} className="mt-1 w-full" />
          </label>
          <div className="flex flex-wrap gap-2">
            <Readout label="Precision" value={fmt(m.precision)} />
            <Readout label="Recall" value={fmt(m.recall)} />
            <Readout label="F1" value={fmt(m.f1)} />
          </div>
          <div className="num text-[13px] text-muted">
            VP {fmtInt(m.tp)} · FP {fmtInt(m.fp)} · FN {fmtInt(m.fn)} · VN {fmtInt(m.tn)} · n = {fmtInt(data.y.length)}
            {data.sampled && ' (amostra)'}
          </div>
          <button className="self-start text-[13px] text-accent hover:underline" onClick={() => setT(best.t)}>
            Usar o limiar de maior F1 em {source === 'oof' ? 'validação' : 'teste'} ({best.t.toFixed(2)})
          </button>
        </div>
      </div>
      {bestOnTest && (
        <div className="mt-3">
          <Notice level="info" title="Limiar escolhido na validação, aplicado ao teste">
            O limiar que maximiza F1 nas previsões out-of-fold é {bestOnTest.t.toFixed(2)}. No teste ele resulta em F1 = {fmt(bestOnTest.f1)}{' '}
            (precision {fmt(bestOnTest.precision)}, recall {fmt(bestOnTest.recall)}), contra F1 = {fmt(default05.f1)} com o limiar padrão 0,50.
          </Notice>
        </div>
      )}
      {source === 'test' && td.oof && (
        <div className="mt-3">
          <Notice level="warn">Escolher o limiar olhando o teste vicia a avaliação. Use esta visão só para entender o comportamento; decida pela validação.</Notice>
        </div>
      )}
      <div className="mt-3">
        <Edu concept="threshold" />
      </div>
    </Panel>
  )
}

export function ProbaHist({ exp }: { exp: Experiment }) {
  const h = exp.results.proba_hist
  if (!h) return null
  const labels = exp.results.class_labels!
  return (
    <Panel title="Probabilidades previstas por classe real (teste)">
      <Plot
        deps={[exp.id]}
        height={280}
        build={(c) => ({
          data: [
            { type: 'histogram', x: h.negative, name: `real ${labels[0]}`, xbins: { start: 0, end: 1, size: 0.025 }, marker: { color: c.series[0] }, opacity: 0.6, histnorm: 'percent' },
            { type: 'histogram', x: h.positive, name: `real ${labels[1]}`, xbins: { start: 0, end: 1, size: 0.025 }, marker: { color: c.series[1] }, opacity: 0.6, histnorm: 'percent' },
          ],
          layout: { barmode: 'overlay', xaxis: { title: `P(${labels[1]})`, range: [0, 1] }, yaxis: { title: '% da classe' } },
        })}
      />
    </Panel>
  )
}

export function RegressionCharts({ exp }: { exp: Experiment }) {
  const pr = exp.results.predictions!
  const res = exp.results.residuals!
  const lo = Math.min(...pr.y_true, ...pr.y_pred)
  const hi = Math.max(...pr.y_true, ...pr.y_pred)
  return (
    <>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Previsto × real (teste)" subtitle={pr.sampled ? `${fmtInt(pr.y_true.length)} de ${fmtInt(pr.n)} pontos (amostra)` : `${fmtInt(pr.n)} pontos`}>
          <Plot
            deps={[exp.id]}
            height={360}
            build={(c) => ({
              data: [
                { type: 'scattergl', mode: 'markers', x: pr.y_true, y: pr.y_pred, marker: { color: c.accent, size: 5, opacity: 0.55 }, name: 'Teste' },
                { type: 'scatter', mode: 'lines', x: [lo, hi], y: [lo, hi], line: { color: c.muted, dash: 'dash', width: 1 }, name: 'Previsão perfeita' },
              ],
              layout: { xaxis: { title: `${exp.target} real` }, yaxis: { title: `${exp.target} previsto` }, showlegend: false },
            })}
          />
        </Panel>
        <Panel title="Resíduos × previsto" subtitle={`Média dos resíduos ${fmt(res.mean)} · desvio ${fmt(res.std)}`}>
          <Plot
            deps={[exp.id]}
            height={360}
            build={(c) => ({
              data: [{ type: 'scattergl', mode: 'markers', x: pr.y_pred, y: res.values, marker: { color: c.series[2], size: 5, opacity: 0.55 } }],
              layout: {
                xaxis: { title: 'Previsto' },
                yaxis: { title: 'Resíduo (real − previsto)' },
                shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: c.muted, dash: 'dash' } }],
              },
            })}
          />
        </Panel>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Distribuição dos resíduos">
          <Plot
            deps={[exp.id]}
            height={280}
            build={(c) => ({
              data: [{ type: 'histogram', x: res.values, nbinsx: 40, marker: { color: c.series[2] }, opacity: 0.8 }],
              layout: { xaxis: { title: 'Resíduo' }, yaxis: { title: 'Frequência' } },
            })}
          />
        </Panel>
        <Edu title="Lendo os resíduos">
          Bom sinal: pontos espalhados ao redor de zero, sem forma (funil, curva). Funil = erro cresce com o valor (heterocedasticidade);
          curva = relação não linear que o modelo não capturou; média longe de zero = viés sistemático.
        </Edu>
      </div>
      {exp.results.temporal_test && (
        <Panel title="Série no período de teste" subtitle="Ordem temporal: valores reais e previstos">
          <Plot
            deps={[exp.id]}
            height={300}
            build={(c) => ({
              data: [
                { type: 'scattergl', mode: 'lines', y: exp.results.temporal_test!.y_true, name: 'Real', line: { color: c.muted, width: 1 } },
                { type: 'scattergl', mode: 'lines', y: exp.results.temporal_test!.y_pred, name: 'Previsto', line: { color: c.accent, width: 1.3 } },
              ],
              layout: { xaxis: { title: 'Índice no teste (ordem temporal)' }, yaxis: { title: exp.target } },
            })}
          />
        </Panel>
      )}
    </>
  )
}

export function LearningCurve({ exp }: { exp: Experiment }) {
  const lc = exp.learning_curve
  if (!lc) return null
  const name = METRICS[lc.metric]?.name ?? lc.metric
  const band = (mean: number[], std: number[], color: string, label: string) => [
    { type: 'scatter', mode: 'lines', x: lc.train_sizes, y: mean.map((m, i) => m + std[i]), line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
    {
      type: 'scatter',
      mode: 'lines',
      x: lc.train_sizes,
      y: mean.map((m, i) => m - std[i]),
      line: { width: 0 },
      fill: 'tonexty',
      fillcolor: color + '33',
      showlegend: false,
      hoverinfo: 'skip',
    },
    { type: 'scatter', mode: 'lines+markers', x: lc.train_sizes, y: mean, name: label, line: { color, width: 2 } },
  ]
  return (
    <Panel title={`Curva de aprendizado (${name})`} subtitle="Média ± desvio entre folds para cada tamanho do treino">
      <Plot
        deps={[exp.id]}
        height={320}
        build={(c) => ({
          data: [...band(lc.train_mean, lc.train_std, c.series[0], 'Treino'), ...band(lc.val_mean, lc.val_std, c.series[1], 'Validação')],
          layout: { xaxis: { title: 'Amostras de treino' }, yaxis: { title: name } },
        })}
      />
      <div className="mt-2">
        <Edu concept="overfitting" />
      </div>
    </Panel>
  )
}

export function Importance({ exp }: { exp: Experiment }) {
  const blocks = [exp.importance, exp.permutation_importance].filter(Boolean) as { kind: string; features: string[]; values: number[]; std?: number[] }[]
  if (!blocks.length) return null
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {blocks.map((b) => (
        <Panel key={b.kind} title="Importância dos atributos" subtitle={b.kind}>
          <Plot
            deps={[exp.id, b.kind]}
            height={Math.min(560, b.features.length * 22 + 60)}
            build={(c) => ({
              data: [
                {
                  type: 'bar',
                  orientation: 'h',
                  y: b.features.map(featureLabel),
                  x: b.values,
                  error_x: b.std ? { type: 'data', array: b.std, color: c.muted, thickness: 1 } : undefined,
                  marker: { color: c.accent },
                  hovertemplate: '%{y}: %{x:.4g}<extra></extra>',
                },
              ],
              layout: { yaxis: { autorange: 'reversed' }, margin: { l: 170, r: 16, t: 6, b: 36 } },
            })}
          />
        </Panel>
      ))}
    </div>
  )
}

export function Coefficients({ exp }: { exp: Experiment }) {
  const co = exp.coefficients
  if (!co) return null
  const rows = co.features.map((f, j) => [featureLabel(f), ...co.rows.map((r) => fmt(r.values[j], 4))])
  return (
    <Panel title="Coeficientes do modelo linear" subtitle={`Intercepto: ${co.intercept.map((v) => fmt(v, 4)).join(' · ')}. Atributos na escala transformada.`}>
      <Table head={['Atributo', ...co.rows.map((r) => (co.rows.length > 1 ? r.label : 'Coeficiente'))]} align={['left', ...co.rows.map(() => 'right' as const)]} rows={rows} className="max-h-96 overflow-y-auto" />
    </Panel>
  )
}

export function CVFolds({ exp }: { exp: Experiment }) {
  if (!exp.cv) return null
  const pm = exp.primary_metric
  const m = exp.cv.metrics[pm]
  if (!m?.folds) return null
  return (
    <Panel title={`${METRICS[pm]?.name ?? pm} por fold`} subtitle={`${exp.cv.splitter}, ${exp.cv.folds} folds. Linha = média`}>
      <Plot
        deps={[exp.id]}
        height={260}
        build={(c) => ({
          data: [
            {
              type: 'bar',
              x: m.folds!.map((_, i) => `Fold ${i + 1}`),
              y: m.folds,
              marker: { color: c.accent },
              hovertemplate: '%{x}: %{y:.4f}<extra></extra>',
            },
          ],
          layout: {
            shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: m.mean, y1: m.mean, line: { color: c.warn, width: 1.5, dash: 'dash' } }],
            yaxis: { title: METRICS[pm]?.name ?? pm },
          },
        })}
      />
    </Panel>
  )
}
