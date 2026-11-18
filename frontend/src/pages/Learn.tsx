import { useState } from 'react'
import { Field, NumberInput, PageHeader, Panel, Readout, Segmented, Table } from '../components/ui'
import { CONCEPTS, METRICS } from '../lib/education'
import { fmt } from '../lib/format'

function ConfusionCalculator() {
  const [v, setV] = useState({ tp: 40, fp: 25, fn: 10, tn: 925 })
  const n = v.tp + v.fp + v.fn + v.tn
  const precision = v.tp + v.fp ? v.tp / (v.tp + v.fp) : 0
  const recall = v.tp + v.fn ? v.tp / (v.tp + v.fn) : 0
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0
  const acc = n ? (v.tp + v.tn) / n : 0
  const spec = v.tn + v.fp ? v.tn / (v.tn + v.fp) : 0
  const bal = (recall + spec) / 2
  const majority = n ? Math.max(v.tp + v.fn, v.fp + v.tn) / n : 0
  const cell = (k: keyof typeof v, label: string) => (
    <Field label={label}>
      <NumberInput value={v[k]} min={0} step={1} onChange={(x) => setV({ ...v, [k]: Math.max(0, Math.round(x ?? 0)) })} />
    </Field>
  )
  return (
    <Panel title="Calculadora de matriz de confusão" subtitle="Altere as contagens e veja como cada métrica reage">
      <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <div className="grid grid-cols-2 gap-3">
          {cell('tn', 'Real normal → previsto normal (VN)')}
          {cell('fp', 'Real normal → previsto falha (FP)')}
          {cell('fn', 'Real falha → previsto normal (FN)')}
          {cell('tp', 'Real falha → previsto falha (VP)')}
        </div>
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-2">
            <Readout label="Accuracy" metric="accuracy" value={fmt(acc)} sub={`só prever "normal": ${fmt(majority)}`} />
            <Readout label="Balanced acc." metric="balanced_accuracy" value={fmt(bal)} />
            <Readout label="Precision" metric="precision" value={fmt(precision)} />
            <Readout label="Recall" metric="recall" value={fmt(recall)} />
            <Readout label="F1" metric="f1" value={fmt(f1)} />
          </div>
          <p className="text-[13.5px] text-muted">
            Com {v.tp + v.fn} falhas em {n} leituras, a prevalência é {fmt(n ? ((v.tp + v.fn) / n) * 100 : 0, 1)}%. Compare a accuracy com a de
            um modelo que sempre responde “normal”: é por isso que, em dados de falha, accuracy sozinha engana.
          </p>
        </div>
      </div>
    </Panel>
  )
}

const SCENARIOS = [
  ['Falha não detectada é perigosa ou muito cara (segurança, parada longa)', 'Recall', 'Aceita mais falsos alarmes; ajuste o limiar para baixo.'],
  ['Cada alarme gera uma intervenção cara (parada de linha, deslocamento)', 'Precision', 'Alarmes precisam ser confiáveis; limiar mais alto.'],
  ['Custos de falso alarme e falha perdida parecidos, classes desbalanceadas', 'F1 / balanced accuracy', 'Métrica padrão deste app.'],
  ['Priorizar uma fila de inspeção por risco', 'ROC AUC / curva PR', 'Avalia o ranqueamento, independe do limiar.'],
  ['Prever uma grandeza física (temperatura, vida útil restante)', 'MAE + RMSE', 'Erro na unidade física; RMSE pune erros grandes.'],
  ['Comparar ajuste entre alvos de escalas diferentes', 'R²', 'Sempre acompanhado de MAE/RMSE.'],
]

export default function LearnPage() {
  const [task, setTask] = useState<'classification' | 'regression'>('classification')
  const metrics = Object.values(METRICS).filter((m) => m.task === task)
  return (
    <>
      <PageHeader title="Guia de métricas e conceitos">
        Referência do modo educacional. Com ele ligado (no topo da tela), notas como estas aparecem junto dos gráficos e resultados em cada
        módulo.
      </PageHeader>
      <div className="flex flex-col gap-6">
        <Panel title="Qual métrica priorizar?">
          <Table head={['Situação', 'Métrica', 'Observação']} rows={SCENARIOS.map((s) => [<span className="whitespace-normal">{s[0]}</span>, <b>{s[1]}</b>, <span className="whitespace-normal text-muted">{s[2]}</span>])} />
        </Panel>

        <ConfusionCalculator />

        <section>
          <div className="mb-3 flex items-center gap-4">
            <h2 className="font-display text-[22px] font-semibold">Métricas</h2>
            <Segmented
              value={task}
              onChange={setTask}
              options={[
                { value: 'classification', label: 'Classificação' },
                { value: 'regression', label: 'Regressão' },
              ]}
            />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            {metrics.map((m) => (
              <Panel key={m.key} title={m.name} subtitle={`${m.higherIsBetter ? 'Maior é melhor' : 'Menor é melhor'} · ${m.range}`}>
                <p className="font-mono text-[13px] text-muted">{m.formula}</p>
                <p className="mt-2">{m.what}</p>
                <p className="mt-2 text-[14px] text-muted">{m.read}</p>
                {m.industry !== '—' && (
                  <p className="mt-2 border-l-2 border-accent pl-3 text-[14px]">
                    <span className="font-semibold">Na indústria: </span>
                    {m.industry}
                  </p>
                )}
              </Panel>
            ))}
          </div>
        </section>

        <section>
          <h2 className="mb-3 font-display text-[22px] font-semibold">Conceitos de validação</h2>
          <div className="grid gap-4 lg:grid-cols-2">
            {Object.entries(CONCEPTS).map(([k, c]) => (
              <Panel key={k} title={c.title}>
                <div className="flex flex-col gap-2 text-[14.5px] leading-relaxed">
                  {c.body.map((p, i) => (
                    <p key={i}>{p}</p>
                  ))}
                </div>
              </Panel>
            ))}
          </div>
        </section>
      </div>
    </>
  )
}
