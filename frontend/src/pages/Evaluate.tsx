import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Experiment } from '../api/types'
import {
  Coefficients,
  ConfusionMatrix,
  CVFolds,
  Importance,
  LearningCurve,
  ProbaHist,
  RegressionCharts,
  RocPr,
  ThresholdExplorer,
} from '../components/EvalCharts'
import { Badge, Edu, Empty, ErrorBox, Loading, MetricHelp, Notice, PageHeader, Panel, Readout, Select, Table } from '../components/ui'
import { interpret, METRICS } from '../lib/education'
import { featureLabel, fmt, fmtDate, fmtInt, fmtSeconds } from '../lib/format'
import { useAsync } from '../lib/useAsync'

const CLS_ORDER = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']
const REG_ORDER = ['mae', 'rmse', 'r2', 'mape', 'mse']

export function MetricsTable({ exp }: { exp: Experiment }) {
  const order = (exp.task === 'classification' ? CLS_ORDER : REG_ORDER).filter((k) => exp.metrics.test[k] !== undefined)
  return (
    <Table
      head={['Métrica', 'Treino', 'Validação cruzada', 'Teste', 'Baseline (teste)']}
      align={['left', 'right', 'right', 'right', 'right']}
      rows={order.map((k) => {
        const cv = exp.cv?.metrics[k]
        return [
          <span className="inline-flex items-center gap-1.5">
            {METRICS[k]?.name ?? k} <MetricHelp metric={k} />
          </span>,
          fmt(exp.metrics.train[k]),
          cv ? `${fmt(cv.mean)} ± ${fmt(cv.std)}` : '—',
          <b>{fmt(exp.metrics.test[k])}</b>,
          exp.metrics.baseline[k] !== undefined ? <span className="text-muted">{fmt(exp.metrics.baseline[k])}</span> : '—',
        ]
      })}
    />
  )
}

function Headline({ exp }: { exp: Experiment }) {
  const keys = exp.task === 'classification' ? ['f1', 'recall', 'precision', 'roc_auc', 'accuracy'] : ['rmse', 'mae', 'r2']
  return (
    <div className="flex flex-wrap gap-2">
      {keys
        .filter((k) => exp.metrics.test[k] !== undefined)
        .map((k) => {
          const cv = exp.cv?.metrics[k]
          return (
            <Readout
              key={k}
              metric={k}
              label={METRICS[k]?.name ?? k}
              value={fmt(exp.metrics.test[k])}
              sub={cv ? `CV ${fmt(cv.mean)} ± ${fmt(cv.std)}` : `treino ${fmt(exp.metrics.train[k])}`}
            />
          )
        })}
    </div>
  )
}

const levelMap = { ok: 'ok', warn: 'warn', alarm: 'alarm', info: 'info' } as const

function Details({ exp }: { exp: Experiment }) {
  const p = exp.preprocessing
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Panel title="Hiperparâmetros">
        <Table head={['Parâmetro', 'Valor']} align={['left', 'right']} rows={Object.entries(exp.params).map(([k, v]) => [k, v === null ? 'None' : String(v)])} />
      </Panel>
      <Panel title="Configuração e execução">
        <Table
          head={['Item', 'Valor']}
          align={['left', 'right']}
          rows={[
            ['Treino / teste', `${fmtInt(exp.n_train)} / ${fmtInt(exp.n_test)}`],
            ['Separação', `${exp.split_strategy}, ${Math.round(p.test_size * 100)}% teste, semente ${p.random_state}`],
            ['Validação cruzada', exp.cv ? `${exp.cv.splitter}, ${exp.cv.folds} folds` : 'desativada'],
            ['Imputação (num./cat.)', `${p.numeric_imputation} / ${p.categorical_imputation}`],
            ['Escala · codificação', `${p.scaling} · ${p.encoding}`],
            ['Outliers', p.outlier_treatment],
            ['Seleção de atributos', p.feature_selection.method === 'none' ? 'nenhuma' : `${p.feature_selection.method} (k=${p.feature_selection.k})`],
            ['Atributos finais', String(exp.final_features.length)],
            ['Tempo de ajuste final', fmtSeconds(exp.timing.fit_s)],
            ['Tempo de predição (teste)', fmtSeconds(exp.timing.predict_s)],
            ['Tempo da validação cruzada', fmtSeconds(exp.timing.cv_s)],
            ['Tempo total', fmtSeconds(exp.timing.total_s)],
            ['Ambiente', `Python ${exp.environment.python} · sklearn ${exp.environment.sklearn} · xgboost ${exp.environment.xgboost}`],
          ]}
        />
      </Panel>
      <Panel title="Atributos usados" className="lg:col-span-2">
        <div className="flex flex-wrap gap-1.5">
          {exp.final_features.map((f) => (
            <span key={f} className="rounded border border-line bg-bg px-1.5 py-0.5 text-[12.5px]">
              {featureLabel(f)}
            </span>
          ))}
        </div>
      </Panel>
    </div>
  )
}

function PerClass({ exp }: { exp: Experiment }) {
  if (!exp.results.per_class) return null
  return (
    <Panel title="Métricas por classe (teste)">
      <Table
        head={['Classe', 'Precision', 'Recall', 'F1', 'Suporte']}
        align={['left', 'right', 'right', 'right', 'right']}
        rows={exp.results.per_class.map((r) => [r.label, fmt(r.precision), fmt(r.recall), fmt(r.f1), fmtInt(r.support)])}
      />
      {exp.results.n_classes > 2 && (
        <p className="mt-2 text-[13px] text-muted">As métricas agregadas usam média {exp.average === 'weighted' ? 'ponderada pelo suporte' : 'macro (cada classe pesa igual)'}.</p>
      )}
    </Panel>
  )
}

export default function EvaluatePage() {
  const { id } = useParams()
  const nav = useNavigate()
  const list = useAsync(() => api.experiments(), [])
  const currentId = id ?? list.data?.[0]?.id
  const exp = useAsync(() => (currentId ? api.experiment(currentId) : Promise.resolve(null)), [currentId], !!currentId)
  const e = exp.data

  if (list.data && !list.data.length)
    return (
      <>
        <PageHeader title="Avaliação" />
        <Empty title="Nenhum experimento ainda" action={<Link className="text-accent underline" to="/train">Treinar um modelo</Link>}>
          Cada treino gera um experimento com métricas de treino, validação cruzada, teste e baseline.
        </Empty>
      </>
    )

  return (
    <>
      <PageHeader
        title="Avaliação"
        actions={
          list.data && (
            <div className="w-80">
              <Select
                ariaLabel="Experimento"
                value={currentId ?? ''}
                onChange={(v) => nav(`/evaluate/${v}`)}
                options={list.data.map((x) => ({ value: x.id, label: `${x.name} — ${fmtDate(x.created_at)}` }))}
              />
            </div>
          )
        }
      >
        Todas as métricas e gráficos abaixo foram calculados com as previsões reais do modelo no conjunto de teste, que não participou do
        ajuste.
      </PageHeader>
      <ErrorBox error={list.error || exp.error} />
      {(exp.loading || list.loading) && !e && <Loading />}
      {e && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <h2 className="font-display text-[22px] font-semibold">{e.name}</h2>
            <Badge tone="accent">{e.model_name}</Badge>
            <Badge>{e.task === 'classification' ? 'Classificação' : 'Regressão'}</Badge>
            <span className="text-[13px] text-muted">
              {e.dataset_name} · alvo <b className="text-ink">{e.target}</b> · {fmtDate(e.created_at)}
            </span>
            {e.model_file && (
              <a className="ml-auto text-[13px] text-accent hover:underline" href={api.modelUrl(e.id)}>
                Baixar pipeline treinado (.joblib)
              </a>
            )}
          </div>

          <Headline exp={e} />

          <div className="grid gap-4 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
            <Panel title="Treino × validação × teste × baseline">
              <MetricsTable exp={e} />
              <div className="mt-3 flex flex-col gap-2">
                <Edu concept="baseline" />
              </div>
            </Panel>
            <Panel title="Interpretação" subtitle="Regras simples aplicadas aos valores deste experimento">
              <div className="flex flex-col gap-2">
                {interpret(e).map((i) => (
                  <Notice key={i.text} level={levelMap[i.level]}>
                    {i.text}
                  </Notice>
                ))}
                {e.warnings.map((w) => (
                  <Notice key={w} level={w.startsWith('Possível vazamento') ? 'alarm' : 'warn'}>
                    {w}
                  </Notice>
                ))}
              </div>
            </Panel>
          </div>

          {e.task === 'classification' ? (
            <>
              <div className="grid gap-4 lg:grid-cols-2">
                <ConfusionMatrix exp={e} />
                <div className="flex flex-col gap-4">
                  <PerClass exp={e} />
                  <CVFolds exp={e} />
                </div>
              </div>
              <RocPr exps={[e]} />
              {e.results.threshold_data && <ThresholdExplorer exp={e} />}
              <ProbaHist exp={e} />
            </>
          ) : (
            <>
              <RegressionCharts exp={e} />
              <CVFolds exp={e} />
            </>
          )}
          <LearningCurve exp={e} />
          <Importance exp={e} />
          <Coefficients exp={e} />
          <Details exp={e} />
          {e.notes_preprocessing.length > 0 && (
            <div className="flex flex-col gap-2">
              {e.notes_preprocessing.map((n) => (
                <Notice key={n}>{n}</Notice>
              ))}
            </div>
          )}
        </div>
      )}
    </>
  )
}
