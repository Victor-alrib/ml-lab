import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { HyperParam, ModelInfo, Task } from '../api/types'
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
  NumberInput,
  PageHeader,
  Panel,
  Select,
  TextInput,
  Toggle,
} from '../components/ui'
import { useApp } from '../context/AppContext'
import { fmtSeconds } from '../lib/format'
import { useAsync } from '../lib/useAsync'
import { useProfile } from '../lib/useProfile'
import { PipelineDiagram } from './Preprocess'

function ParamInput({ p, value, onChange }: { p: HyperParam; value: unknown; onChange: (v: unknown) => void }) {
  if (p.type === 'bool') return <Toggle checked={Boolean(value)} onChange={onChange} label={p.label} hint={p.help} />
  return (
    <Field label={p.label} hint={p.help}>
      {p.type === 'select' ? (
        <Select value={String(value ?? '')} onChange={onChange} options={p.options!} />
      ) : (
        <NumberInput
          value={value as number | null}
          onChange={onChange}
          min={p.min}
          max={p.max}
          step={p.type === 'int' ? 1 : p.step}
          allowEmpty={p.nullable}
        />
      )}
    </Field>
  )
}

function useElapsed(active: boolean) {
  const [t, setT] = useState(0)
  useEffect(() => {
    if (!active) return setT(0)
    const start = performance.now()
    const id = setInterval(() => setT((performance.now() - start) / 1000), 200)
    return () => clearInterval(id)
  }, [active])
  return t
}

export default function TrainPage() {
  const { datasetId, dataset, preprocess, models } = useApp()
  const profile = useProfile(datasetId)
  const nav = useNavigate()
  const info = useAsync(
    () => (datasetId && preprocess ? api.targetInfo(datasetId, preprocess.target) : Promise.resolve(null)),
    [datasetId, preprocess?.target],
    !!preprocess,
  )
  const task: Task | undefined = preprocess?.task === 'auto' ? info.data?.task : (preprocess?.task as Task | undefined)
  const available = useMemo(() => models.filter((m) => task && m.tasks.includes(task)), [models, task])

  const [modelKey, setModelKey] = useState<string>('')
  const [params, setParams] = useState<Record<string, unknown>>({})
  const [cvOn, setCvOn] = useState(true)
  const [folds, setFolds] = useState(5)
  const [average, setAverage] = useState('macro')
  const [lc, setLc] = useState(false)
  const [perm, setPerm] = useState(false)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [batchDone, setBatchDone] = useState<string[]>([])
  const elapsed = useElapsed(!!busy)

  const model: ModelInfo | undefined = available.find((m) => m.key === modelKey) ?? available[0]
  const schema = (model && task && model.params[task]) || []

  useEffect(() => {
    if (!model || !task) return
    setModelKey(model.key)
    setParams(Object.fromEntries((model.params[task] ?? []).map((p) => [p.name, p.default])))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.key, task])

  const nClasses = info.data?.classes?.length ?? 0

  if (!datasetId || !dataset)
    return (
      <>
        <PageHeader title="Treinamento" />
        <Empty title="Nenhum dataset ativo" action={<Link className="text-accent underline" to="/datasets">Carregar um dataset</Link>} />
      </>
    )
  if (!preprocess)
    return (
      <>
        <PageHeader title="Treinamento" />
        <Empty
          title="Configure o pré-processamento primeiro"
          action={
            <Link className="text-accent underline" to="/preprocess">
              Ir para pré-processamento
            </Link>
          }
        >
          O treinamento usa o pipeline definido lá: alvo, atributos, imputação, escala e a separação treino/teste.
        </Empty>
      </>
    )

  const body = (key: string, p: Record<string, unknown>, nm?: string) => ({
    dataset_id: datasetId,
    model: key,
    params: p,
    preprocessing: preprocess,
    cv: { enabled: cvOn, folds },
    average,
    learning_curve: lc,
    permutation_importance: perm,
    name: nm || undefined,
  })

  async function trainOne() {
    if (!model) return
    setBusy(model.key)
    setError(null)
    try {
      const exp = await api.train(body(model.key, params, name))
      nav(`/evaluate/${exp.id}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  async function trainAll() {
    setError(null)
    setBatchDone([])
    const ids: string[] = []
    for (const m of available) {
      setBusy(m.key)
      try {
        const defaults = Object.fromEntries((m.params[task!] ?? []).map((p) => [p.name, p.default]))
        const exp = await api.train(body(m.key, defaults))
        ids.push(exp.id)
        setBatchDone((d) => [...d, m.key])
      } catch (e) {
        setError(`${m.name}: ${(e as Error).message}`)
      }
    }
    setBusy(null)
    if (ids.length) nav(`/experiments?compare=${ids.join(',')}`)
  }

  const visible = (p: HyperParam) =>
    !p.depends_on || Object.entries(p.depends_on).every(([k, vals]) => vals.includes(String(params[k])))

  const featureCount = preprocess.features?.length ?? 0
  const kinds = profile.data?.columns.filter((c) => preprocess.features?.includes(c.name)) ?? []
  const nNum = kinds.filter((c) => c.stats.mean !== undefined && c.semantic_type !== 'categorica').length

  return (
    <>
      <PageHeader title="Treinamento">
        Escolha o algoritmo e os hiperparâmetros. Cada execução é salva automaticamente como um experimento.
      </PageHeader>

      <div className="mb-4 flex flex-col gap-3 rounded-md border border-line bg-panel px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-[14px]">
          <span>
            <span className="text-muted">Alvo </span>
            <b>{preprocess.target}</b>
          </span>
          <span>
            <span className="text-muted">Tarefa </span>
            {task ? (task === 'classification' ? `Classificação${nClasses ? ` (${nClasses} classes)` : ''}` : 'Regressão') : '…'}
          </span>
          <span>
            <span className="text-muted">Atributos </span>
            {featureCount}
          </span>
          <span>
            <span className="text-muted">Teste </span>
            {Math.round(preprocess.test_size * 100)}% ({preprocess.split_strategy === 'temporal' ? 'temporal' : preprocess.split_strategy === 'stratified' ? 'estratificado' : 'aleatório'})
          </span>
          <Link to="/preprocess" className="ml-auto text-[13px] text-accent hover:underline">
            Alterar pré-processamento
          </Link>
        </div>
        <PipelineDiagram cfg={preprocess} nNum={nNum} nCat={kinds.length - nNum} model={model?.name} />
      </div>

      {info.loading && <Loading />}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Panel title="Algoritmo">
          <div className="flex flex-col gap-2">
            {available.map((m) => (
              <button
                key={m.key}
                onClick={() => setModelKey(m.key)}
                className={cx(
                  'rounded border px-3 py-2.5 text-left transition-colors',
                  m.key === model?.key ? 'border-accent bg-accent/8' : 'border-line hover:border-muted',
                )}
              >
                <div className="flex items-center gap-2">
                  <span className={cx('font-semibold', m.key === model?.key && 'text-accent')}>{m.name}</span>
                  <Badge>{m.family}</Badge>
                  {busy === m.key && <span className="ml-auto text-[12px] text-muted">treinando…</span>}
                  {batchDone.includes(m.key) && busy !== m.key && <span className="ml-auto text-[12px] text-ok">concluído</span>}
                </div>
                <p className="mt-0.5 text-[13px] text-muted">{m.description}</p>
              </button>
            ))}
          </div>
          {model?.notes.map((n) => (
            <p key={n} className="mt-3 text-[13px] text-muted">
              {n}
            </p>
          ))}
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel title={`Hiperparâmetros — ${model?.name ?? ''}`}>
            <div className="grid gap-4 sm:grid-cols-2">
              {schema.filter(visible).map((p) => (
                <div key={p.name} className={p.type === 'bool' ? 'sm:col-span-2' : ''}>
                  <ParamInput p={p} value={params[p.name]} onChange={(v) => setParams({ ...params, [p.name]: v })} />
                </div>
              ))}
            </div>
            <div className="mt-3">
              <Button size="sm" variant="ghost" onClick={() => setParams(Object.fromEntries(schema.map((p) => [p.name, p.default])))}>
                Restaurar padrões
              </Button>
            </div>
          </Panel>

          <Panel title="Avaliação">
            <div className="grid gap-4 sm:grid-cols-2">
              <Toggle checked={cvOn} onChange={setCvOn} label="Validação cruzada no treino" hint="O pipeline inteiro é reajustado em cada fold." />
              {cvOn && (
                <Field label="Número de folds (k)">
                  <NumberInput value={folds} min={2} max={20} step={1} onChange={(v) => setFolds(Math.min(20, Math.max(2, v ?? 5)))} />
                </Field>
              )}
              {task === 'classification' && nClasses > 2 && (
                <Field label="Média das métricas (multiclasse)" hint="Macro: cada classe pesa igual. Weighted: pondera pelo suporte.">
                  <Select
                    value={average}
                    onChange={setAverage}
                    options={[
                      { value: 'macro', label: 'Macro' },
                      { value: 'weighted', label: 'Weighted' },
                    ]}
                  />
                </Field>
              )}
              <Toggle checked={lc} onChange={setLc} label="Curva de aprendizado" hint="Treina com 15% a 100% do treino, com CV em cada tamanho. Mais lento." />
              <Toggle checked={perm} onChange={setPerm} label="Importância por permutação" hint="Mede a queda de desempenho no teste ao embaralhar cada atributo original." />
              <Field label="Nome do experimento (opcional)" className="sm:col-span-2">
                <TextInput value={name} onChange={(e) => setName(e.target.value)} placeholder={`${model?.name ?? ''} · ${dataset.name}`} />
              </Field>
            </div>
            <div className="mt-3">
              <Edu concept="cv" />
            </div>
          </Panel>

          <ErrorBox error={error} />
          {preprocess.split_strategy === 'temporal' && cvOn && (
            <Notice>Com separação temporal, a validação cruzada usa TimeSeriesSplit: cada fold treina no passado e valida no período seguinte.</Notice>
          )}
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" onClick={trainOne} loading={!!busy && busy === model?.key && !batchDone.length} disabled={!!busy || !model}>
              Treinar {model?.name}
            </Button>
            <Button onClick={trainAll} disabled={!!busy || !available.length}>
              Treinar todos com parâmetros padrão e comparar
            </Button>
            {busy && <span className="num text-[13px] text-muted">Tempo decorrido: {fmtSeconds(elapsed)}</span>}
          </div>
        </div>
      </div>
    </>
  )
}
