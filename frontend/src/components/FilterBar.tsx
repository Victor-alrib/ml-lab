import { useState } from 'react'
import type { Filter, Profile } from '../api/types'
import { isNumericType } from '../lib/useProfile'
import { fmt } from '../lib/format'
import { Button, NumberInput, Select } from './ui'

function describe(f: Filter) {
  if (f.op === 'between') {
    const [a, b] = f.value as [number | null, number | null]
    return `${f.column} entre ${fmt(a)} e ${fmt(b)}`
  }
  if (f.op === 'in') return `${f.column} ∈ {${(f.value as string[]).join(', ')}}`
  if (f.op === 'notnull') return `${f.column} preenchido`
  return `${f.column} ${f.op} ${String(f.value)}`
}

/** Filtros interativos aplicados no servidor sobre o dataset completo. */
export function FilterBar({
  profile,
  filters,
  onChange,
  rowsFiltered,
  rowsTotal,
}: {
  profile: Profile
  filters: Filter[]
  onChange: (f: Filter[]) => void
  rowsFiltered?: number
  rowsTotal?: number
}) {
  const [col, setCol] = useState('')
  const c = profile.columns.find((x) => x.name === col)
  const numeric = c ? isNumericType(c.semantic_type) : false
  const [lo, setLo] = useState<number | null>(null)
  const [hi, setHi] = useState<number | null>(null)
  const [vals, setVals] = useState<string[]>([])

  function pick(name: string) {
    setCol(name)
    const cc = profile.columns.find((x) => x.name === name)
    setLo(cc?.stats.min ?? null)
    setHi(cc?.stats.max ?? null)
    setVals([])
  }

  function add() {
    if (!c) return
    const f: Filter = numeric ? { column: c.name, op: 'between', value: [lo, hi] } : { column: c.name, op: 'in', value: vals }
    if (!numeric && vals.length === 0) return
    onChange([...filters.filter((x) => x.column !== c.name), f])
    setCol('')
  }

  const filterable = profile.columns.filter((x) => isNumericType(x.semantic_type) || (x.stats.top_values && x.unique <= 50))

  return (
    <div className="rounded-md border border-line bg-panel px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13px] font-semibold">Filtros</span>
        {filters.length === 0 && <span className="text-[13px] text-muted">Nenhum — todas as linhas</span>}
        {filters.map((f) => (
          <span key={f.column} className="num inline-flex items-center gap-1.5 rounded border border-accent/50 bg-accent/8 px-2 py-0.5 text-[13px]">
            {describe(f)}
            <button
              aria-label={`Remover filtro ${f.column}`}
              className="text-muted hover:text-alarm"
              onClick={() => onChange(filters.filter((x) => x !== f))}
            >
              ×
            </button>
          </span>
        ))}
        {filters.length > 0 && (
          <Button size="sm" variant="ghost" onClick={() => onChange([])}>
            Limpar
          </Button>
        )}
        {rowsFiltered !== undefined && rowsTotal !== undefined && (
          <span className="num ml-auto text-[13px] text-muted">
            {rowsFiltered.toLocaleString('pt-BR')} de {rowsTotal.toLocaleString('pt-BR')} linhas
          </span>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-end gap-2">
        <div className="w-56">
          <Select
            ariaLabel="Coluna para filtrar"
            value={col}
            onChange={pick}
            options={[{ value: '', label: 'Adicionar filtro por…' }, ...filterable.map((x) => ({ value: x.name, label: x.name }))]}
          />
        </div>
        {c && numeric && (
          <>
            <div className="w-32">
              <NumberInput ariaLabel="Mínimo" value={lo} onChange={setLo} />
            </div>
            <span className="pb-1.5 text-sm text-muted">até</span>
            <div className="w-32">
              <NumberInput ariaLabel="Máximo" value={hi} onChange={setHi} />
            </div>
          </>
        )}
        {c && !numeric && (
          <div className="flex max-w-2xl flex-wrap gap-1.5">
            {c.stats.top_values?.map((t) => (
              <label key={t.value} className="flex items-center gap-1 rounded border border-line px-2 py-0.5 text-[13px]">
                <input
                  type="checkbox"
                  checked={vals.includes(t.value)}
                  onChange={(e) => setVals(e.target.checked ? [...vals, t.value] : vals.filter((v) => v !== t.value))}
                />
                {t.value}
              </label>
            ))}
          </div>
        )}
        {c && (
          <Button size="sm" variant="primary" onClick={add}>
            Aplicar filtro
          </Button>
        )}
      </div>
    </div>
  )
}
