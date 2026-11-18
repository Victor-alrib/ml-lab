const nf = (d: number) => new Intl.NumberFormat('pt-BR', { maximumFractionDigits: d, minimumFractionDigits: 0 })

export function fmt(v: unknown, digits = 3): string {
  if (v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v))) return '—'
  if (typeof v !== 'number') return String(v)
  const a = Math.abs(v)
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(2).replace('.', ',')
  return nf(a >= 100 ? Math.min(digits, 2) : digits).format(v)
}

export const fmtInt = (v: number | null | undefined) => (v == null ? '—' : new Intl.NumberFormat('pt-BR').format(v))
export const fmtPct = (v: number | null | undefined, d = 1) => (v == null ? '—' : `${nf(d).format(v)}%`)

export function fmtSeconds(s: number) {
  if (s < 1) return `${nf(0).format(s * 1000)} ms`
  if (s < 60) return `${nf(2).format(s)} s`
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`
}

export function fmtDate(iso: string) {
  return new Date(iso).toLocaleString('pt-BR', { dateStyle: 'short', timeStyle: 'short' })
}

export function fmtBytes(b: number) {
  if (b < 1024) return `${b} B`
  if (b < 1024 ** 2) return `${nf(1).format(b / 1024)} KB`
  return `${nf(1).format(b / 1024 ** 2)} MB`
}

/** Remove prefixos do ColumnTransformer (num__, cat__) para exibição. */
export const featureLabel = (f: string) => f.replace(/^(num|cat|missingindicator)__/, '')
