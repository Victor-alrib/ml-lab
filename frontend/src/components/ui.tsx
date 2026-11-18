import { useId, useState, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { useApp } from '../context/AppContext'
import { CONCEPTS, METRICS } from '../lib/education'

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(' ')
}

export function Panel({
  title,
  actions,
  children,
  className,
  pad = true,
  subtitle,
}: {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  pad?: boolean
}) {
  return (
    <section className={cx('rounded-md border border-line bg-panel', className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5">
          <div>
            {title && <h2 className="font-display text-[17px] font-semibold leading-tight">{title}</h2>}
            {subtitle && <p className="text-[13px] text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={pad ? 'p-4' : ''}>{children}</div>
    </section>
  )
}

type BtnVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
export function Button({
  variant = 'secondary',
  size = 'md',
  loading,
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: BtnVariant; size?: 'sm' | 'md'; loading?: boolean }) {
  const styles: Record<BtnVariant, string> = {
    primary: 'bg-accent text-accent-ink hover:brightness-110 border-transparent font-semibold',
    secondary: 'bg-panel-2 text-ink border-line hover:border-muted',
    ghost: 'bg-transparent text-muted border-transparent hover:text-ink hover:bg-panel-2',
    danger: 'bg-transparent text-alarm border-line hover:border-alarm',
  }
  return (
    <button
      {...rest}
      disabled={rest.disabled || loading}
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded border transition-colors disabled:cursor-not-allowed disabled:opacity-50',
        size === 'sm' ? 'px-2.5 py-1 text-[13px]' : 'px-3.5 py-1.5 text-sm',
        styles[variant],
        className,
      )}
    >
      {loading && <Spinner small />}
      {children}
    </button>
  )
}

export function Spinner({ small }: { small?: boolean }) {
  return (
    <span
      role="status"
      aria-label="Carregando"
      className={cx(
        'inline-block animate-spin rounded-full border-2 border-current border-r-transparent',
        small ? 'h-3.5 w-3.5' : 'h-5 w-5',
      )}
    />
  )
}

export function Loading({ label = 'Calculando…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 px-1 py-6 text-sm text-muted">
      <Spinner /> {label}
    </div>
  )
}

export function Field({
  label,
  hint,
  children,
  className,
}: {
  label: ReactNode
  hint?: ReactNode
  children: ReactNode
  className?: string
}) {
  const id = useId()
  return (
    <div className={cx('flex flex-col gap-1', className)}>
      <label htmlFor={id} className="text-[13px] font-medium text-muted">
        {label}
      </label>
      <div id={id}>{children}</div>
      {hint && <p className="text-[12px] leading-snug text-muted">{hint}</p>}
    </div>
  )
}

const inputCls =
  'w-full rounded border border-line bg-bg px-2.5 py-1.5 text-sm text-ink placeholder:text-muted/70 focus:border-accent focus:outline-none'

export function Select<T extends string>({
  value,
  onChange,
  options,
  className,
  ariaLabel,
}: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: string }[]
  className?: string
  ariaLabel?: string
}) {
  return (
    <select
      aria-label={ariaLabel}
      className={cx(inputCls, 'pr-7', className)}
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

export function TextInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cx(inputCls, props.className)} />
}

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
  allowEmpty,
  ariaLabel,
}: {
  value: number | null
  onChange: (v: number | null) => void
  min?: number
  max?: number
  step?: number
  allowEmpty?: boolean
  ariaLabel?: string
}) {
  const [text, setText] = useState<string | null>(null)
  const shown = text ?? (value === null || value === undefined ? '' : String(value))
  return (
    <input
      aria-label={ariaLabel}
      type="number"
      className={cx(inputCls, 'num')}
      value={shown}
      min={min}
      max={max}
      step={step ?? 'any'}
      placeholder={allowEmpty ? 'sem limite' : undefined}
      onChange={(e) => {
        setText(e.target.value)
        if (e.target.value === '') {
          if (allowEmpty) onChange(null)
          return
        }
        const n = Number(e.target.value)
        if (Number.isFinite(n)) onChange(n)
      }}
      onBlur={() => setText(null)}
    />
  )
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: ReactNode
  hint?: ReactNode
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5 text-sm">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cx(
          'relative mt-0.5 h-5 w-9 shrink-0 rounded-full border transition-colors',
          checked ? 'border-accent bg-accent' : 'border-line bg-panel-2',
        )}
      >
        <span
          className={cx(
            'absolute top-0.5 h-3.5 w-3.5 rounded-full transition-all',
            checked ? 'left-[18px] bg-accent-ink' : 'left-0.5 bg-muted',
          )}
        />
      </button>
      <span>
        <span className="text-ink">{label}</span>
        {hint && <span className="block text-[12px] leading-snug text-muted">{hint}</span>}
      </span>
    </label>
  )
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: string }[]
}) {
  return (
    <div role="tablist" className="inline-flex flex-wrap rounded border border-line bg-bg p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            'rounded-[3px] px-3 py-1 text-[13px] transition-colors',
            value === o.value ? 'bg-panel-2 font-semibold text-ink shadow-[inset_0_-2px_0_var(--accent)]' : 'text-muted hover:text-ink',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

const levelStyles = {
  ok: 'border-ok/40 bg-ok/8 [&_.lvl]:bg-ok',
  warn: 'border-warn/50 bg-warn/8 [&_.lvl]:bg-warn',
  alarm: 'border-alarm/50 bg-alarm/8 [&_.lvl]:bg-alarm',
  info: 'border-line bg-panel-2 [&_.lvl]:bg-accent',
}

/** Mensagem de status no estilo ISA-101: cor só quando carrega significado. */
export function Notice({
  level = 'info',
  children,
  title,
}: {
  level?: keyof typeof levelStyles
  children: ReactNode
  title?: ReactNode
}) {
  return (
    <div className={cx('flex gap-3 rounded border px-3 py-2 text-[13.5px] leading-snug', levelStyles[level])}>
      <span className="lvl mt-1 h-2 w-2 shrink-0 rounded-[1px]" aria-hidden />
      <div>
        {title && <div className="mb-0.5 font-semibold">{title}</div>}
        {children}
      </div>
    </div>
  )
}

export function ErrorBox({ error }: { error: string | null }) {
  if (!error) return null
  return <Notice level="alarm">{error}</Notice>
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-start gap-2 rounded-md border border-dashed border-line px-6 py-10">
      <h3 className="font-display text-lg font-semibold">{title}</h3>
      {children && <div className="max-w-xl text-sm text-muted">{children}</div>}
      {action}
    </div>
  )
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'accent' | 'warn' | 'alarm' | 'ok' }) {
  const t = {
    neutral: 'border-line text-muted',
    accent: 'border-accent/50 text-accent',
    warn: 'border-warn/60 text-warn',
    alarm: 'border-alarm/60 text-alarm',
    ok: 'border-ok/60 text-ok',
  }[tone]
  return <span className={cx('inline-flex items-center rounded border px-1.5 py-px text-[12px] leading-5', t)}>{children}</span>
}

/** Valor de destaque: número grande + rótulo. */
export function Readout({
  label,
  value,
  sub,
  tone,
  metric,
}: {
  label: ReactNode
  value: ReactNode
  sub?: ReactNode
  tone?: 'ok' | 'warn' | 'alarm'
  metric?: string
}) {
  return (
    <div className="min-w-[120px] flex-1 rounded border border-line bg-bg px-3 py-2">
      <div className="flex items-center gap-1 text-[13px] text-muted">
        {label}
        {metric && <MetricHelp metric={metric} />}
      </div>
      <div
        className={cx(
          'num font-display text-[26px] font-semibold leading-tight',
          tone === 'ok' && 'text-ok',
          tone === 'warn' && 'text-warn',
          tone === 'alarm' && 'text-alarm',
        )}
      >
        {value}
      </div>
      {sub && <div className="num text-[12px] text-muted">{sub}</div>}
    </div>
  )
}

/** Ícone "?" com explicação da métrica (sempre disponível). */
export function MetricHelp({ metric }: { metric: string }) {
  const doc = METRICS[metric]
  const [open, setOpen] = useState(false)
  if (!doc) return null
  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label={`O que é ${doc.name}?`}
        onClick={() => setOpen((o) => !o)}
        onBlur={() => setOpen(false)}
        className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-line text-[10px] leading-none text-muted hover:border-accent hover:text-accent"
      >
        ?
      </button>
      {open && (
        <span className="absolute top-6 left-0 z-30 w-72 rounded border border-line bg-panel p-3 text-left text-[13px] leading-snug text-ink shadow-lg">
          <span className="block font-semibold">{doc.name}</span>
          <span className="num mt-0.5 block font-mono text-[12px] text-muted">{doc.formula}</span>
          <span className="mt-1.5 block">{doc.what}</span>
          <span className="mt-1.5 block text-muted">{doc.read}</span>
        </span>
      )}
    </span>
  )
}

/** Nota didática — aparece apenas com o modo educacional ligado. */
export function Edu({ children, concept, title }: { children?: ReactNode; concept?: keyof typeof CONCEPTS; title?: string }) {
  const { eduMode } = useApp()
  const [expanded, setExpanded] = useState(false)
  if (!eduMode) return null
  const c = concept ? CONCEPTS[concept] : null
  return (
    <aside className="rounded border-l-2 border-accent bg-accent/6 px-3 py-2 text-[13.5px] leading-relaxed">
      <div className="font-semibold text-accent">{title ?? c?.title ?? 'Para entender'}</div>
      {children && <div className="mt-0.5 text-ink/90">{children}</div>}
      {c && (
        <>
          {!children && <p className="mt-0.5 text-ink/90">{c.body[0]}</p>}
          {expanded && (
            <div className="mt-1 space-y-1.5 text-ink/90">
              {c.body.slice(children ? 0 : 1).map((p, i) => (
                <p key={i}>{p}</p>
              ))}
            </div>
          )}
          {(c.body.length > 1 || children) && (
            <button className="mt-1 text-[12.5px] text-accent hover:underline" onClick={() => setExpanded((e) => !e)}>
              {expanded ? 'Mostrar menos' : 'Saiba mais'}
            </button>
          )}
        </>
      )}
    </aside>
  )
}

export function Table({
  head,
  rows,
  className,
  align,
}: {
  head: ReactNode[]
  rows: ReactNode[][]
  className?: string
  align?: ('left' | 'right')[]
}) {
  return (
    <div className={cx('overflow-x-auto rounded border border-line', className)}>
      <table className="w-full border-collapse text-[13.5px]">
        <thead>
          <tr className="bg-panel-2 text-left">
            {head.map((h, i) => (
              <th
                key={i}
                className={cx(
                  'border-b border-line px-3 py-1.5 font-semibold whitespace-nowrap text-muted',
                  align?.[i] === 'right' && 'text-right',
                )}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="num">
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line/60 last:border-0 hover:bg-panel-2/60">
              {r.map((c, j) => (
                <td key={j} className={cx('px-3 py-1.5 whitespace-nowrap', align?.[j] === 'right' && 'text-right')}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function PageHeader({ title, children, actions }: { title: string; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div className="max-w-3xl">
        <h1 className="font-display text-[28px] leading-tight font-semibold">{title}</h1>
        {children && <p className="mt-1 text-[14.5px] text-muted">{children}</p>}
      </div>
      {actions}
    </div>
  )
}
