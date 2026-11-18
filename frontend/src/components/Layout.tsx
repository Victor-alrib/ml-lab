import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import { cx, Select } from './ui'

/** Etapas do fluxo de trabalho, em ordem — desenhadas como uma linha de processo. */
const FLOW = [
  { to: '/datasets', label: 'Datasets', hint: 'Carregar e inspecionar' },
  { to: '/eda', label: 'Análise exploratória', hint: 'Distribuições e relações' },
  { to: '/preprocess', label: 'Pré-processamento', hint: 'Pipeline sem vazamento' },
  { to: '/train', label: 'Treinamento', hint: 'Modelo e hiperparâmetros' },
  { to: '/evaluate', label: 'Avaliação', hint: 'Métricas e diagnósticos' },
]
const EXTRA = [
  { to: '/experiments', label: 'Experimentos', hint: 'Histórico e comparação' },
  { to: '/learn', label: 'Guia de métricas', hint: 'Modo educacional' },
]

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg viewBox="0 0 32 32" className="h-8 w-8" aria-hidden>
        <rect width="32" height="32" rx="6" fill="var(--panel-2)" stroke="var(--line)" />
        <path d="M6 22 L12 14 L17 18 L26 8" stroke="var(--accent)" strokeWidth="2.6" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="26" cy="8" r="2.6" fill="var(--warn)" />
      </svg>
      <div className="leading-none">
        <div className="font-display text-[21px] font-bold tracking-tight">ML Lab</div>
        <div className="mt-0.5 text-[11.5px] text-muted">Machine Learning industrial</div>
      </div>
    </div>
  )
}

function FlowNav({ onNavigate }: { onNavigate?: () => void }) {
  const { pathname } = useLocation()
  const activeIdx = FLOW.findIndex((f) => pathname.startsWith(f.to))
  return (
    <nav aria-label="Módulos" className="flex flex-col gap-6">
      <ol className="relative">
        {FLOW.map((f, i) => {
          const active = i === activeIdx
          const passed = activeIdx >= 0 && i < activeIdx
          return (
            <li key={f.to} className="relative pl-9">
              {i < FLOW.length - 1 && (
                <span
                  aria-hidden
                  className={cx(
                    'absolute top-[22px] left-[13px] h-[calc(100%-6px)] w-[2px]',
                    passed ? 'bg-accent' : 'bg-line',
                  )}
                />
              )}
              <span
                aria-hidden
                className={cx(
                  'num absolute top-[9px] left-[4px] flex h-5 w-5 items-center justify-center rounded-full border-2 text-[10.5px] font-bold',
                  active && 'border-accent bg-accent text-accent-ink',
                  passed && 'border-accent bg-panel text-accent',
                  !active && !passed && 'border-line bg-panel text-muted',
                )}
              >
                {i + 1}
              </span>
              <NavLink
                to={f.to}
                onClick={onNavigate}
                className={cx(
                  'block rounded px-2 py-1.5',
                  active ? 'bg-panel-2' : 'hover:bg-panel-2/60',
                )}
              >
                <span className={cx('block text-[14.5px]', active ? 'font-semibold text-ink' : 'text-ink/85')}>{f.label}</span>
                <span className="block text-[12px] text-muted">{f.hint}</span>
              </NavLink>
            </li>
          )
        })}
      </ol>
      <ul className="flex flex-col gap-0.5 border-t border-line pt-4">
        {EXTRA.map((f) => (
          <li key={f.to}>
            <NavLink
              to={f.to}
              onClick={onNavigate}
              className={({ isActive }) => cx('block rounded px-2 py-1.5', isActive ? 'bg-panel-2' : 'hover:bg-panel-2/60')}
            >
              {({ isActive }) => (
                <>
                  <span className={cx('block text-[14.5px]', isActive ? 'font-semibold' : 'text-ink/85')}>{f.label}</span>
                  <span className="block text-[12px] text-muted">{f.hint}</span>
                </>
              )}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  )
}

export function Layout() {
  const { datasets, datasetId, setDatasetId, theme, toggleTheme, eduMode, setEduMode, apiOnline } = useApp()
  const [menuOpen, setMenuOpen] = useState(false)
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])

  return (
    <div className="flex min-h-full">
      <aside
        className={cx(
          'fixed inset-y-0 left-0 z-40 w-64 shrink-0 overflow-y-auto border-r border-line bg-panel px-4 py-5 transition-transform lg:sticky lg:top-0 lg:h-screen lg:translate-x-0',
          menuOpen ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <div className="mb-7">
          <Logo />
        </div>
        <FlowNav onNavigate={() => setMenuOpen(false)} />
      </aside>
      {menuOpen && <div className="fixed inset-0 z-30 bg-black/40 lg:hidden" onClick={() => setMenuOpen(false)} />}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex flex-wrap items-center gap-3 border-b border-line bg-bg/95 px-4 py-2.5 backdrop-blur lg:px-8">
          <button
            className="rounded border border-line px-2 py-1 text-sm lg:hidden"
            onClick={() => setMenuOpen(true)}
            aria-label="Abrir menu"
          >
            Menu
          </button>
          <div className="flex min-w-0 items-center gap-2">
            <span className="text-[13px] text-muted">Dataset ativo</span>
            {datasets.length ? (
              <Select
                ariaLabel="Dataset ativo"
                className="max-w-[260px] py-1"
                value={datasetId ?? ''}
                onChange={(v) => setDatasetId(v)}
                options={datasets.map((d) => ({ value: d.id, label: d.name }))}
              />
            ) : (
              <span className="text-[13px] text-muted italic">nenhum</span>
            )}
          </div>
          <div className="ml-auto flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-[12.5px] text-muted" title="Status da API">
              <span
                className={cx(
                  'h-2 w-2 rounded-full',
                  apiOnline === null ? 'bg-line' : apiOnline ? 'bg-ok' : 'bg-alarm',
                )}
              />
              {apiOnline === false ? 'API offline' : 'API'}
            </span>
            <label className="flex cursor-pointer items-center gap-2 text-[13px]">
              <input type="checkbox" className="accent-[var(--accent)]" checked={eduMode} onChange={(e) => setEduMode(e.target.checked)} />
              Modo educacional
            </label>
            <button
              onClick={toggleTheme}
              className="rounded border border-line px-2.5 py-1 text-[13px] text-muted hover:text-ink"
              aria-label="Alternar tema"
            >
              {theme === 'dark' ? 'Tema claro' : 'Tema escuro'}
            </button>
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 lg:px-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
