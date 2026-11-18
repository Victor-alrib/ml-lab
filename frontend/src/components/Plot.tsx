import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-dist-min'
import { useApp } from '../context/AppContext'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any

export function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export function seriesColors(): string[] {
  return [1, 2, 3, 4, 5, 6].map((i) => cssVar(`--series-${i}`))
}

export function themeColors() {
  return {
    ink: cssVar('--ink'),
    muted: cssVar('--muted'),
    line: cssVar('--line'),
    panel: cssVar('--panel'),
    bg: cssVar('--bg'),
    accent: cssVar('--accent'),
    warn: cssVar('--warn'),
    alarm: cssVar('--alarm'),
    ok: cssVar('--ok'),
    series: seriesColors(),
  }
}

/**
 * Wrapper fino sobre o Plotly. `build` recebe as cores do tema atual e
 * devolve { data, layout } — assim os gráficos acompanham o dark/light mode.
 */
export function Plot({
  build,
  height = 340,
  deps,
  className,
}: {
  build: (c: ReturnType<typeof themeColors>) => { data: Any[]; layout?: Any }
  height?: number
  deps: unknown[]
  className?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { theme } = useApp()

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const c = themeColors()
    const { data, layout = {} } = build(c)
    const axis = (a: Any = {}) => ({
      gridcolor: c.line,
      zerolinecolor: c.line,
      linecolor: c.line,
      tickfont: { color: c.muted, size: 11.5 },
      automargin: true,
      ...a,
      title: typeof a.title === 'string' ? { text: a.title, font: { color: c.muted, size: 12.5 } } : a.title,
    })
    const fullLayout: Any = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Barlow, system-ui, sans-serif', color: c.ink, size: 13 },
      margin: { l: 56, r: 16, t: 24, b: 48 },
      colorway: c.series,
      hoverlabel: { bgcolor: c.panel, bordercolor: c.line, font: { color: c.ink, family: 'Barlow' } },
      legend: { orientation: 'h', y: -0.18, font: { color: c.muted, size: 12 } },
      separators: ',.',
      ...layout,
      xaxis: axis(layout.xaxis),
      yaxis: axis(layout.yaxis),
      ...(layout.xaxis2 ? { xaxis2: axis(layout.xaxis2) } : {}),
      ...(layout.yaxis2 ? { yaxis2: axis(layout.yaxis2) } : {}),
      height,
      autosize: true,
    }
    Plotly.react(el, data, fullLayout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
      locale: 'pt-BR',
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, theme, height])

  useEffect(() => {
    const el = ref.current
    return () => {
      if (el) Plotly.purge(el)
    }
  }, [])

  return <div ref={ref} className={className} style={{ width: '100%', minHeight: height }} />
}
