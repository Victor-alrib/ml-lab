import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Executa uma função assíncrona quando as dependências mudam.
 * Respostas antigas são descartadas e `data` só é devolvido se corresponder
 * às dependências atuais (evita exibir resultado de outro dataset).
 */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[], enabled = true) {
  const key = JSON.stringify(deps)
  const [state, setState] = useState<{ key: string; data: T | null }>({ key: '', data: null })
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const seq = useRef(0)

  const run = useCallback(async () => {
    const id = ++seq.current
    setLoading(true)
    setError(null)
    try {
      const r = await fn()
      if (id === seq.current) setState({ key, data: r })
    } catch (e) {
      if (id === seq.current) {
        setError((e as Error).message)
        setState({ key, data: null })
      }
    } finally {
      if (id === seq.current) setLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  useEffect(() => {
    if (enabled) run()
  }, [run, enabled])

  const fresh = state.key === key
  return {
    data: fresh ? state.data : null,
    /** último resultado, mesmo que de dependências anteriores (útil para evitar "piscar") */
    stale: state.data,
    error,
    loading: loading || (enabled && !fresh),
    reload: run,
    setData: (d: T | null) => setState({ key, data: d }),
  }
}
