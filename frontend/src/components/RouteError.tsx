import { Link, useRouteError } from 'react-router-dom'

export function RouteError() {
  const err = useRouteError() as Error | undefined
  return (
    <div className="mx-auto max-w-2xl px-6 py-16">
      <h1 className="font-display text-[26px] font-semibold">Este módulo encontrou um erro</h1>
      <p className="mt-2 text-muted">
        A interface falhou ao exibir esta tela. Os datasets e experimentos salvos não foram afetados.
      </p>
      {err?.message && <pre className="mt-4 overflow-x-auto rounded border border-line bg-panel p-3 font-mono text-[13px] text-alarm">{err.message}</pre>}
      <div className="mt-5 flex gap-4 text-accent">
        <button className="hover:underline" onClick={() => window.location.reload()}>
          Recarregar a página
        </button>
        <Link className="hover:underline" to="/datasets">
          Voltar para Datasets
        </Link>
      </div>
    </div>
  )
}
