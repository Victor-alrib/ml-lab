import { api } from '../api/client'
import type { Profile } from '../api/types'
import { useAsync } from './useAsync'

const cache = new Map<string, Profile>()

/** Perfil do dataset (tipos, estatísticas) com cache por sessão. */
export function useProfile(datasetId: string | null) {
  return useAsync<Profile | null>(
    async () => {
      if (!datasetId) return null
      const hit = cache.get(datasetId)
      if (hit) return hit
      const p = await api.profile(datasetId)
      cache.set(datasetId, p)
      return p
    },
    [datasetId],
  )
}

export const isNumericType = (t: string) => t === 'numerica_continua' || t === 'numerica_discreta'

export function numericColumns(p: Profile | null, includeBinary = true) {
  if (!p) return []
  return p.columns
    .filter((c) => isNumericType(c.semantic_type) || (includeBinary && c.semantic_type === 'binaria' && c.stats?.mean !== undefined))
    .map((c) => c.name)
}

export function groupableColumns(p: Profile | null) {
  if (!p) return []
  return p.columns.filter((c) => ['categorica', 'binaria', 'numerica_discreta'].includes(c.semantic_type) && c.unique <= 12).map((c) => c.name)
}
