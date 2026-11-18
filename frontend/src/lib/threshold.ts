/** Métricas binárias para um limiar — calculadas sobre as probabilidades reais do experimento. */
export interface ThresholdMetrics {
  t: number
  tp: number
  fp: number
  tn: number
  fn: number
  precision: number
  recall: number
  f1: number
  accuracy: number
  specificity: number
}

export function metricsAt(y: number[], p: number[], t: number): ThresholdMetrics {
  let tp = 0,
    fp = 0,
    tn = 0,
    fn = 0
  for (let i = 0; i < y.length; i++) {
    const pred = p[i] >= t ? 1 : 0
    if (pred === 1 && y[i] === 1) tp++
    else if (pred === 1) fp++
    else if (y[i] === 1) fn++
    else tn++
  }
  const precision = tp + fp ? tp / (tp + fp) : 0
  const recall = tp + fn ? tp / (tp + fn) : 0
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0
  return { t, tp, fp, tn, fn, precision, recall, f1, accuracy: (tp + tn) / y.length, specificity: tn + fp ? tn / (tn + fp) : 0 }
}

export function sweep(y: number[], p: number[], steps = 99): ThresholdMetrics[] {
  return Array.from({ length: steps }, (_, i) => metricsAt(y, p, (i + 1) / (steps + 1)))
}

export function bestF1(curve: ThresholdMetrics[]) {
  return curve.reduce((a, b) => (b.f1 > a.f1 ? b : a), curve[0])
}
