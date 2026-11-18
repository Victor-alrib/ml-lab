import type { Experiment } from '../api/types'
import { fmt } from './format'

export interface MetricDoc {
  key: string
  name: string
  task: 'classification' | 'regression'
  formula: string
  what: string
  read: string
  industry: string
  higherIsBetter: boolean
  range: string
}

export const METRICS: Record<string, MetricDoc> = {
  accuracy: {
    key: 'accuracy', name: 'Accuracy', task: 'classification', higherIsBetter: true, range: '0 a 1',
    formula: '(VP + VN) / total',
    what: 'Fração de previsões corretas, somando todas as classes.',
    read: 'Só é confiável com classes equilibradas. Se 92% das leituras são "sem falha", um modelo que nunca prevê falha já tem 0,92.',
    industry: 'Em manutenção preditiva quase nunca é a métrica principal: falhas são raras e a accuracy esconde isso.',
  },
  balanced_accuracy: {
    key: 'balanced_accuracy', name: 'Balanced accuracy', task: 'classification', higherIsBetter: true, range: '0 a 1',
    formula: 'média do recall de cada classe',
    what: 'Média do acerto dentro de cada classe; cada classe pesa igual, não importa o tamanho.',
    read: '0,5 em problema binário equivale a chutar. Bom substituto da accuracy em dados desbalanceados.',
    industry: 'Útil para comparar modelos de falha quando a proporção de eventos varia entre linhas ou plantas.',
  },
  precision: {
    key: 'precision', name: 'Precision', task: 'classification', higherIsBetter: true, range: '0 a 1',
    formula: 'VP / (VP + FP)',
    what: 'Dos alarmes que o modelo disparou, quantos eram reais.',
    read: 'Precision baixa = muitos falsos alarmes. A equipe de manutenção perde confiança e passa a ignorar o sistema.',
    industry: 'Priorize quando cada intervenção é cara (parar uma linha, deslocar técnico).',
  },
  recall: {
    key: 'recall', name: 'Recall (sensibilidade)', task: 'classification', higherIsBetter: true, range: '0 a 1',
    formula: 'VP / (VP + FN)',
    what: 'Das falhas reais, quantas o modelo detectou.',
    read: 'Recall baixo = falhas passando despercebidas (falsos negativos).',
    industry: 'Priorize quando uma falha não detectada é grave: segurança, parada não programada, dano ao equipamento.',
  },
  f1: {
    key: 'f1', name: 'F1-Score', task: 'classification', higherIsBetter: true, range: '0 a 1',
    formula: '2 · P · R / (P + R)',
    what: 'Média harmônica entre precision e recall. Cai muito se qualquer uma das duas for baixa.',
    read: 'Bom resumo para classes desbalanceadas, mas assume que FP e FN custam o mesmo.',
    industry: 'Métrica padrão deste app para classificação. Se os custos de FP e FN forem muito diferentes, ajuste o limiar.',
  },
  roc_auc: {
    key: 'roc_auc', name: 'ROC AUC', task: 'classification', higherIsBetter: true, range: '0,5 (aleatório) a 1',
    formula: 'área sob a curva TPR × FPR',
    what: 'Probabilidade de o modelo dar escore maior a um positivo do que a um negativo sorteados ao acaso.',
    read: 'Independe do limiar. Com classes muito raras pode parecer otimista — olhe também a curva Precision-Recall.',
    industry: 'Mede a capacidade de ranquear equipamentos por risco, útil para priorizar a fila de inspeção.',
  },
  log_loss: {
    key: 'log_loss', name: 'Log loss', task: 'classification', higherIsBetter: false, range: '0 a ∞',
    formula: '−média[y·log(p) + (1−y)·log(1−p)]',
    what: 'Penaliza probabilidades confiantes e erradas.',
    read: 'Avalia a qualidade das probabilidades, não só da classe prevista.',
    industry: 'Importa quando a probabilidade alimenta uma decisão (ex.: custo esperado de falha).',
  },
  mae: {
    key: 'mae', name: 'MAE', task: 'regression', higherIsBetter: false, range: '0 a ∞ (unidade do alvo)',
    formula: 'média(|y − ŷ|)',
    what: 'Erro absoluto médio, na mesma unidade do alvo.',
    read: 'MAE = 2,3 em temperatura significa que, em média, a previsão erra 2,3 °C. Robusto a erros isolados grandes.',
    industry: 'Fácil de comunicar para operação e manutenção.',
  },
  rmse: {
    key: 'rmse', name: 'RMSE', task: 'regression', higherIsBetter: false, range: '0 a ∞ (unidade do alvo)',
    formula: '√média((y − ŷ)²)',
    what: 'Raiz do erro quadrático médio; erros grandes pesam mais.',
    read: 'Sempre ≥ MAE. Se RMSE for bem maior que o MAE, existem alguns erros grandes.',
    industry: 'Prefira quando erros grandes são desproporcionalmente perigosos (ex.: subestimar temperatura de rolamento).',
  },
  mse: {
    key: 'mse', name: 'MSE', task: 'regression', higherIsBetter: false, range: '0 a ∞ (unidade²)',
    formula: 'média((y − ŷ)²)',
    what: 'Erro quadrático médio. É o que a regressão linear minimiza.',
    read: 'Fica na unidade ao quadrado; por isso o RMSE é mais usado para interpretar.',
    industry: '—',
  },
  r2: {
    key: 'r2', name: 'R²', task: 'regression', higherIsBetter: true, range: '−∞ a 1',
    formula: '1 − SQres / SQtot',
    what: 'Fração da variância do alvo explicada pelo modelo, comparada a prever sempre a média.',
    read: 'R² = 0 equivale a prever a média; negativo é pior que isso. Não diz se o erro é aceitável na prática.',
    industry: 'Complemente sempre com MAE/RMSE na unidade física.',
  },
  mape: {
    key: 'mape', name: 'MAPE', task: 'regression', higherIsBetter: false, range: '0 a ∞ (%)',
    formula: 'média(|y − ŷ| / |y|) · 100',
    what: 'Erro percentual médio.',
    read: 'Explode quando o alvo é próximo de zero; só é calculado quando não há zeros no alvo.',
    industry: 'Bom para alvos sempre positivos e longe de zero (consumo de energia, produção).',
  },
}

export interface ConceptDoc {
  title: string
  body: string[]
}

export const CONCEPTS: Record<string, ConceptDoc> = {
  leakage: {
    title: 'Vazamento de dados (data leakage)',
    body: [
      'Acontece quando o modelo recebe, no treino, alguma informação que não estaria disponível no momento real da previsão. O resultado é um desempenho excelente no papel e ruim em produção.',
      'Vazamento de pré-processamento: calcular média, desvio ou categorias com o dataset inteiro antes de separar treino e teste. Neste app, todo passo que aprende estatística fica dentro de um Pipeline ajustado só no treino — e, na validação cruzada, refeito em cada fold.',
      'Vazamento de atributo: usar uma variável derivada do alvo ou registrada depois do evento (ex.: "modo de falha" para prever "houve falha"). O app testa cada atributo isoladamente e alerta quando um só já prevê o alvo quase perfeitamente.',
      'Vazamento temporal: embaralhar séries temporais deixa o modelo "ver o futuro". Use a separação temporal quando a ordem dos dados importa.',
    ],
  },
  split: {
    title: 'Treino, validação e teste',
    body: [
      'Treino: dados usados para ajustar o modelo e o pré-processamento.',
      'Validação (cruzada): avaliação repetida dentro do treino, usada para comparar configurações e escolher hiperparâmetros.',
      'Teste: separado no início e usado uma única vez, no fim. Se você ajusta hiperparâmetros olhando o teste, ele deixa de ser uma estimativa honesta.',
    ],
  },
  cv: {
    title: 'Validação cruzada (k-fold)',
    body: [
      'O treino é dividido em k partes. O modelo é treinado k vezes, cada vez avaliado em uma parte diferente. A média mostra o desempenho esperado; o desvio padrão mostra a estabilidade.',
      'StratifiedKFold mantém a proporção das classes em cada fold. TimeSeriesSplit sempre treina no passado e valida no futuro.',
      'Desvio padrão alto entre folds indica que o resultado depende muito de quais amostras caíram no teste — desconfie de diferenças pequenas entre modelos.',
    ],
  },
  overfitting: {
    title: 'Overfitting e underfitting',
    body: [
      'Overfitting: o modelo decora o treino. Sinal: métrica de treino muito melhor que a de validação/teste. Remédios: limitar profundidade, aumentar min_samples_leaf, regularizar, mais dados.',
      'Underfitting: o modelo é simples demais. Sinal: treino e validação ruins e próximos. Remédios: modelo mais flexível, mais atributos, menos regularização.',
      'A curva de aprendizado mostra os dois: se as curvas ainda estão se aproximando no fim, mais dados devem ajudar.',
    ],
  },
  baseline: {
    title: 'Baseline',
    body: [
      'Um modelo ingênuo treinado nas mesmas condições: prever sempre a classe mais frequente (classificação) ou a média do treino (regressão).',
      'Se o seu modelo não supera o baseline com folga, ele não aprendeu nada útil — por mais alta que pareça a accuracy.',
    ],
  },
  threshold: {
    title: 'Limiar de decisão',
    body: [
      'Classificadores binários geram uma probabilidade; a classe é decidida comparando com um limiar (padrão 0,5).',
      'Baixar o limiar aumenta o recall (pega mais falhas) e reduz a precision (mais falsos alarmes). A escolha depende do custo de cada erro.',
      'Escolha o limiar usando as previsões out-of-fold da validação cruzada, não o conjunto de teste — senão o teste deixa de ser independente.',
    ],
  },
  significance: {
    title: 'Diferença estatisticamente significativa',
    body: [
      'Dois modelos avaliados nos mesmos folds podem ser comparados fold a fold. O app usa o t-test pareado com correção de Nadeau & Bengio, que compensa a sobreposição entre os conjuntos de treino dos folds.',
      'p < 0,05 sugere que a diferença não é só ruído da amostragem. p alto não prova que os modelos são iguais — só que estes dados não permitem distinguir.',
    ],
  },
}

export interface Insight {
  level: 'ok' | 'warn' | 'alarm' | 'info'
  text: string
}

/** Gera interpretações a partir dos valores REAIS de um experimento (regras simples e explícitas). */
export function interpret(exp: Experiment): Insight[] {
  const out: Insight[] = []
  const t = exp.metrics.test
  const tr = exp.metrics.train
  const b = exp.metrics.baseline
  if (exp.task === 'classification') {
    const pos = exp.results.positive_class
    if (exp.results.n_classes === 2 && pos) {
      out.push({
        level: 'info',
        text: `Classe positiva: "${pos}". Precision, recall e F1 se referem a ela.`,
      })
    }
    const gap = tr.f1 - t.f1
    if (gap > 0.1)
      out.push({ level: 'warn', text: `F1 no treino (${fmt(tr.f1)}) é ${fmt(gap)} maior que no teste (${fmt(t.f1)}): sinal de overfitting.` })
    else out.push({ level: 'ok', text: `F1 de treino (${fmt(tr.f1)}) e teste (${fmt(t.f1)}) estão próximos: sem sinal forte de overfitting.` })
    if (t.accuracy - b.accuracy < 0.01 && exp.results.n_classes === 2)
      out.push({
        level: 'alarm',
        text: `Accuracy do modelo (${fmt(t.accuracy)}) não supera a do baseline que sempre prevê a classe majoritária (${fmt(b.accuracy)}). Avalie por recall, F1 e balanced accuracy.`,
      })
    if (t.balanced_accuracy < 0.6)
      out.push({ level: 'warn', text: `Balanced accuracy de ${fmt(t.balanced_accuracy)} — pouco acima do acaso (0,5 no caso binário).` })
    if (t.precision > 0 && t.recall / t.precision > 2)
      out.push({ level: 'info', text: `Recall (${fmt(t.recall)}) bem maior que precision (${fmt(t.precision)}): o modelo pega muitas ocorrências, com muitos falsos alarmes.` })
    if (t.recall > 0 && t.precision / t.recall > 2)
      out.push({ level: 'info', text: `Precision (${fmt(t.precision)}) bem maior que recall (${fmt(t.recall)}): os alarmes são confiáveis, mas muitas ocorrências passam despercebidas. Teste um limiar menor ou peso balanceado de classes.` })
    if (t.f1 > 0.995 || t.roc_auc > 0.999)
      out.push({ level: 'alarm', text: 'Desempenho praticamente perfeito. Antes de comemorar, verifique os alertas de vazamento: algum atributo pode estar entregando a resposta.' })
  } else {
    const r2gap = tr.r2 - t.r2
    if (r2gap > 0.1)
      out.push({ level: 'warn', text: `R² no treino (${fmt(tr.r2)}) é bem maior que no teste (${fmt(t.r2)}): sinal de overfitting.` })
    else out.push({ level: 'ok', text: `R² de treino (${fmt(tr.r2)}) e teste (${fmt(t.r2)}) estão próximos.` })
    if (t.r2 < 0) out.push({ level: 'alarm', text: `R² negativo (${fmt(t.r2)}): o modelo é pior do que prever sempre a média do treino.` })
    const imp = 1 - t.rmse / b.rmse
    out.push({
      level: imp > 0.1 ? 'ok' : 'warn',
      text: `RMSE ${fmt(t.rmse)} contra ${fmt(b.rmse)} do baseline (média do treino): redução de ${fmt(imp * 100, 1)}%.`,
    })
    if (t.rmse > 1.5 * t.mae)
      out.push({ level: 'info', text: `RMSE (${fmt(t.rmse)}) bem acima do MAE (${fmt(t.mae)}): poucos erros grandes concentram o erro. Veja o gráfico de resíduos.` })
    if (t.r2 > 0.999) out.push({ level: 'alarm', text: 'R² praticamente perfeito: verifique os alertas de vazamento.' })
  }
  if (exp.cv) {
    const pm = exp.primary_metric
    const m = exp.cv.metrics[pm]
    if (m && Math.abs(m.mean) > 0 && m.std / Math.abs(m.mean) > 0.15)
      out.push({
        level: 'warn',
        text: `${METRICS[pm]?.name ?? pm} varia bastante entre folds (${fmt(m.mean)} ± ${fmt(m.std)}): o resultado é instável; diferenças pequenas entre modelos podem ser ruído.`,
      })
  }
  return out
}
