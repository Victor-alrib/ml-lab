export type Task = 'classification' | 'regression'

export interface DatasetMeta {
  id: string
  name: string
  source: string
  description?: string | null
  n_rows: number
  n_cols: number
  created_at: string
  size_bytes: number
  parse_info?: { encoding?: string; sep?: string; decimal?: string }
  columns?: string[]
}

export interface SampleInfo {
  key: string
  name: string
  description: string
  task_hint: string
}

export interface NumericStats {
  count: number
  mean: number
  std: number | null
  min: number
  q1: number
  median: number
  q3: number
  max: number
  skewness: number | null
  kurtosis: number | null
  cv_pct: number | null
}

export interface CategoricalStats {
  count: number
  unique: number
  top: string
  top_freq: number
  top_values: { value: string; count: number; pct: number }[]
}

export interface ColumnProfile {
  name: string
  dtype: string
  semantic_type: string
  semantic_label: string
  missing: number
  missing_pct: number
  unique: number
  possible_id: boolean
  stats: Partial<NumericStats & CategoricalStats>
  outliers: null | {
    iqr_count: number
    iqr_pct: number
    z_count: number
    z_pct: number
    lower_bound: number | null
    upper_bound: number | null
  }
}

export interface Profile {
  n_rows: number
  n_cols: number
  memory_mb: number
  duplicated_rows: number
  rows_with_missing: number
  rows_with_missing_pct: number
  total_missing: number
  type_counts: Record<string, number>
  columns: ColumnProfile[]
  warnings: string[]
}

export interface Filter {
  column: string
  op: 'between' | 'in' | 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte' | 'notnull' | 'isnull'
  value?: unknown
}

export interface HyperParam {
  name: string
  label: string
  type: 'int' | 'float' | 'select' | 'bool'
  default: unknown
  help: string
  min?: number
  max?: number
  step?: number
  log?: boolean
  nullable?: boolean
  options?: { value: string; label: string }[]
  depends_on?: Record<string, string[]>
}

export interface ModelInfo {
  key: string
  name: string
  family: string
  tasks: Task[]
  description: string
  params: Partial<Record<Task, HyperParam[]>>
  notes: string[]
}

export interface PreprocessConfig {
  target: string
  task: 'auto' | Task
  features: string[] | null
  numeric_imputation: 'mean' | 'median' | 'most_frequent' | 'constant' | 'drop_rows'
  numeric_fill_value: number
  categorical_imputation: 'most_frequent' | 'constant' | 'drop_rows'
  add_missing_indicator: boolean
  outlier_treatment: 'none' | 'clip_iqr'
  scaling: 'none' | 'standard' | 'minmax' | 'robust'
  encoding: 'onehot' | 'ordinal'
  max_categories: number
  feature_selection: { method: 'none' | 'kbest_f' | 'kbest_mi' | 'variance'; k: number; variance_threshold: number }
  split_strategy: 'random' | 'stratified' | 'temporal'
  time_column: string | null
  test_size: number
  random_state: number
  positive_class: string | null
}

export interface PreviewResult {
  task: Task
  n_train: number
  n_test: number
  split_strategy: string
  numeric_features: string[]
  categorical_features: string[]
  ignored: string[]
  n_features_in: number
  n_features_after_encoding: number
  n_features_selected: number
  selected_features: string[]
  learned_params: {
    column: string
    kind: string
    impute_value: number | string | null
    train_missing: number
    test_missing: number
    scaler?: string
    center?: number
    scale?: number
    clip_lower?: number
    clip_upper?: number
    n_categories?: number
    categories?: string[]
  }[]
  selection_scores: { feature: string; score: number | null; selected: boolean }[] | null
  transformed_sample: { columns: string[]; rows: (number | null)[][] }
  test_transformed_shape: number[]
  notes: string[]
  warnings: string[]
  leakage: { feature: string; score: number; metric: string }[]
  class_labels: string[] | null
  positive_class: string | null
  class_distribution?: { train: ClassDist[]; test: ClassDist[] }
  target_summary?: { train: { mean: number; std: number }; test: { mean: number; std: number } }
}

export interface ClassDist {
  label: string
  count: number
  pct: number
}

export type Metrics = Record<string, number>

export interface CVMetric {
  mean: number
  std: number
  folds?: (number | null)[]
  train_mean: number
  train_std: number
}

export interface Experiment {
  id: string
  name: string
  created_at: string
  dataset_id: string
  dataset_name: string
  task: Task
  model: string
  model_name: string
  family: string
  params: Record<string, unknown>
  preprocessing: PreprocessConfig
  cv_config: { enabled: boolean; folds: number }
  average: string | null
  target: string
  features_used: string[]
  final_features: string[]
  n_train: number
  n_test: number
  split_strategy: string
  metrics: { test: Metrics; train: Metrics; baseline: Metrics }
  primary_metric: string
  cv: null | { splitter: string; folds: number; metrics: Record<string, CVMetric>; fit_time_mean: number }
  learning_curve: null | {
    metric: string
    train_sizes: number[]
    train_mean: number[]
    train_std: number[]
    val_mean: number[]
    val_std: number[]
  }
  importance: null | { kind: string; features: string[]; values: number[] }
  permutation_importance: null | { kind: string; features: string[]; values: number[]; std: number[] }
  coefficients: null | { rows: { label: string; values: number[] }[]; features: string[]; intercept: number[] }
  timing: { fit_s: number; predict_s: number; cv_s: number; total_s: number }
  results: {
    task: Task
    class_labels: string[] | null
    positive_class: string | null
    n_classes: number
    confusion_matrix?: number[][]
    per_class?: { label: string; precision: number; recall: number; f1: number; support: number }[]
    roc_curve?: { fpr: number[]; tpr: number[]; auc: number }
    pr_curve?: { precision: number[]; recall: number[]; prevalence: number }
    proba_hist?: { negative: number[]; positive: number[] }
    threshold_data?: {
      test: { y: number[]; p: number[]; sampled: boolean }
      oof?: { y: number[]; p: number[]; sampled: boolean }
    }
    predictions?: { y_true: number[]; y_pred: number[]; sampled: boolean; n: number }
    residuals?: { values: number[]; mean: number; std: number }
    temporal_test?: { y_true: number[]; y_pred: number[] }
  }
  notes_preprocessing: string[]
  warnings: string[]
  tags: string[]
  notes: string | null
  model_file?: string
  environment: Record<string, string>
}

export type ExperimentSummary = Pick<
  Experiment,
  | 'id'
  | 'name'
  | 'created_at'
  | 'dataset_id'
  | 'dataset_name'
  | 'task'
  | 'model'
  | 'model_name'
  | 'family'
  | 'params'
  | 'target'
  | 'n_train'
  | 'n_test'
  | 'primary_metric'
  | 'timing'
  | 'tags'
  | 'split_strategy'
  | 'average'
  | 'metrics'
  | 'notes'
  | 'preprocessing'
> & {
  cv: null | { folds: number; splitter: string; metrics: Record<string, CVMetric> }
  n_features: number
  has_model: boolean
}

export interface PairTest {
  a: string
  b: string
  a_name: string
  b_name: string
  available: boolean
  reason?: string
  metric?: string
  k?: number
  mean_diff?: number
  folds_diff?: number[]
  t_corrected?: number
  p_corrected?: number
  t_standard?: number
  p_standard?: number
  significant_5pct?: boolean
  better?: string | null
  note?: string | null
}

export interface CompareResult {
  metric: string
  same_task: boolean
  pairs: PairTest[]
  experiments: Experiment[]
}
