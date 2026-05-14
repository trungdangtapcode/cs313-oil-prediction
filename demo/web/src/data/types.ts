export type Manifest = {
  app_name: string;
  data_version: string;
  report_timestamp: string;
  problem: string;
  target: string;
  best_model: {
    model: string;
    model_config: string;
    experiment: string;
    accuracy: number;
    f1_macro: number;
    auc: number;
    n: number;
    coverage: number;
  };
  counts: {
    leaderboard_rows: number;
    test_days: number;
    feature_rows: number;
    assets: number;
    live_examples?: number;
  };
  local_control_status: {
    github_repo: string;
    gcp_project: string;
    gcp_cli_account: string;
    ci_note: string;
  };
};

export type LeaderboardRow = {
  rank: number;
  Model: string;
  ModelConfig: string;
  Experiment: string;
  ModelType: string;
  Accuracy: number;
  F1_macro: number;
  AUC: number;
  MCC: number;
  Brier: number;
  N: number;
  Coverage: number;
  ThresholdMode: string;
};

export type FeatureRow = {
  rank: number;
  feature: string;
  group: string;
  selected: boolean;
  risk: string;
  why: string;
  MI: number;
  abs_sp: number;
  mix_score: number;
};

export type PredictionDay = {
  date: string;
  target: number;
  models: Record<
    string,
    {
      experiment: string;
      proba_up: number;
      pred_05: number;
      pred_val_threshold: number;
      threshold: number;
    }
  >;
};

export type DecisionRow = {
  date: string;
  target: number;
  actual_label: "UP" | "DOWN";
  proba_up: number;
  pred: number;
  pred_label: "UP" | "DOWN";
  correct: boolean;
  confidence_margin: number;
  val_threshold: number;
  val_threshold_pred: number;
};

export type CurveRow = {
  threshold?: number;
  min_margin?: number;
  coverage_rate?: number;
  accuracy: number | null;
  f1_macro: number | null;
  pos_rate: number | null;
  n: number;
};

export type DatasetStage = {
  file: string;
  rows: number;
  columns: number;
  date_min: string;
  date_max: string;
};

export type AssetItem = {
  group: string;
  name: string;
  file: string;
  path: string;
  source: string;
};

export type MlopsStatus = {
  report_timestamp: string;
  validation_rules: string[];
  source_hashes: Record<string, string>;
  metric_contract: Record<string, unknown>;
};

export type TradingStrategyRow = {
  strategy: string;
  id: string;
  source_model: string;
  threshold: number | null;
  validation_sharpe: number | null;
  validation_return: number | null;
  description: string;
  total_return: number;
  zero_cost_return: number;
  cost_drag: number;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number;
  trades: number;
  turnover: number;
  exposure: number;
  ridge_alpha?: number;
  ridge_val_rmse?: number;
  calibration_intercept?: number;
  calibration_slope?: number;
};

export type TradingYearRow = {
  strategy: string;
  id: string;
  year: string;
  days: number;
  total_return: number;
  zero_cost_return: number;
  cost_drag: number;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number;
  trades: number;
  turnover: number;
  exposure: number;
};

export type TradingSummary = {
  mode: string;
  assumptions: {
    objective: string;
    walk_forward_contract: string;
    prediction_target: string;
    execution_lag_days: number;
    transaction_cost: number;
    reversal_cost_note: string;
    benchmark: string;
  };
  research_notes: string[];
  models: Array<{
    id: string;
    label: string;
    source_model: string;
    description: string;
  }>;
  comparison: TradingStrategyRow[];
  yearly: TradingYearRow[];
  equity_curve: Array<Record<string, number | string>>;
};

export type LiveExample = {
  id: string;
  label: string;
  description: string;
  source_date: string;
  target_date: string;
  features: Record<string, number>;
  actual: {
    target: number;
    label: "UP" | "DOWN";
  };
  prediction: {
    model: string;
    proba_up: number;
    pred: number;
    label: "UP" | "DOWN";
    correct: boolean;
    confidence_margin: number;
  };
};

export type LiveExamplesPayload = {
  mode: string;
  model: string;
  feature_columns: string[];
  rows: LiveExample[];
  note: string;
};

export type FeatureAttribution = {
  feature: string;
  value: number;
  absValue: number;
  direction: "UP" | "DOWN";
  featureValue: number;
};

export type LiveExplanations = {
  note: string;
  shap: {
    method: string;
    baselineProba: number;
    predictionProba: number;
    features: FeatureAttribution[];
  };
  lime: {
    method: string;
    localRows: number;
    fidelityR2: number;
    intercept: number;
    approximationProba: number;
    features: FeatureAttribution[];
  };
};

export type LivePredictionResult = {
  mode: "historical_replay" | "nearest_neighbor_demo_scorer";
  model: string;
  sourceExampleId: string | null;
  targetDate: string;
  sourceDate?: string;
  probaUp: number;
  pred: number;
  predLabel: "UP" | "DOWN";
  confidenceMargin: number;
  actual?: {
    target: number;
    label: "UP" | "DOWN";
  };
  correct?: boolean;
  featureCompleteness?: number;
  providedFeatures?: string[];
  imputedFeatures?: string[];
  audit: string[];
  explanations?: LiveExplanations;
  neighbors: Array<{
    id: string;
    targetDate: string;
    sourceDate: string;
    distance: number;
    probaUp: number;
    predLabel: "UP" | "DOWN";
    actualLabel: "UP" | "DOWN";
    correct: boolean;
  }>;
};

export type DemoData = {
  manifest: Manifest;
  leaderboard: { rows: LeaderboardRow[]; best_by_experiment: LeaderboardRow[] };
  predictions: { models: string[]; rows: PredictionDay[] };
  decisions: { model: string; rows: DecisionRow[] };
  features: { rows: FeatureRow[]; selected: Record<string, unknown> };
  featureGroups: { rows: Array<{ group: string; total: number; selected: number; low_risk: number; features: string[] }> };
  confidence: { model: string; rows: CurveRow[] };
  thresholds: { model: string; rows: CurveRow[] };
  datasets: { rows: DatasetStage[] };
  leakage: { rows: Array<Record<string, string>> };
  assets: { rows: AssetItem[] };
  mlops: MlopsStatus;
  trading: TradingSummary;
};
