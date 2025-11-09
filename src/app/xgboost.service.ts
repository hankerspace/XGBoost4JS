import { Injectable } from '@angular/core';
import { XGBoost, XGBoostParams, DatasetType, generateDataset, metrics, generateTimeSeries, windowedSupervised, recursiveForecast, SeriesType, TimestampTrainingData, prepareTimestampFeatures, extractTimestampFeatures, timestampFeaturesToArray, generateTimeSeriesWithTimestamps } from './xgboost';

export interface TrainResult {
  model: XGBoost;
  predictions: number[];
  metrics: ReturnType<typeof metrics>;
  importance: number[];
}

export interface TimestampTrainResult {
  model: XGBoost;
  predictions: number[];
  metrics?: ReturnType<typeof metrics>;
  importance: number[];
}

@Injectable({ providedIn: 'root' })
export class XGBoostService {
  generate(type: DatasetType, n: number) {
    return generateDataset(type, n);
  }

  trainAndPredict(params: XGBoostParams, train: { X: number[][]; y: number[] }, test: { X: number[][]; y: number[] }): TrainResult {
    const model = new XGBoost(params);
    model.fit(train.X, train.y);
    const predictions = model.predictBatch(test.X);
    const m = metrics(predictions, test.y);
    const importance = model.getFeatureImportance();
    return { model, predictions, metrics: m, importance };
  }

  /**
   * Train and predict using timestamp-based features
   * @param params - XGBoost parameters
   * @param trainData - Training data with timestamps and target values
   * @param testData - Test data with timestamps and target values
   * @returns Training result with model, predictions, and metrics
   */
  trainAndPredictWithTimestamps(
    params: XGBoostParams,
    trainData: TimestampTrainingData,
    testData: TimestampTrainingData
  ): TimestampTrainResult {
    const model = new XGBoost(params);
    model.fitWithTimestamps(trainData);
    const predictions = model.predictBatchWithTimestamps(testData.timestamps, testData.customFeatures);
    const importance = model.getFeatureImportance();
    
    // Calculate metrics if task is binary classification
    if (params.task === 'binary') {
      const m = metrics(predictions, testData.y);
      return { model, predictions, metrics: m, importance };
    }
    
    return { model, predictions, importance };
  }

  /**
   * Expose timestamp feature extraction utilities
   */
  extractTimestampFeatures = extractTimestampFeatures;
  timestampFeaturesToArray = timestampFeaturesToArray;
  prepareTimestampFeatures = prepareTimestampFeatures;

  // ====== API Séries temporelles ======
  generateSeries(
    type: SeriesType,
    total: number,
    options?: { amplitude?: number; frequency?: number; phase?: number; noise?: number; trendSlope?: number; drift?: number }
  ) {
    return generateTimeSeries(type, total, options);
  }

  /**
   * Generate time series with timestamps and optional custom features
   */
  generateSeriesWithTimestamps(
    type: SeriesType,
    total: number,
    startDate: Date,
    intervalMs: number = 3600000,
    options?: {
      amplitude?: number;
      frequency?: number;
      phase?: number;
      noise?: number;
      trendSlope?: number;
      drift?: number;
      generateCustomFeatures?: boolean;
    }
  ) {
    return generateTimeSeriesWithTimestamps(type, total, startDate, intervalMs, options);
  }

  /**
   * Train and predict time series using timestamps and automatic feature extraction
   * @param params - XGBoost parameters
   * @param timestamps - Array of timestamps for the entire series
   * @param values - Array of values for the entire series
   * @param trainLen - Number of points to use for training
   * @param customFeatures - Optional custom features for each timestamp
   * @returns Training result with predictions and metrics
   */
  trainAndForecastWithTimestamps(
    params: XGBoostParams,
    timestamps: Date[],
    values: number[],
    trainLen: number,
    customFeatures?: number[][]
  ) {
    const model = new XGBoost({ ...params, task: 'regression' });
    
    // Prepare training data with timestamps
    const trainTimestamps = timestamps.slice(0, trainLen);
    const trainValues = values.slice(0, trainLen);
    const trainCustomFeatures = customFeatures ? customFeatures.slice(0, trainLen) : undefined;
    
    const trainData: TimestampTrainingData = {
      timestamps: trainTimestamps,
      y: trainValues,
      customFeatures: trainCustomFeatures
    };
    
    model.fitWithTimestamps(trainData);
    
    // Prepare test data
    const testTimestamps = timestamps.slice(trainLen);
    const testCustomFeatures = customFeatures ? customFeatures.slice(trainLen) : undefined;
    
    // Make predictions
    const preds = model.predictBatchWithTimestamps(testTimestamps, testCustomFeatures);
    const truth = values.slice(trainLen);
    
    // Calculate regression metrics
    const n = truth.length;
    const eps = 1e-8;
    const errors = truth.map((v, i) => preds[i] - v);
    const absErrors = errors.map((e) => Math.abs(e));
    const se = errors.map((e) => e * e);
    const mae = n ? absErrors.reduce((a, b) => a + b, 0) / n : 0;
    const mse = n ? se.reduce((a, b) => a + b, 0) / n : 0;
    const rmse = Math.sqrt(mse);
    const bias = n ? errors.reduce((a, b) => a + b, 0) / n : 0;
    const maxAe = n ? Math.max(...absErrors) : 0;
    const mape = n
      ? (100 * truth.reduce((acc, v, i) => acc + Math.abs(errors[i]) / Math.max(eps, Math.abs(v)), 0)) / n
      : 0;
    const meanY = n ? truth.reduce((a, b) => a + b, 0) / n : 0;
    const sst = n ? truth.reduce((acc, v) => acc + Math.pow(v - meanY, 2), 0) : 0;
    const sse = n ? se.reduce((a, b) => a + b, 0) : 0;
    const r2 = sst > eps ? 1 - sse / sst : 0;
    
    return { model, preds, truth, mae, rmse, mse, mape, r2, bias, maxAe };
  }

  /** Entraîne un modèle de régression sur une série glissante et prédit les steps suivants en mode récursif. */
  trainAndForecastTS(params: XGBoostParams, series: number[], lag: number, trainLen: number, forecastLen: number) {
    const model = new XGBoost({ ...params, task: 'regression' });
    // Construire l'ensemble supervisé sur la fenêtre d'apprentissage
    const trainSlice = series.slice(0, trainLen);
    const { X, y } = windowedSupervised(trainSlice, lag);
    model.fit(X, y);

    // Démarre la prédiction depuis les dernières valeurs réelles connues
    const seed = series.slice(trainLen - lag, trainLen);
    const preds = recursiveForecast(model, seed, forecastLen);
    const truth = series.slice(trainLen, trainLen + forecastLen);

    // Calculer des métriques régression détaillées
    const n = truth.length;
    const eps = 1e-8;
    const errors = truth.map((v, i) => preds[i] - v);
    const absErrors = errors.map((e) => Math.abs(e));
    const se = errors.map((e) => e * e);
    const mae = n ? absErrors.reduce((a, b) => a + b, 0) / n : 0;
    const mse = n ? se.reduce((a, b) => a + b, 0) / n : 0;
    const rmse = Math.sqrt(mse);
    const bias = n ? errors.reduce((a, b) => a + b, 0) / n : 0;
    const maxAe = n ? Math.max(...absErrors) : 0;
    const mape = n
      ? (100 * truth.reduce((acc, v, i) => acc + Math.abs(errors[i]) / Math.max(eps, Math.abs(v)), 0)) / n
      : 0;
    const meanY = n ? truth.reduce((a, b) => a + b, 0) / n : 0;
    const sst = n ? truth.reduce((acc, v) => acc + Math.pow(v - meanY, 2), 0) : 0;
    const sse = n ? se.reduce((a, b) => a + b, 0) : 0;
    const r2 = sst > eps ? 1 - sse / sst : 0;

    return { model, preds, truth, mae, rmse, mse, mape, r2, bias, maxAe };
  }

  /**
   * Recherche simple d'hyperparamètres via grille sur une validation temporelle (holdout).
   * On entraîne sur la partie initiale du train (70%) et on prédit récursivement la fin (30%).
   * Les candidats sont fournis sous forme de listes de nombres.
   */
  async tuneHyperparamsTS(
    space: { learningRates: number[]; maxDepths: number[]; minChildWeights: number[]; rounds: number[] },
    series: number[],
    lag: number,
    trainLen: number,
    maxCombos = 120,
    onProgress?: (done: number, total: number) => void
  ): Promise<{ params: XGBoostParams; mae: number; rmse: number; bias: number }[]> {
    const innerTrainLen = Math.max(lag + 10, Math.floor(trainLen * 0.7));
    const valLen = Math.max(5, trainLen - innerTrainLen);
    const leaderboard: { params: XGBoostParams; mae: number; rmse: number; bias: number }[] = [];

    // Calcul du nombre total de combinaisons et plafonnement par maxCombos
    const totalCombos =
      (space.learningRates?.length || 0) *
      (space.maxDepths?.length || 0) *
      (space.minChildWeights?.length || 0) *
      (space.rounds?.length || 0);
    const total = Math.min(totalCombos, Math.max(0, maxCombos));

    let tested = 0;
    const yieldEvery = 10; // Laisser respirer l’UI périodiquement

    outer: for (const lr of space.learningRates) {
      for (const depth of space.maxDepths) {
        for (const mcw of space.minChildWeights) {
          for (const rounds of space.rounds) {
            if (tested >= total) break outer;

            const params: XGBoostParams = {
              task: 'regression',
              learningRate: lr,
              maxDepth: Math.round(depth),
              minChildWeight: Math.round(mcw),
              numRounds: Math.round(rounds),
            };

            // Construire ensemble supervisé sur la sous-fenêtre d'apprentissage interne
            const base = series.slice(0, innerTrainLen);
            const { X, y } = windowedSupervised(base, lag);
            const model = new XGBoost(params);
            model.fit(X, y);

            // Prévoir la validation à partir des dernières valeurs connues
            const seed = series.slice(innerTrainLen - lag, innerTrainLen);
            const preds = recursiveForecast(model, seed, valLen);
            const truth = series.slice(innerTrainLen, innerTrainLen + valLen);

            const n = truth.length;
            const mae = n ? truth.reduce((a, v, i) => a + Math.abs(v - preds[i]), 0) / n : Number.POSITIVE_INFINITY;
            const rmse = n ? Math.sqrt(truth.reduce((a, v, i) => a + Math.pow(v - preds[i], 2), 0) / n) : Number.POSITIVE_INFINITY;
            const bias = n ? preds.reduce((a, p, i) => a + (p - truth[i]), 0) / n : Number.POSITIVE_INFINITY;
            leaderboard.push({ params, mae, rmse, bias });

            tested++;
            if (onProgress) onProgress(tested, total);
            if (tested % yieldEvery === 0) {
              await new Promise<void>((r) => setTimeout(r));
            }
          }
        }
      }
    }

    // Tri: |biais| minimal (meilleur), puis RMSE puis MAE
    leaderboard.sort((a, b) => {
      const ab = Math.abs(a.bias) - Math.abs(b.bias);
      if (ab !== 0) return ab;
      const r = a.rmse - b.rmse;
      if (r !== 0) return r;
      return a.mae - b.mae;
    });
    return leaderboard.slice(0, 10);
  }
}
