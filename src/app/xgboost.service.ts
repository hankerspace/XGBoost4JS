import { Injectable } from '@angular/core';
import {
  XGBoost, XGBoostParams,
  generateMultivariateSeries, prepareMultivariateDataset, calcMetrics,
  getMultivariateFeatureNames
} from 'xgboost-js';

@Injectable({ providedIn: 'root' })
export class XGBoostService {

  /**
   * Generates a multivariate time series.
   * @param length Total length
   * @param generators List of generator functions for each feature
   */
  generateMultivariate(length: number, generators: ((t: number) => number)[]) {
    return generateMultivariateSeries(length, generators);
  }

  /**
   * Trains an XGBoost model to predict a target feature based on other features and lags.
   * @param params XGBoost Parameters
   * @param data Full Multivariate Data (N x M)
   * @param targetIndex Index of the feature to predict
   * @param lag Number of lag steps
   * @param trainRatio Ratio of data to use for training (e.g. 0.8)
   */
  trainAndEvaluate(
    params: XGBoostParams,
    data: number[][],
    targetIndex: number,
    lag: number,
    trainRatio: number = 0.8
  ) {
    // 1. Prepare Dataset (X, y)
    const { X, y } = prepareMultivariateDataset(data, targetIndex, lag, true);

    if (X.length === 0) {
      throw new Error("Not enough data for lag " + lag);
    }

    // 2. Split Train/Test
    const splitIdx = Math.floor(X.length * trainRatio);

    const X_train = X.slice(0, splitIdx);
    const y_train = y.slice(0, splitIdx);
    const X_test = X.slice(splitIdx);
    const y_test = y.slice(splitIdx);

    // 3. Train
    const model = new XGBoost({ ...params, objective: 'reg:squarederror' });
    model.fit(X_train, y_train);

    // 4. Predict (Test set)
    const preds = model.predict(X_test);

    // 5. Metrics
    const metrics = calcMetrics(preds, y_test);

    // 6. Feature Importance & Names
    const importance = model.getFeatureImportance();
    const numFeatures = data[0].length;
    const featureNames = getMultivariateFeatureNames(numFeatures, targetIndex, lag, 'F', true);

    return {
      model,
      metrics,
      importance,
      featureNames,
      testResults: {
        actual: y_test,
        predicted: preds
      },
      trainSize: X_train.length,
      testSize: X_test.length,
      splitIdx // Useful to align timestamps
    };
  }
}
