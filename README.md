# XGBoost4JS

A pure TypeScript implementation of the **XGBoost** algorithm (eXtreme Gradient Boosting). This library allows you to train gradient boosting models and perform inference directly in the browser or in a Node.js environment, without any native dependencies.

## Features

- **Zero external dependencies**: Written entirely in TypeScript.
- **Lightweight**: Designed to be executed client-side.
- **Supports**:
  - Regression (`reg:squarederror`)
  - Binary classification (`binary:logistic`)
- **Advanced features**:
  - Feature Importance (Gain)
  - Time series handling (utilities included)

## Installation

Currently, the library is not published on npm. You can integrate `XGBoost4JS` into your project by simply copying the file `src/app/xgboost.ts`.

1. Copy `src/app/xgboost.ts` into your project (for example, in a `src/utils/` folder).
2. Import `XGBoost` where you need it.

## Usage

### 1. Import

```typescript
import { XGBoost, XGBoostParams } from './xgboost';
```

### 2. Data Preparation

The data must be formatted as follows:
- **X**: An array of number arrays (`number[][]`), where each sub-array represents a sample row (features).
- **y**: An array of numbers (`number[]`) representing the labels or target values.

```typescript
// Simple data example (Approximate logical XOR for demo)
const X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
];
const y = [0, 1, 1, 0];
```

### 3. Configuration and Training

```typescript
const params: XGBoostParams = {
  n_estimators: 10,       // Number of trees
  learning_rate: 0.3,     // Learning rate (eta)
  max_depth: 3,           // Maximum depth of trees
  min_child_weight: 1,    // Minimum child weight
  gamma: 0,               // Minimum loss reduction required to make a further partition
  lambda: 1,              // L2 regularization
  objective: 'binary:logistic', // 'reg:squarederror' for regression
  seed: 42                // Seed for reproducibility
};

const model = new XGBoost(params);
model.fit(X, y);
```

### 4. Prediction

```typescript
const predictions = model.predict(X);
console.log(predictions); 
// Output: [0.05, 0.95, 0.95, 0.05] (approximate values)
```

### 5. Feature Importance

You can retrieve the importance of each feature (based on gain) after training:

```typescript
const importance = model.getFeatureImportance();
// Returns a Map where the key is the feature index and the value is its importance
importance.forEach((gain, featureIndex) => {
  console.log(`Feature ${featureIndex}: ${gain}`);
});
```

## API

### `XGBoost` Class

#### Constructor
`new XGBoost(params: XGBoostParams)`

#### Methods
- `fit(X: number[][], y: number[]): void`: Trains the model on the provided data.
- `predict(X: number[][]): number[]`: Predicts values for the new data.
- `getFeatureImportance(): Map<string, number>`: Returns the feature importance.

### Time Series Utilities

The library also includes functions to facilitate the preparation of time series data (in `xgboost.ts`):

- `prepareMultivariateDataset`: Transforms a raw time series into a supervised dataset (X, y) with lag handling.
- `generateMultivariateSeries`: Generates synthetic data for testing.

## Example with Angular Service

The project includes an integration example in `src/app/xgboost.service.ts` which shows how to encapsulate the logic for time series predictions.

```typescript
// Extract from xgboost.service.ts
trainAndEvaluate(params, data, targetIndex, lag, trainRatio) {
    const { X, y } = prepareMultivariateDataset(data, targetIndex, lag, true);
    // ... split train/test ...
    const model = new XGBoost({ ...params, objective: 'reg:squarederror' });
    model.fit(X_train, y_train);
    return model.predict(X_test);
}
```

## License

MIT
