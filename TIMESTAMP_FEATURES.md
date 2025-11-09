# XGBoost4JS - Timestamp Features Usage Examples

This document demonstrates how to use the new timestamp feature extraction capabilities in XGBoost4JS.

## Overview

The XGBoost algorithm has been enhanced to automatically extract temporal features from timestamps, making it easier to work with time-based data. The system automatically generates 13 temporal features from any timestamp:

### Automatically Extracted Features:
1. **hour** (0-23): Hour of the day
2. **dayOfWeek** (0-6): Day of the week (0=Sunday, 6=Saturday)
3. **dayOfMonth** (1-31): Day of the month
4. **month** (1-12): Month of the year
5. **quarter** (1-4): Quarter of the year
6. **isNight** (0/1): Binary indicator for night time (18:00-06:00)
7. **isWeekend** (0/1): Binary indicator for weekend (Saturday/Sunday)
8. **hourSin**: Cyclical encoding of hour (preserves continuity)
9. **hourCos**: Cyclical encoding of hour
10. **dayOfWeekSin**: Cyclical encoding of day of week
11. **dayOfWeekCos**: Cyclical encoding of day of week
12. **monthSin**: Cyclical encoding of month
13. **monthCos**: Cyclical encoding of month

## Basic Usage

### 1. Extract Features from a Timestamp

```typescript
import { extractTimestampFeatures } from './xgboost';

// Using Unix timestamp (milliseconds)
const features1 = extractTimestampFeatures(1710512400000);

// Using Date object
const features2 = extractTimestampFeatures(new Date());

// Using ISO string
const features3 = extractTimestampFeatures('2024-03-15T14:30:00Z');

console.log(features1);
// Output:
// {
//   hour: 14,
//   dayOfWeek: 5,
//   dayOfMonth: 15,
//   month: 3,
//   quarter: 1,
//   isNight: 0,
//   isWeekend: 0,
//   hourSin: 0.5,
//   hourCos: -0.866,
//   ...
// }
```

### 2. Training with Timestamps Only

```typescript
import { XGBoost, TimestampTrainingData } from './xgboost';

// Prepare training data
const trainData: TimestampTrainingData = {
  timestamps: [
    new Date('2024-01-01T08:00:00Z'),
    new Date('2024-01-01T20:00:00Z'),
    new Date('2024-01-02T09:00:00Z'),
    new Date('2024-01-02T22:00:00Z'),
    // ... more timestamps
  ],
  y: [0, 1, 0, 1] // Target values
};

// Create and train model
const model = new XGBoost({
  task: 'binary',
  numRounds: 100,
  maxDepth: 4,
  learningRate: 0.1
});

model.fitWithTimestamps(trainData);

// Make predictions
const prediction = model.predictWithTimestamp(new Date('2024-01-03T21:00:00Z'));
console.log('Prediction:', prediction);
```

### 3. Training with Timestamps + Custom Features

You can combine automatic timestamp features with your own custom features:

```typescript
import { XGBoost, TimestampTrainingData } from './xgboost';

// Prepare training data with custom features
const trainData: TimestampTrainingData = {
  timestamps: [
    new Date('2024-06-15T14:00:00Z'),
    new Date('2024-06-16T14:00:00Z'),
    new Date('2024-12-15T14:00:00Z'),
    new Date('2024-12-16T14:00:00Z'),
  ],
  customFeatures: [
    [35.5, 65],  // temperature, humidity
    [32.0, 70],
    [10.5, 80],
    [12.0, 75],
  ],
  y: [1, 1, 0, 0] // e.g., 1=hot, 0=cold
};

const model = new XGBoost({
  task: 'binary',
  numRounds: 50,
  maxDepth: 4,
  learningRate: 0.1
});

model.fitWithTimestamps(trainData);

// Predict with timestamp and custom features
const pred = model.predictWithTimestamp(
  new Date('2024-07-01T14:00:00Z'),
  [34.0, 68] // temperature, humidity
);

console.log('Prediction:', pred);
```

### 4. Using with XGBoostService

```typescript
import { XGBoostService } from './xgboost.service';

// In your Angular component
constructor(private xgboostService: XGBoostService) {}

trainModel() {
  const trainData: TimestampTrainingData = {
    timestamps: [ /* ... */ ],
    customFeatures: [ /* ... */ ], // optional
    y: [ /* ... */ ]
  };
  
  const testData: TimestampTrainingData = {
    timestamps: [ /* ... */ ],
    customFeatures: [ /* ... */ ], // optional
    y: [ /* ... */ ]
  };
  
  const result = this.xgboostService.trainAndPredictWithTimestamps(
    {
      task: 'binary',
      numRounds: 100,
      maxDepth: 4,
      learningRate: 0.1
    },
    trainData,
    testData
  );
  
  console.log('Predictions:', result.predictions);
  console.log('Feature Importance:', result.importance);
  if (result.metrics) {
    console.log('Accuracy:', result.metrics.accuracy);
  }
}
```

### 5. Batch Predictions

```typescript
import { XGBoost } from './xgboost';

// After training...
const timestamps = [
  new Date('2024-01-10T09:00:00Z'),
  new Date('2024-01-10T15:00:00Z'),
  new Date('2024-01-10T21:00:00Z'),
];

// Without custom features
const predictions1 = model.predictBatchWithTimestamps(timestamps);

// With custom features
const customFeatures = [
  [22.5, 55],
  [25.0, 50],
  [18.0, 65],
];
const predictions2 = model.predictBatchWithTimestamps(timestamps, customFeatures);

console.log('Predictions:', predictions2);
```

## Example Use Cases

### Energy Consumption Prediction

```typescript
// Predict energy usage based on time of day, day of week, and temperature
const energyData: TimestampTrainingData = {
  timestamps: historicalTimestamps,
  customFeatures: historicalTemperatures.map(t => [t]),
  y: historicalEnergyConsumption
};

const model = new XGBoost({
  task: 'regression',
  numRounds: 200,
  maxDepth: 6,
  learningRate: 0.05
});

model.fitWithTimestamps(energyData);

// Predict for next week
const nextWeekPrediction = model.predictWithTimestamp(
  new Date('2024-06-24T14:00:00Z'),
  [28.5] // forecasted temperature
);
```

### Traffic Prediction

```typescript
// Predict traffic congestion based on time patterns
const trafficData: TimestampTrainingData = {
  timestamps: observationTimestamps,
  customFeatures: weatherConditions, // e.g., [rainfall, visibility]
  y: trafficLevels // 0=low, 1=high
};

const model = new XGBoost({
  task: 'binary',
  numRounds: 150,
  maxDepth: 5,
  learningRate: 0.1
});

model.fitWithTimestamps(trafficData);

// Predict traffic for rush hour
const rushHourPrediction = model.predictWithTimestamp(
  new Date('2024-06-24T08:30:00Z'),
  [0, 10] // no rain, 10km visibility
);
```

### Sales Forecasting

```typescript
// Predict sales based on temporal patterns and promotions
const salesData: TimestampTrainingData = {
  timestamps: salesTimestamps,
  customFeatures: promotions.map(p => [
    p.discountPercent,
    p.isHolidaySeason ? 1 : 0
  ]),
  y: salesVolumes
};

const model = new XGBoost({
  task: 'regression',
  numRounds: 100,
  maxDepth: 5,
  learningRate: 0.1
});

model.fitWithTimestamps(salesData);
```

## Feature Importance

After training, you can analyze which features are most important:

```typescript
const importance = model.getFeatureImportance();

// The first 13 values correspond to timestamp features (in order):
// [hour, dayOfWeek, dayOfMonth, month, quarter, isNight, isWeekend, 
//  hourSin, hourCos, dayOfWeekSin, dayOfWeekCos, monthSin, monthCos]

// Remaining values (if any) correspond to your custom features
console.log('Hour importance:', importance[0]);
console.log('Is weekend importance:', importance[6]);
console.log('Custom feature 1 importance:', importance[13]);
```

## Advanced: Manual Feature Preparation

If you need more control, you can manually prepare features:

```typescript
import { 
  extractTimestampFeatures, 
  timestampFeaturesToArray,
  prepareTimestampFeatures 
} from './xgboost';

// Extract and convert to array manually
const timestamp = new Date('2024-03-15T14:30:00Z');
const features = extractTimestampFeatures(timestamp);
const featureArray = timestampFeaturesToArray(features);

// Add custom features
const customFeatures = [25.0, 60]; // temperature, humidity
const fullFeatureArray = [...featureArray, ...customFeatures];

// Use with standard fit/predict methods
const model = new XGBoost({ task: 'binary', numRounds: 100 });
model.fit(X_prepared, y);
const prediction = model.predictSingle(fullFeatureArray);
```

## Tips

1. **Cyclical Encodings**: The sin/cos encodings ensure that values at boundaries (e.g., 23:00 and 0:00) are treated as close together, which is important for temporal patterns.

2. **Feature Scaling**: The timestamp features have different scales (e.g., hour: 0-23, month: 1-12). XGBoost handles this well through its tree-based approach.

3. **Custom Features**: Always maintain the same number and order of custom features between training and prediction.

4. **Task Selection**: Use `task: 'binary'` for classification problems and `task: 'regression'` for continuous value prediction.

5. **Hyperparameter Tuning**: Adjust `numRounds`, `maxDepth`, and `learningRate` based on your data size and complexity.
