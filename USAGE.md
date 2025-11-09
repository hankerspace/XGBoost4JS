# XGBoost4JS - Usage Guide

## Overview

XGBoost4JS is an Angular application that implements the XGBoost (Extreme Gradient Boosting) algorithm with automatic timestamp-based feature engineering. This implementation allows you to train models that automatically extract time-based features from timestamps while also supporting custom user-defined features.

## Features

### 1. Automatic Timestamp Feature Engineering

The system automatically extracts the following features from any timestamp:

- **hour** (0-23): Hour of the day
- **dayOfWeek** (0-6): Day of the week (0 = Sunday, 6 = Saturday)
- **dayOfMonth** (1-31): Day of the month
- **month** (0-11): Month of the year
- **isWeekend** (0 or 1): Whether it's a weekend
- **isDayTime** (0 or 1): Whether it's daytime (6:00 AM - 6:00 PM)
- **quarterOfDay** (0-3): Quarter of the day (0: 0-6h, 1: 6-12h, 2: 12-18h, 3: 18-24h)
- **weekOfYear** (1-53): ISO week number of the year

### 2. Custom Feature Support

You can add any number of custom features (e.g., temperature, humidity, price) that will be automatically combined with timestamp features.

### 3. XGBoost Algorithm

A full implementation of the gradient boosting decision tree algorithm with configurable hyperparameters:

- `maxDepth`: Maximum depth of each tree (default: 3)
- `learningRate`: Learning rate/eta (default: 0.1)
- `numTrees`: Number of boosting rounds (default: 10)
- `minChildWeight`: Minimum sum of instance weight needed in a child (default: 1)
- `subsample`: Subsample ratio of training instances (default: 1.0)
- `colsampleByTree`: Subsample ratio of columns when constructing each tree (default: 1.0)

## API Usage

### Basic Example

```typescript
import { XGBoostService } from './services/xgboost.service';
import { TrainingDataPoint } from './models/xgboost.interface';

// Inject the service
constructor(private xgboostService: XGBoostService) {}

// Prepare training data
const trainingData: TrainingDataPoint[] = [
  {
    timestamp: new Date(2024, 0, 15, 8, 0, 0),
    features: { temperature: 20 },
    target: 75
  },
  {
    timestamp: new Date(2024, 0, 15, 14, 0, 0),
    features: { temperature: 25 },
    target: 85
  },
  // ... more data points
];

// Train the model
this.xgboostService.train(trainingData, {
  numTrees: 20,
  maxDepth: 4,
  learningRate: 0.1
});

// Make predictions
const prediction = this.xgboostService.predict({
  timestamp: new Date(2024, 0, 15, 10, 0, 0),
  features: { temperature: 22 }
});

console.log('Predicted value:', prediction);
```

### Advanced Configuration

```typescript
// Train with custom hyperparameters
this.xgboostService.train(trainingData, {
  maxDepth: 5,              // Deeper trees
  learningRate: 0.05,       // Slower learning
  numTrees: 50,             // More boosting rounds
  minChildWeight: 5,        // Require more instances per leaf
  subsample: 0.8,           // Use 80% of data per tree
  colsampleByTree: 0.8      // Use 80% of features per tree
});

// Get model information
const summary = this.xgboostService.getModelSummary();
console.log('Number of trees:', summary.numTrees);
console.log('Features used:', summary.featureNames);
```

### Using Only Timestamp Features

```typescript
// You don't need to provide custom features
const trainingData: TrainingDataPoint[] = [
  {
    timestamp: new Date(2024, 0, 15, 8, 0, 0),
    target: 75
  },
  {
    timestamp: new Date(2024, 0, 15, 14, 0, 0),
    target: 85
  }
];

this.xgboostService.train(trainingData);

// Predict with only timestamp
const prediction = this.xgboostService.predict({
  timestamp: new Date(2024, 0, 15, 10, 0, 0)
});
```

### Extracting Timestamp Features Manually

```typescript
import { TimestampFeatureService } from './services/timestamp-feature.service';

constructor(private timestampFeatureService: TimestampFeatureService) {}

// Extract features from a timestamp
const features = this.timestampFeatureService.extractTimestampFeatures(
  new Date(2024, 0, 15, 10, 30, 0)
);

console.log(features);
// {
//   hour: 10,
//   dayOfWeek: 1,
//   dayOfMonth: 15,
//   month: 0,
//   isWeekend: 0,
//   isDayTime: 1,
//   quarterOfDay: 1,
//   weekOfYear: 3
// }
```

## Use Cases

### Energy Consumption Prediction

Predict energy consumption based on time of day, day of week, and temperature:

```typescript
const energyData: TrainingDataPoint[] = [
  // Weekday peak hours
  { timestamp: new Date(2024, 0, 15, 9, 0), features: { temp: 15 }, target: 80 },
  { timestamp: new Date(2024, 0, 16, 18, 0), features: { temp: 20 }, target: 90 },
  
  // Weekend off-peak
  { timestamp: new Date(2024, 0, 13, 10, 0), features: { temp: 18 }, target: 40 },
  { timestamp: new Date(2024, 0, 14, 15, 0), features: { temp: 22 }, target: 45 }
];

xgboostService.train(energyData);
```

### Sales Forecasting

Predict sales based on time patterns and external factors:

```typescript
const salesData: TrainingDataPoint[] = [
  {
    timestamp: new Date(2024, 0, 15, 12, 0),
    features: { 
      promotion: 1,
      competitors: 3,
      weather: 1  // sunny
    },
    target: 1500  // sales amount
  }
];
```

### Traffic Prediction

Predict traffic volume based on time and conditions:

```typescript
const trafficData: TrainingDataPoint[] = [
  {
    timestamp: new Date(2024, 0, 15, 8, 0),
    features: { 
      rainfall: 0,
      event: 0
    },
    target: 5000  // number of vehicles
  }
];
```

## Running the Demo

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Open http://localhost:4200 in your browser

4. The demo shows an interactive energy consumption prediction model where you can:
   - Select different dates and times
   - Adjust temperature values
   - See predictions based on learned patterns

## Testing

Run the test suite:

```bash
npm test
```

The test suite includes:
- Timestamp feature extraction tests
- XGBoost training and prediction tests
- Integration tests with combined features
- Edge case handling

## Building for Production

```bash
npm run build
```

The build artifacts will be stored in the `dist/` directory.

## Architecture

```
src/app/
├── models/
│   └── xgboost.interface.ts      # TypeScript interfaces and types
├── services/
│   ├── timestamp-feature.service.ts   # Timestamp feature extraction
│   ├── xgboost.service.ts            # XGBoost algorithm implementation
│   └── *.spec.ts                     # Unit tests
└── app.ts                             # Main application component with demo
```

## Performance Considerations

- Training complexity: O(n * d * t) where n = samples, d = depth, t = trees
- Prediction complexity: O(d * t)
- For large datasets (>10,000 samples), consider using subsample < 1.0
- For many features (>50), consider using colsampleByTree < 1.0

## Limitations

- This is a simplified implementation for educational/demonstration purposes
- For production use cases with very large datasets, consider using the native XGBoost library
- No GPU acceleration support
- Limited to regression tasks (continuous target values)

## License

MIT
