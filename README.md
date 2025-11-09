# XGBoost4JS

An Angular-based implementation of XGBoost (Extreme Gradient Boosting) for JavaScript/TypeScript, with built-in support for time series forecasting and automatic timestamp feature extraction.

## Features

- **XGBoost Implementation**: Binary classification and regression support
- **Time Series Analysis**: Built-in tools for time series forecasting
- **Automatic Timestamp Features**: Automatically extracts 13 temporal features from timestamps including:
  - Hour, day of week, day of month, month, quarter
  - Day/night and weekend/weekday indicators
  - Cyclical encodings (sin/cos) for continuous temporal patterns
- **Custom Features**: Combine automatic timestamp features with your own custom features
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Interactive Visualization**: Built-in charts for data exploration and results

## Quick Start with Timestamp Features

```typescript
import { XGBoost, TimestampTrainingData } from './xgboost';

// Prepare training data
const trainData: TimestampTrainingData = {
  timestamps: [
    new Date('2024-01-01T08:00:00Z'),
    new Date('2024-01-01T20:00:00Z'),
    // ... more timestamps
  ],
  customFeatures: [
    [25.5, 60],  // Optional: temperature, humidity, etc.
    [18.0, 65],
    // ...
  ],
  y: [0, 1] // Target values
};

// Train model
const model = new XGBoost({
  task: 'binary',
  numRounds: 100,
  maxDepth: 4,
  learningRate: 0.1
});

model.fitWithTimestamps(trainData);

// Make predictions
const prediction = model.predictWithTimestamp(
  new Date('2024-01-02T21:00:00Z'),
  [20.0, 70] // Optional custom features
);
```

For detailed examples and usage, see [TIMESTAMP_FEATURES.md](TIMESTAMP_FEATURES.md).

## Development server

This project was generated using [Angular CLI](https://github.com/angular/angular-cli) version 20.3.8.

To start a local development server, run:

```bash
ng serve
```

Once the server is running, open your browser and navigate to `http://localhost:4200/`. The application will automatically reload whenever you modify any of the source files.

## Code scaffolding

Angular CLI includes powerful code scaffolding tools. To generate a new component, run:

```bash
ng generate component component-name
```

For a complete list of available schematics (such as `components`, `directives`, or `pipes`), run:

```bash
ng generate --help
```

## Building

To build the project run:

```bash
ng build
```

This will compile your project and store the build artifacts in the `dist/` directory. By default, the production build optimizes your application for performance and speed.

## Running unit tests

To execute unit tests with the [Karma](https://karma-runner.github.io) test runner, use the following command:

```bash
ng test
```

## Running end-to-end tests

For end-to-end (e2e) testing, run:

```bash
ng e2e
```

Angular CLI does not come with an end-to-end testing framework by default. You can choose one that suits your needs.

## Additional Resources

For more information on using the Angular CLI, including detailed command references, visit the [Angular CLI Overview and Command Reference](https://angular.dev/tools/cli) page.
