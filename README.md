# XGBoost4JS

XGBoost4JS is an implementation of the XGBoost (Extreme Gradient Boosting) algorithm in TypeScript/Angular with automatic timestamp-based feature engineering.

## Key Features

- 🚀 **XGBoost Algorithm**: Full gradient boosting decision tree implementation
- ⏰ **Automatic Timestamp Features**: Automatically extracts 8+ time-based features from timestamps
- 🎯 **Custom Features**: Support for user-defined features alongside automatic features
- 📊 **Interactive Demo**: Energy consumption prediction demo included
- ✅ **Type-Safe**: Written in TypeScript with comprehensive interfaces
- 🧪 **Well-Tested**: 20+ unit tests covering all functionality

## Quick Start

```typescript
import { XGBoostService } from './services/xgboost.service';

// Prepare training data
const trainingData = [
  {
    timestamp: new Date(2024, 0, 15, 8, 0, 0),
    features: { temperature: 20 },
    target: 75
  },
  // ... more data
];

// Train and predict
xgboostService.train(trainingData, { numTrees: 20 });
const prediction = xgboostService.predict({
  timestamp: new Date(2024, 0, 15, 10, 0, 0),
  features: { temperature: 22 }
});
```

See [USAGE.md](./USAGE.md) for detailed documentation and examples.

## Automatic Timestamp Features

The algorithm automatically extracts these features from any timestamp:
- Hour (0-23)
- Day of week (0-6)
- Day of month (1-31)
- Month (0-11)
- Weekend indicator (0/1)
- Daytime indicator (0/1)
- Quarter of day (0-3)
- Week of year (1-53)

## Project Structure

This project was generated using [Angular CLI](https://github.com/angular/angular-cli) version 20.3.8.

## Development server

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
