/**
 * Simple example demonstrating timestamp feature extraction and prediction
 * This file can be used to quickly test the new timestamp features
 */

import { XGBoost, TimestampTrainingData, extractTimestampFeatures } from './src/app/xgboost';

console.log('=== XGBoost Timestamp Features Demo ===\n');

// 1. Feature Extraction Example
console.log('1. Extracting features from a timestamp:');
const exampleDate = new Date('2024-06-15T14:30:00Z');
const features = extractTimestampFeatures(exampleDate);
console.log('Timestamp:', exampleDate.toISOString());
console.log('Extracted features:', features);
console.log('');

// 2. Training Example: Predict business hours
console.log('2. Training model to predict business hours (9-17, Mon-Fri):');
const timestamps: Date[] = [];
const y: number[] = [];

// Generate 2 weeks of hourly data
for (let day = 0; day < 14; day++) {
  for (let hour = 0; hour < 24; hour++) {
    const date = new Date(Date.UTC(2024, 0, 1 + day, hour, 0, 0));
    timestamps.push(date);
    const dayOfWeek = date.getDay();
    // Business hours: weekday (1-5) and 9-17
    y.push((dayOfWeek >= 1 && dayOfWeek <= 5 && hour >= 9 && hour < 17) ? 1 : 0);
  }
}

const trainData: TimestampTrainingData = { timestamps, y };

const model = new XGBoost({
  task: 'binary',
  numRounds: 100,
  maxDepth: 5,
  learningRate: 0.1
});

model.fitWithTimestamps(trainData);
console.log('Model trained with', timestamps.length, 'samples');
console.log('');

// 3. Prediction Examples
console.log('3. Making predictions:');

// Monday 10 AM (should be business hours)
const testDate1 = new Date(Date.UTC(2024, 0, 15, 10, 0, 0));
const pred1 = model.predictWithTimestamp(testDate1);
console.log('Monday 10 AM:', testDate1.toISOString());
console.log('Prediction (business hours probability):', pred1.toFixed(4));
console.log('Expected: High probability (> 0.5)');
console.log('');

// Saturday 10 AM (should NOT be business hours)
const testDate2 = new Date(Date.UTC(2024, 0, 13, 10, 0, 0));
const pred2 = model.predictWithTimestamp(testDate2);
console.log('Saturday 10 AM:', testDate2.toISOString());
console.log('Prediction (business hours probability):', pred2.toFixed(4));
console.log('Expected: Low probability (< 0.5)');
console.log('');

// 4. Custom Features Example
console.log('4. Training with custom features (temperature):');
const timestamps2: Date[] = [];
const customFeatures: number[][] = [];
const y2: number[] = [];

// Summer (hot) samples
for (let i = 0; i < 50; i++) {
  timestamps2.push(new Date(Date.UTC(2024, 5 + (i % 3), 1 + i, 14, 0, 0)));
  customFeatures.push([30 + Math.random() * 10]); // 30-40°C
  y2.push(1); // Hot
}

// Winter (cold) samples
for (let i = 0; i < 50; i++) {
  timestamps2.push(new Date(Date.UTC(2024, 11, 1 + i, 14, 0, 0)));
  customFeatures.push([5 + Math.random() * 10]); // 5-15°C
  y2.push(0); // Cold
}

const trainData2: TimestampTrainingData = {
  timestamps: timestamps2,
  customFeatures: customFeatures,
  y: y2
};

const model2 = new XGBoost({
  task: 'binary',
  numRounds: 50,
  maxDepth: 4,
  learningRate: 0.1
});

model2.fitWithTimestamps(trainData2);
console.log('Model trained with timestamp + temperature features');
console.log('');

// Predict hot summer day
const predHot = model2.predictWithTimestamp(
  new Date(Date.UTC(2024, 6, 15, 14, 0, 0)),
  [35] // 35°C
);
console.log('July 15, 35°C - Prediction (hot probability):', predHot.toFixed(4));

// Predict cold winter day
const predCold = model2.predictWithTimestamp(
  new Date(Date.UTC(2024, 0, 15, 14, 0, 0)),
  [8] // 8°C
);
console.log('January 15, 8°C - Prediction (hot probability):', predCold.toFixed(4));
console.log('');

console.log('=== Demo Complete ===');
console.log('For more examples, see TIMESTAMP_FEATURES.md');
