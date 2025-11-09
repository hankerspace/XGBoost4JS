import { TestBed } from '@angular/core/testing';
import { XGBoostService } from './xgboost.service';
import { TimestampFeatureService } from './timestamp-feature.service';
import { TrainingDataPoint } from '../models/xgboost.interface';

describe('XGBoostService', () => {
  let service: XGBoostService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [XGBoostService, TimestampFeatureService]
    });
    service = TestBed.inject(XGBoostService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  describe('train and predict', () => {
    it('should train a model and make predictions', () => {
      // Create simple training data with a pattern:
      // Higher hour -> higher target value
      const trainingData: TrainingDataPoint[] = [
        { timestamp: new Date(2024, 0, 15, 8, 0, 0), target: 10 },
        { timestamp: new Date(2024, 0, 15, 12, 0, 0), target: 20 },
        { timestamp: new Date(2024, 0, 15, 16, 0, 0), target: 30 },
        { timestamp: new Date(2024, 0, 15, 20, 0, 0), target: 40 },
      ];

      service.train(trainingData, { numTrees: 5, maxDepth: 3, learningRate: 0.1 });

      // Predict for a time in the middle
      const prediction = service.predict({
        timestamp: new Date(2024, 0, 15, 14, 0, 0)
      });

      // Prediction should be reasonable (between 20 and 30)
      expect(prediction).toBeGreaterThan(15);
      expect(prediction).toBeLessThan(35);
    });

    it('should handle user-provided features', () => {
      const trainingData: TrainingDataPoint[] = [
        { timestamp: new Date(2024, 0, 15, 10, 0, 0), features: { temperature: 20 }, target: 10 },
        { timestamp: new Date(2024, 0, 15, 10, 0, 0), features: { temperature: 30 }, target: 20 },
        { timestamp: new Date(2024, 0, 15, 10, 0, 0), features: { temperature: 40 }, target: 30 },
      ];

      service.train(trainingData, { numTrees: 5, maxDepth: 3 });

      const prediction = service.predict({
        timestamp: new Date(2024, 0, 15, 10, 0, 0),
        features: { temperature: 25 }
      });

      expect(prediction).toBeGreaterThan(10);
      expect(prediction).toBeLessThan(25);
    });

    it('should throw error when predicting without training', () => {
      expect(() => {
        service.predict({ timestamp: new Date() });
      }).toThrowError(/Model not trained/);
    });

    it('should throw error with empty training data', () => {
      expect(() => {
        service.train([]);
      }).toThrowError(/Training data is empty/);
    });
  });

  describe('getModelSummary', () => {
    it('should return model information after training', () => {
      const trainingData: TrainingDataPoint[] = [
        { timestamp: new Date(2024, 0, 15, 8, 0, 0), target: 10 },
        { timestamp: new Date(2024, 0, 15, 12, 0, 0), target: 20 },
      ];

      service.train(trainingData, { numTrees: 3 });
      const summary = service.getModelSummary();

      expect(summary.numTrees).toBe(3);
      expect(summary.featureNames.length).toBeGreaterThan(0);
      expect(summary.config).toBeDefined();
      expect(summary.config.numTrees).toBe(3);
    });
  });

  describe('weekend/weekday pattern', () => {
    it('should learn weekend vs weekday patterns', () => {
      // Weekend targets are higher
      const trainingData: TrainingDataPoint[] = [
        // Weekdays
        { timestamp: new Date(2024, 0, 15, 10, 0, 0), target: 10 }, // Monday
        { timestamp: new Date(2024, 0, 16, 10, 0, 0), target: 10 }, // Tuesday
        { timestamp: new Date(2024, 0, 17, 10, 0, 0), target: 10 }, // Wednesday
        // Weekend
        { timestamp: new Date(2024, 0, 13, 10, 0, 0), target: 50 }, // Saturday
        { timestamp: new Date(2024, 0, 14, 10, 0, 0), target: 50 }, // Sunday
        { timestamp: new Date(2024, 0, 20, 10, 0, 0), target: 50 }, // Saturday
      ];

      service.train(trainingData, { numTrees: 10, maxDepth: 3, learningRate: 0.3 });

      // Predict for weekday and weekend
      const weekdayPrediction = service.predict({
        timestamp: new Date(2024, 0, 18, 10, 0, 0) // Thursday
      });

      const weekendPrediction = service.predict({
        timestamp: new Date(2024, 0, 21, 10, 0, 0) // Sunday
      });

      // Weekend prediction should be notably higher than weekday
      expect(weekendPrediction).toBeGreaterThan(weekdayPrediction);
    });
  });
});
