import { 
  extractTimestampFeatures, 
  timestampFeaturesToArray, 
  prepareTimestampFeatures,
  TimestampTrainingData,
  XGBoost
} from './xgboost';

describe('Timestamp Features', () => {
  describe('extractTimestampFeatures', () => {
    it('should extract features from a Unix timestamp', () => {
      // 2024-03-15 14:30:00 UTC (Friday)
      const timestamp = 1710512400000;
      const features = extractTimestampFeatures(timestamp);
      
      expect(features.hour).toBe(14);
      expect(features.dayOfWeek).toBe(5); // Friday
      expect(features.dayOfMonth).toBe(15);
      expect(features.month).toBe(3); // March
      expect(features.quarter).toBe(1); // Q1
      expect(features.isNight).toBe(0); // Day time
      expect(features.isWeekend).toBe(0); // Weekday
    });

    it('should extract features from a Date object', () => {
      // 2024-06-22 20:00:00 UTC (Saturday night)
      const date = new Date(Date.UTC(2024, 5, 22, 20, 0, 0));
      const features = extractTimestampFeatures(date);
      
      expect(features.hour).toBe(20);
      expect(features.dayOfWeek).toBe(6); // Saturday
      expect(features.dayOfMonth).toBe(22);
      expect(features.month).toBe(6); // June
      expect(features.quarter).toBe(2); // Q2
      expect(features.isNight).toBe(1); // Night time (>= 18:00)
      expect(features.isWeekend).toBe(1); // Weekend
    });

    it('should extract features from an ISO string', () => {
      // 2024-12-01 05:00:00 UTC (Sunday early morning)
      const isoString = '2024-12-01T05:00:00Z';
      const features = extractTimestampFeatures(isoString);
      
      expect(features.hour).toBe(5);
      expect(features.dayOfWeek).toBe(0); // Sunday
      expect(features.month).toBe(12); // December
      expect(features.quarter).toBe(4); // Q4
      expect(features.isNight).toBe(1); // Night time (< 6:00)
      expect(features.isWeekend).toBe(1); // Weekend
    });

    it('should provide cyclical encodings for hour', () => {
      const midnight = new Date(Date.UTC(2024, 0, 1, 0, 0, 0));
      const noon = new Date(Date.UTC(2024, 0, 1, 12, 0, 0));
      
      const features0 = extractTimestampFeatures(midnight);
      const features12 = extractTimestampFeatures(noon);
      
      // At midnight, hour angle is 0, so sin=0, cos=1
      expect(features0.hourSin).toBeCloseTo(0, 5);
      expect(features0.hourCos).toBeCloseTo(1, 5);
      
      // At noon, hour angle is π, so sin≈0, cos≈-1
      expect(features12.hourSin).toBeCloseTo(0, 5);
      expect(features12.hourCos).toBeCloseTo(-1, 5);
    });

    it('should provide cyclical encodings for day of week', () => {
      // Sunday (0)
      const sunday = new Date(Date.UTC(2024, 0, 7, 12, 0, 0));
      const features = extractTimestampFeatures(sunday);
      
      // At day 0, angle is 0, so sin=0, cos=1
      expect(features.dayOfWeekSin).toBeCloseTo(0, 5);
      expect(features.dayOfWeekCos).toBeCloseTo(1, 5);
    });

    it('should provide cyclical encodings for month', () => {
      // January (month 1)
      const jan = new Date(Date.UTC(2024, 0, 15, 12, 0, 0));
      const features = extractTimestampFeatures(jan);
      
      // Verify cyclical encoding exists
      expect(typeof features.monthSin).toBe('number');
      expect(typeof features.monthCos).toBe('number');
      expect(features.monthSin).toBeGreaterThan(-1);
      expect(features.monthSin).toBeLessThan(1);
    });
  });

  describe('timestampFeaturesToArray', () => {
    it('should convert features to array in consistent order', () => {
      const features = extractTimestampFeatures(new Date(Date.UTC(2024, 5, 15, 14, 30, 0)));
      const array = timestampFeaturesToArray(features);
      
      expect(Array.isArray(array)).toBe(true);
      expect(array.length).toBe(13);
      
      // Check the order matches expected
      expect(array[0]).toBe(features.hour);
      expect(array[1]).toBe(features.dayOfWeek);
      expect(array[2]).toBe(features.dayOfMonth);
      expect(array[3]).toBe(features.month);
      expect(array[4]).toBe(features.quarter);
      expect(array[5]).toBe(features.isNight);
      expect(array[6]).toBe(features.isWeekend);
      expect(array[7]).toBe(features.hourSin);
      expect(array[8]).toBe(features.hourCos);
      expect(array[9]).toBe(features.dayOfWeekSin);
      expect(array[10]).toBe(features.dayOfWeekCos);
      expect(array[11]).toBe(features.monthSin);
      expect(array[12]).toBe(features.monthCos);
    });
  });

  describe('prepareTimestampFeatures', () => {
    it('should prepare features from timestamps only', () => {
      const data: TimestampTrainingData = {
        timestamps: [
          new Date(Date.UTC(2024, 0, 1, 10, 0, 0)),
          new Date(Date.UTC(2024, 0, 2, 14, 0, 0)),
          new Date(Date.UTC(2024, 0, 3, 18, 0, 0)),
        ],
        y: [1, 0, 1]
      };
      
      const X = prepareTimestampFeatures(data);
      
      expect(X.length).toBe(3);
      expect(X[0].length).toBe(13); // 13 timestamp features
      expect(X[1].length).toBe(13);
      expect(X[2].length).toBe(13);
    });

    it('should combine timestamp features with custom features', () => {
      const data: TimestampTrainingData = {
        timestamps: [
          new Date(Date.UTC(2024, 0, 1, 10, 0, 0)),
          new Date(Date.UTC(2024, 0, 2, 14, 0, 0)),
        ],
        customFeatures: [
          [25.5, 60],  // e.g., temperature and humidity
          [28.0, 55],
        ],
        y: [1, 0]
      };
      
      const X = prepareTimestampFeatures(data);
      
      expect(X.length).toBe(2);
      expect(X[0].length).toBe(15); // 13 timestamp + 2 custom features
      expect(X[1].length).toBe(15);
      
      // Check that custom features are appended
      expect(X[0][13]).toBe(25.5);
      expect(X[0][14]).toBe(60);
      expect(X[1][13]).toBe(28.0);
      expect(X[1][14]).toBe(55);
    });

    it('should handle mixed timestamp types', () => {
      const data: TimestampTrainingData = {
        timestamps: [
          1710512400000, // Unix timestamp
          new Date(Date.UTC(2024, 3, 15, 14, 0, 0)), // Date object
          '2024-05-15T14:00:00Z', // ISO string
        ],
        y: [1, 0, 1]
      };
      
      const X = prepareTimestampFeatures(data);
      
      expect(X.length).toBe(3);
      expect(X[0].length).toBe(13);
      expect(X[1].length).toBe(13);
      expect(X[2].length).toBe(13);
    });
  });

  describe('XGBoost with timestamp features', () => {
    it('should train and predict using fitWithTimestamps', () => {
      // Create simple training data: predict 1 for night hours, 0 for day hours
      const timestamps: Date[] = [];
      const y: number[] = [];
      
      // Generate multiple weeks of data for better training
      for (let week = 0; week < 4; week++) {
        for (let day = 0; day < 7; day++) {
          for (let hour = 0; hour < 24; hour++) {
            const date = new Date(Date.UTC(2024, 0, 1 + week * 7 + day, hour, 0, 0));
            timestamps.push(date);
            // Night hours: 18-6
            y.push((hour >= 18 || hour < 6) ? 1 : 0);
          }
        }
      }
      
      const data: TimestampTrainingData = { timestamps, y };
      
      const model = new XGBoost({ 
        task: 'binary', 
        numRounds: 100, 
        maxDepth: 5,
        learningRate: 0.1 
      });
      model.fitWithTimestamps(data);
      
      // Test prediction for night time (should predict 1)
      const testDate1 = new Date(Date.UTC(2024, 1, 1, 20, 0, 0)); // 8 PM
      const pred1 = model.predictWithTimestamp(testDate1);
      expect(pred1).toBeGreaterThan(0.5); // Should predict night
      
      // Test prediction for day time (should predict 0)
      const testDate2 = new Date(Date.UTC(2024, 1, 1, 12, 0, 0)); // Noon
      const pred2 = model.predictWithTimestamp(testDate2);
      expect(pred2).toBeLessThan(0.5); // Should predict day
    });

    it('should predict with custom features', () => {
      // Simple dataset: predict based on temperature with more training data
      const timestamps: Date[] = [];
      const customFeatures: number[][] = [];
      const y: number[] = [];
      
      // Generate more training samples
      for (let i = 0; i < 50; i++) {
        // Summer dates with high temperatures
        timestamps.push(new Date(Date.UTC(2024, 5 + (i % 3), 1 + i, 14, 0, 0)));
        customFeatures.push([30 + Math.random() * 10]); // 30-40 degrees
        y.push(1); // Hot
        
        // Winter dates with low temperatures
        timestamps.push(new Date(Date.UTC(2024, 11 + (i % 2), 1 + i, 14, 0, 0)));
        customFeatures.push([5 + Math.random() * 10]); // 5-15 degrees
        y.push(0); // Cold
      }
      
      const data: TimestampTrainingData = {
        timestamps,
        customFeatures,
        y
      };
      
      const model = new XGBoost({ 
        task: 'binary', 
        numRounds: 50,
        maxDepth: 4,
        learningRate: 0.1
      });
      model.fitWithTimestamps(data);
      
      // Predict with timestamp and temperature
      const pred1 = model.predictWithTimestamp(
        new Date(Date.UTC(2024, 6, 15, 14, 0, 0)), 
        [35]
      );
      const pred2 = model.predictWithTimestamp(
        new Date(Date.UTC(2024, 0, 15, 14, 0, 0)), 
        [8]
      );
      
      expect(pred1).toBeGreaterThan(pred2); // Hot day should have higher probability
    });

    it('should predict batch with timestamps', () => {
      // Generate more training data for better model learning
      const timestamps: Date[] = [];
      const y: number[] = [];
      
      // Generate several days of hourly data
      for (let day = 0; day < 14; day++) {
        for (let hour = 0; hour < 24; hour++) {
          const date = new Date(Date.UTC(2024, 0, 1 + day, hour, 0, 0));
          timestamps.push(date);
          // 1 for night hours (18-6), 0 for day
          y.push((hour >= 18 || hour < 6) ? 1 : 0);
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
      
      const testTimestamps = [
        new Date(Date.UTC(2024, 0, 15, 9, 0, 0)),  // Morning (day)
        new Date(Date.UTC(2024, 0, 15, 21, 0, 0)), // Night
      ];
      
      const predictions = model.predictBatchWithTimestamps(testTimestamps);
      
      expect(predictions.length).toBe(2);
      expect(predictions[0]).toBeLessThan(predictions[1]); // Night should have higher probability
    });
  });
});
