import { TestBed } from '@angular/core/testing';
import { TimestampFeatureService } from './timestamp-feature.service';

describe('TimestampFeatureService', () => {
  let service: TimestampFeatureService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(TimestampFeatureService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  describe('extractTimestampFeatures', () => {
    it('should extract features from a Date object', () => {
      // Monday, January 15, 2024, 10:30:00
      const date = new Date(2024, 0, 15, 10, 30, 0);
      const features = service.extractTimestampFeatures(date);

      expect(features.hour).toBe(10);
      expect(features.dayOfWeek).toBe(1); // Monday
      expect(features.dayOfMonth).toBe(15);
      expect(features.month).toBe(0); // January
      expect(features.isWeekend).toBe(0); // Monday is not weekend
      expect(features.isDayTime).toBe(1); // 10:30 is daytime
      expect(features.quarterOfDay).toBe(1); // 10:30 is in quarter 1 (6-12h)
      expect(features.weekOfYear).toBeGreaterThan(0);
    });

    it('should extract features from a timestamp number', () => {
      // Monday, January 15, 2024, 10:30:00
      const timestamp = new Date(2024, 0, 15, 10, 30, 0).getTime();
      const features = service.extractTimestampFeatures(timestamp);

      expect(features.hour).toBe(10);
      expect(features.dayOfWeek).toBe(1);
    });

    it('should correctly identify weekend', () => {
      // Saturday
      const saturday = new Date(2024, 0, 13, 10, 0, 0); // Jan 13, 2024 is Saturday
      const saturdayFeatures = service.extractTimestampFeatures(saturday);
      expect(saturdayFeatures.isWeekend).toBe(1);

      // Sunday
      const sunday = new Date(2024, 0, 14, 10, 0, 0); // Jan 14, 2024 is Sunday
      const sundayFeatures = service.extractTimestampFeatures(sunday);
      expect(sundayFeatures.isWeekend).toBe(1);

      // Monday
      const monday = new Date(2024, 0, 15, 10, 0, 0); // Jan 15, 2024 is Monday
      const mondayFeatures = service.extractTimestampFeatures(monday);
      expect(mondayFeatures.isWeekend).toBe(0);
    });

    it('should correctly identify day/night', () => {
      // Daytime (10:00)
      const day = new Date(2024, 0, 15, 10, 0, 0);
      const dayFeatures = service.extractTimestampFeatures(day);
      expect(dayFeatures.isDayTime).toBe(1);

      // Nighttime (22:00)
      const night = new Date(2024, 0, 15, 22, 0, 0);
      const nightFeatures = service.extractTimestampFeatures(night);
      expect(nightFeatures.isDayTime).toBe(0);

      // Early morning (4:00)
      const earlyMorning = new Date(2024, 0, 15, 4, 0, 0);
      const earlyMorningFeatures = service.extractTimestampFeatures(earlyMorning);
      expect(earlyMorningFeatures.isDayTime).toBe(0);
    });

    it('should correctly calculate quarter of day', () => {
      const q0 = new Date(2024, 0, 15, 3, 0, 0); // 03:00 - Quarter 0
      expect(service.extractTimestampFeatures(q0).quarterOfDay).toBe(0);

      const q1 = new Date(2024, 0, 15, 8, 0, 0); // 08:00 - Quarter 1
      expect(service.extractTimestampFeatures(q1).quarterOfDay).toBe(1);

      const q2 = new Date(2024, 0, 15, 14, 0, 0); // 14:00 - Quarter 2
      expect(service.extractTimestampFeatures(q2).quarterOfDay).toBe(2);

      const q3 = new Date(2024, 0, 15, 20, 0, 0); // 20:00 - Quarter 3
      expect(service.extractTimestampFeatures(q3).quarterOfDay).toBe(3);
    });
  });

  describe('combineFeatures', () => {
    it('should combine timestamp features with user features', () => {
      const date = new Date(2024, 0, 15, 10, 30, 0);
      const userFeatures = {
        temperature: 25.5,
        humidity: 60
      };

      const combined = service.combineFeatures(date, userFeatures);

      expect(combined.hour).toBe(10);
      expect(combined['temperature']).toBe(25.5);
      expect(combined['humidity']).toBe(60);
    });

    it('should work without user features', () => {
      const date = new Date(2024, 0, 15, 10, 30, 0);
      const combined = service.combineFeatures(date);

      expect(combined.hour).toBe(10);
      expect(combined.dayOfWeek).toBe(1);
    });
  });

  describe('featuresToArray and getFeatureNames', () => {
    it('should convert features to array in consistent order', () => {
      const features = {
        hour: 10,
        temperature: 25,
        dayOfWeek: 1
      };

      const array = service.featuresToArray(features);
      const names = service.getFeatureNames(features);

      expect(array.length).toBe(3);
      expect(names.length).toBe(3);
      
      // Check that values correspond to names
      names.forEach((name, index) => {
        expect(array[index]).toBe(features[name as keyof typeof features]);
      });
    });

    it('should maintain consistent order across multiple calls', () => {
      const features = {
        z: 3,
        a: 1,
        m: 2
      };

      const array1 = service.featuresToArray(features);
      const array2 = service.featuresToArray(features);
      const names1 = service.getFeatureNames(features);
      const names2 = service.getFeatureNames(features);

      expect(array1).toEqual(array2);
      expect(names1).toEqual(names2);
      // Should be sorted alphabetically
      expect(names1).toEqual(['a', 'm', 'z']);
      expect(array1).toEqual([1, 2, 3]);
    });
  });
});
