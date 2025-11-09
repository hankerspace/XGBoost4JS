import { Injectable } from '@angular/core';
import { TimestampFeatures, CombinedFeatures } from '../models/xgboost.interface';

/**
 * Service for extracting features from timestamps
 */
@Injectable({
  providedIn: 'root'
})
export class TimestampFeatureService {

  /**
   * Extract timestamp-based features from a date or timestamp
   */
  extractTimestampFeatures(timestamp: Date | number): TimestampFeatures {
    const date = typeof timestamp === 'number' ? new Date(timestamp) : timestamp;
    
    const hour = date.getHours();
    const dayOfWeek = date.getDay(); // 0 = Sunday, 6 = Saturday
    const dayOfMonth = date.getDate();
    const month = date.getMonth();
    
    // Weekend detection (Saturday=6, Sunday=0)
    const isWeekend = (dayOfWeek === 0 || dayOfWeek === 6) ? 1 : 0;
    
    // Day/Night detection (daytime is 6:00 - 18:00)
    const isDayTime = (hour >= 6 && hour < 18) ? 1 : 0;
    
    // Quarter of day (0-3)
    const quarterOfDay = Math.floor(hour / 6);
    
    // Week of year calculation
    const weekOfYear = this.getWeekOfYear(date);
    
    return {
      hour,
      dayOfWeek,
      dayOfMonth,
      month,
      isWeekend,
      isDayTime,
      quarterOfDay,
      weekOfYear
    };
  }

  /**
   * Combine timestamp features with user-provided features
   */
  combineFeatures(timestamp: Date | number, userFeatures?: Record<string, number>): CombinedFeatures {
    const timestampFeatures = this.extractTimestampFeatures(timestamp);
    
    return {
      ...timestampFeatures,
      ...(userFeatures || {})
    };
  }

  /**
   * Calculate the week number of the year (ISO week date)
   */
  private getWeekOfYear(date: Date): number {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    const dayNum = d.getUTCDay() || 7;
    d.setUTCDate(d.getUTCDate() + 4 - dayNum);
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    return Math.ceil((((d.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
  }

  /**
   * Convert combined features to an array of values
   */
  featuresToArray(features: Record<string, number>): number[] {
    const keys = Object.keys(features).sort(); // Sort for consistency
    return keys.map(key => features[key]);
  }

  /**
   * Get feature names in the same order as featuresToArray
   */
  getFeatureNames(features: Record<string, number>): string[] {
    return Object.keys(features).sort();
  }
}
