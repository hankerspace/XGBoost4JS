/**
 * Interface for training data point
 */
export interface TrainingDataPoint {
  timestamp: Date | number; // Unix timestamp in ms or Date object
  features?: Record<string, number>; // User-provided features
  target: number; // Value to predict
}

/**
 * Interface for prediction input
 */
export interface PredictionInput {
  timestamp: Date | number;
  features?: Record<string, number>;
}

/**
 * Interface for timestamp-based features extracted automatically
 */
export interface TimestampFeatures {
  hour: number; // 0-23
  dayOfWeek: number; // 0-6 (0=Sunday)
  dayOfMonth: number; // 1-31
  month: number; // 0-11
  isWeekend: number; // 0 or 1
  isDayTime: number; // 0 or 1 (1 if hour between 6-18)
  quarterOfDay: number; // 0-3 (0: 0-6h, 1: 6-12h, 2: 12-18h, 3: 18-24h)
  weekOfYear: number; // 1-53
}

/**
 * Combined features for training/prediction
 */
export interface CombinedFeatures extends TimestampFeatures {
  [key: string]: number; // User features merged with timestamp features
}

/**
 * XGBoost model configuration
 */
export interface XGBoostConfig {
  maxDepth?: number; // Maximum depth of trees
  learningRate?: number; // Learning rate (eta)
  numTrees?: number; // Number of boosting rounds
  minChildWeight?: number; // Minimum sum of instance weight needed in a child
  subsample?: number; // Subsample ratio of training instances
  colsampleByTree?: number; // Subsample ratio of columns when constructing each tree
}

/**
 * Tree node structure
 */
export interface TreeNode {
  isLeaf: boolean;
  value: number; // Prediction value for leaf, split threshold for internal node
  featureIndex?: number; // Feature to split on (for internal nodes)
  featureName?: string; // Name of the feature
  left?: TreeNode;
  right?: TreeNode;
}

/**
 * Decision tree structure
 */
export interface DecisionTree {
  root: TreeNode;
  weight: number; // Weight of this tree in the ensemble
}
