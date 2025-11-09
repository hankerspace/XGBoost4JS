import { Injectable } from '@angular/core';
import {
  TrainingDataPoint,
  PredictionInput,
  XGBoostConfig,
  DecisionTree,
  TreeNode,
  CombinedFeatures
} from '../models/xgboost.interface';
import { TimestampFeatureService } from './timestamp-feature.service';

/**
 * XGBoost implementation service
 */
@Injectable({
  providedIn: 'root'
})
export class XGBoostService {
  private trees: DecisionTree[] = [];
  private featureNames: string[] = [];
  private config: Required<XGBoostConfig>;
  private baseScore: number = 0;

  constructor(private timestampFeatureService: TimestampFeatureService) {
    this.config = this.getDefaultConfig();
  }

  /**
   * Get default configuration
   */
  private getDefaultConfig(): Required<XGBoostConfig> {
    return {
      maxDepth: 3,
      learningRate: 0.1,
      numTrees: 10,
      minChildWeight: 1,
      subsample: 1.0,
      colsampleByTree: 1.0
    };
  }

  /**
   * Train the XGBoost model
   */
  train(trainingData: TrainingDataPoint[], config?: XGBoostConfig): void {
    // Update config with user-provided values
    if (config) {
      this.config = { ...this.config, ...config };
    }

    // Reset model
    this.trees = [];
    this.featureNames = [];

    // Extract and combine features
    const processedData = trainingData.map(point => ({
      features: this.timestampFeatureService.combineFeatures(point.timestamp, point.features),
      target: point.target
    }));

    if (processedData.length === 0) {
      throw new Error('Training data is empty');
    }

    // Store feature names from first data point
    this.featureNames = this.timestampFeatureService.getFeatureNames(processedData[0].features);

    // Convert features to arrays
    const X = processedData.map(d => this.timestampFeatureService.featuresToArray(d.features));
    const y = processedData.map(d => d.target);

    // Calculate base score (mean of targets)
    this.baseScore = y.reduce((sum, val) => sum + val, 0) / y.length;

    // Initialize predictions with base score
    let predictions = new Array(y.length).fill(this.baseScore);

    // Build trees iteratively
    for (let i = 0; i < this.config.numTrees; i++) {
      // Calculate residuals (gradient for squared loss)
      const residuals = y.map((target, idx) => target - predictions[idx]);

      // Subsample data if needed
      const subsampleIndices = this.subsampleIndices(X.length, this.config.subsample);
      const subsampledX = subsampleIndices.map(idx => X[idx]);
      const subsampledResiduals = subsampleIndices.map(idx => residuals[idx]);

      // Build a decision tree
      const tree = this.buildTree(subsampledX, subsampledResiduals, 0);
      this.trees.push(tree);

      // Update predictions
      for (let j = 0; j < X.length; j++) {
        predictions[j] += this.config.learningRate * this.predictTree(tree, X[j]);
      }
    }
  }

  /**
   * Make a prediction
   */
  predict(input: PredictionInput): number {
    if (this.trees.length === 0) {
      throw new Error('Model not trained. Call train() first.');
    }

    const features = this.timestampFeatureService.combineFeatures(input.timestamp, input.features);
    const featureArray = this.timestampFeatureService.featuresToArray(features);

    let prediction = this.baseScore;
    for (const tree of this.trees) {
      prediction += this.config.learningRate * this.predictTree(tree, featureArray);
    }

    return prediction;
  }

  /**
   * Build a decision tree using CART algorithm
   */
  private buildTree(X: number[][], y: number[], depth: number): DecisionTree {
    const root = this.buildNode(X, y, depth);
    return {
      root,
      weight: 1.0
    };
  }

  /**
   * Build a tree node recursively
   */
  private buildNode(X: number[][], y: number[], depth: number): TreeNode {
    // Check stopping criteria
    if (depth >= this.config.maxDepth || X.length < this.config.minChildWeight || this.isHomogeneous(y)) {
      return {
        isLeaf: true,
        value: this.calculateLeafValue(y)
      };
    }

    // Find best split
    const bestSplit = this.findBestSplit(X, y);
    
    if (!bestSplit) {
      return {
        isLeaf: true,
        value: this.calculateLeafValue(y)
      };
    }

    // Split data
    const { leftIndices, rightIndices } = this.splitData(X, bestSplit.featureIndex, bestSplit.threshold);

    if (leftIndices.length < this.config.minChildWeight || rightIndices.length < this.config.minChildWeight) {
      return {
        isLeaf: true,
        value: this.calculateLeafValue(y)
      };
    }

    const leftX = leftIndices.map(i => X[i]);
    const leftY = leftIndices.map(i => y[i]);
    const rightX = rightIndices.map(i => X[i]);
    const rightY = rightIndices.map(i => y[i]);

    return {
      isLeaf: false,
      value: bestSplit.threshold,
      featureIndex: bestSplit.featureIndex,
      featureName: this.featureNames[bestSplit.featureIndex],
      left: this.buildNode(leftX, leftY, depth + 1),
      right: this.buildNode(rightX, rightY, depth + 1)
    };
  }

  /**
   * Find the best split for a node
   */
  private findBestSplit(X: number[][], y: number[]): { featureIndex: number; threshold: number; gain: number } | null {
    let bestGain = -Infinity;
    let bestFeatureIndex = -1;
    let bestThreshold = 0;

    const numFeatures = X[0].length;
    const featuresToConsider = this.subsampleFeatures(numFeatures, this.config.colsampleByTree);

    for (const featureIndex of featuresToConsider) {
      const values = X.map(row => row[featureIndex]);
      const uniqueValues = [...new Set(values)].sort((a, b) => a - b);

      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
        const gain = this.calculateGain(X, y, featureIndex, threshold);

        if (gain > bestGain) {
          bestGain = gain;
          bestFeatureIndex = featureIndex;
          bestThreshold = threshold;
        }
      }
    }

    if (bestFeatureIndex === -1) {
      return null;
    }

    return { featureIndex: bestFeatureIndex, threshold: bestThreshold, gain: bestGain };
  }

  /**
   * Calculate the gain of a split
   */
  private calculateGain(X: number[][], y: number[], featureIndex: number, threshold: number): number {
    const { leftIndices, rightIndices } = this.splitData(X, featureIndex, threshold);

    if (leftIndices.length === 0 || rightIndices.length === 0) {
      return -Infinity;
    }

    const leftY = leftIndices.map(i => y[i]);
    const rightY = rightIndices.map(i => y[i]);

    const totalVariance = this.calculateVariance(y);
    const leftVariance = this.calculateVariance(leftY);
    const rightVariance = this.calculateVariance(rightY);

    const leftWeight = leftIndices.length / y.length;
    const rightWeight = rightIndices.length / y.length;

    return totalVariance - (leftWeight * leftVariance + rightWeight * rightVariance);
  }

  /**
   * Split data based on feature and threshold
   */
  private splitData(X: number[][], featureIndex: number, threshold: number): { leftIndices: number[]; rightIndices: number[] } {
    const leftIndices: number[] = [];
    const rightIndices: number[] = [];

    for (let i = 0; i < X.length; i++) {
      if (X[i][featureIndex] <= threshold) {
        leftIndices.push(i);
      } else {
        rightIndices.push(i);
      }
    }

    return { leftIndices, rightIndices };
  }

  /**
   * Calculate variance of an array
   */
  private calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return variance;
  }

  /**
   * Calculate leaf value (mean of targets)
   */
  private calculateLeafValue(y: number[]): number {
    if (y.length === 0) return 0;
    return y.reduce((sum, val) => sum + val, 0) / y.length;
  }

  /**
   * Check if all values are approximately equal
   */
  private isHomogeneous(y: number[], epsilon: number = 1e-6): boolean {
    if (y.length === 0) return true;
    const first = y[0];
    return y.every(val => Math.abs(val - first) < epsilon);
  }

  /**
   * Predict using a single tree
   */
  private predictTree(tree: DecisionTree, features: number[]): number {
    return this.predictNode(tree.root, features);
  }

  /**
   * Traverse tree to make prediction
   */
  private predictNode(node: TreeNode, features: number[]): number {
    if (node.isLeaf) {
      return node.value;
    }

    if (node.featureIndex === undefined) {
      return 0;
    }

    if (features[node.featureIndex] <= node.value) {
      return node.left ? this.predictNode(node.left, features) : 0;
    } else {
      return node.right ? this.predictNode(node.right, features) : 0;
    }
  }

  /**
   * Subsample indices for training
   */
  private subsampleIndices(length: number, ratio: number): number[] {
    if (ratio >= 1.0) {
      return Array.from({ length }, (_, i) => i);
    }

    const sampleSize = Math.floor(length * ratio);
    const indices = Array.from({ length }, (_, i) => i);
    
    // Fisher-Yates shuffle
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    return indices.slice(0, sampleSize);
  }

  /**
   * Subsample features for each tree
   */
  private subsampleFeatures(numFeatures: number, ratio: number): number[] {
    if (ratio >= 1.0) {
      return Array.from({ length: numFeatures }, (_, i) => i);
    }

    const sampleSize = Math.floor(numFeatures * ratio);
    const indices = Array.from({ length: numFeatures }, (_, i) => i);
    
    // Fisher-Yates shuffle
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    return indices.slice(0, sampleSize);
  }

  /**
   * Get model summary
   */
  getModelSummary(): { numTrees: number; featureNames: string[]; config: Required<XGBoostConfig> } {
    return {
      numTrees: this.trees.length,
      featureNames: this.featureNames,
      config: this.config
    };
  }
}
