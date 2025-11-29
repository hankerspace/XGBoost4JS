// xgboost.ts

// --- Types & Interfaces ---
export type Objective = 'reg:squarederror' | 'binary:logistic';

export interface XGBoostParams {
  learningRate?: number;
  maxDepth?: number;
  minChildWeight?: number; // Cover minimum (hessian sum)
  numRounds?: number;
  regLambda?: number; // L2 regularization
  gamma?: number;     // Min loss reduction for split
  subsample?: number; // Row subsampling (not fully implemented for simplicity, but good to have in interface)
  objective?: Objective;
  seed?: number;
}

export class Node {
  isLeaf: boolean = false;
  weight: number = 0;     // Leaf value

  featureIndex: number = -1;
  threshold: number = 0;
  left: Node | null = null;
  right: Node | null = null;

  // Debug / Info
  gain: number = 0;
  cover: number = 0;
}

// --- RNG (Mulberry32) ---
export function mulberry32(a: number) {
  return function() {
    let t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
}

// --- Core XGBoost Algorithm ---
export class XGBoost {
  trees: Node[] = [];
  params: Required<XGBoostParams>;
  baseScore: number = 0.5;

  constructor(params: XGBoostParams) {
    this.params = {
      learningRate: 0.1,
      maxDepth: 3,
      minChildWeight: 1,
      numRounds: 10,
      regLambda: 1.0,
      gamma: 0.0,
      subsample: 1.0,
      objective: 'reg:squarederror',
      seed: 0,
      ...params
    };
  }

  fit(X: number[][], y: number[]) {
    this.trees = [];
    const n = X.length;
    if (n === 0) return;

    // Initial Prediction
    if (this.params.objective === 'reg:squarederror') {
      this.baseScore = y.reduce((a,b)=>a+b, 0) / n;
    } else {
      this.baseScore = 0.5;
    }

    // Initialize predictions
    let preds = new Float64Array(n);
    if (this.params.objective === 'reg:squarederror') {
       preds.fill(this.baseScore);
    } else {
       // For logistic, if baseScore=0.5, logit=0
       preds.fill(0.0);
    }

    // Gradient Boosting Loop
    for (let iter = 0; iter < this.params.numRounds; iter++) {
      const grad = new Float64Array(n);
      const hess = new Float64Array(n);

      // 1. Calculate Gradients & Hessians
      for (let i = 0; i < n; i++) {
        if (this.params.objective === 'reg:squarederror') {
          // Loss = 1/2 * (y - p)^2
          // grad = p - y
          // hess = 1
          grad[i] = preds[i] - y[i];
          hess[i] = 1.0;
        } else {
          // Binary Logistic
          // p = sigmoid(logit)
          const p = this.sigmoid(preds[i]);
          grad[i] = p - y[i];
          hess[i] = Math.max(1e-16, p * (1.0 - p));
        }
      }

      // 2. Build Tree
      const indices = Array.from({length: n}, (_, i) => i);
      const tree = this.buildTree(X, grad, hess, indices, 0);

      if (tree) {
        this.trees.push(tree);
        // 3. Update Predictions
        for (let i = 0; i < n; i++) {
          const pred = this.predictRawSingle(X[i], tree);
          preds[i] += this.params.learningRate * pred;
        }
      } else {
        break; // Cannot grow more trees
      }
    }
  }

  predict(X: number[][]): number[] {
    return X.map(row => {
      let score = 0;
      if (this.params.objective === 'reg:squarederror') {
        score = this.baseScore;
      } else {
        // binary:logistic starts at 0 logit (0.5 prob)
        score = 0;
      }

      for (const tree of this.trees) {
        score += this.params.learningRate * this.predictRawSingle(row, tree);
      }

      if (this.params.objective === 'binary:logistic') {
        return this.sigmoid(score);
      }
      return score;
    });
  }

  // Internal: Predict single row on one tree
  private predictRawSingle(row: number[], node: Node): number {
    if (node.isLeaf) return node.weight;
    if (row[node.featureIndex] < node.threshold) {
      return node.left ? this.predictRawSingle(row, node.left) : node.weight;
    } else {
      return node.right ? this.predictRawSingle(row, node.right) : node.weight;
    }
  }

  private buildTree(X: number[][], grad: Float64Array, hess: Float64Array, indices: number[], depth: number): Node | null {
    // Calculate G, H for current node
    let G = 0, H = 0;
    for (const i of indices) {
      G += grad[i];
      H += hess[i];
    }

    const node = new Node();
    const weight = -G / (H + this.params.regLambda);
    node.weight = weight;
    node.cover = H;

    if (depth >= this.params.maxDepth || indices.length < 2 || H < this.params.minChildWeight) {
      node.isLeaf = true;
      return node;
    }

    // Find Best Split
    let bestGain = -Infinity;
    let bestFeat = -1;
    let bestThresh = 0;
    let bestLeftIndices: number[] = [];
    let bestRightIndices: number[] = [];

    const m = X[0].length; // num features

    for (let f = 0; f < m; f++) {
      // Optimization: Sort only once? No, indices change.
      // Just sort the current indices by feature f.
      // Making a copy to sort to avoid messing up if we needed original order (not needed here).
      const sortedIndices = indices.slice().sort((a, b) => X[a][f] - X[b][f]);

      let G_L = 0, H_L = 0;
      for (let i = 0; i < sortedIndices.length - 1; i++) {
        const idx = sortedIndices[i];
        G_L += grad[idx];
        H_L += hess[idx];

        const G_R = G - G_L;
        const H_R = H - H_L;

        if (H_L < this.params.minChildWeight || H_R < this.params.minChildWeight) continue;

        // Skip if duplicate values
        if (X[idx][f] === X[sortedIndices[i+1]][f]) continue;

        const gain = 0.5 * (
          (G_L*G_L)/(H_L + this.params.regLambda) +
          (G_R*G_R)/(H_R + this.params.regLambda) -
          (G*G)/(H + this.params.regLambda)
        ) - this.params.gamma;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeat = f;
          bestThresh = (X[idx][f] + X[sortedIndices[i+1]][f]) / 2;
        }
      }
    }

    if (bestGain > 0) {
       for(const i of indices) {
         if (X[i][bestFeat] < bestThresh) bestLeftIndices.push(i);
         else bestRightIndices.push(i);
       }

       // Check sanity
       if(bestLeftIndices.length === 0 || bestRightIndices.length === 0) {
         node.isLeaf = true;
         return node;
       }

       node.featureIndex = bestFeat;
       node.threshold = bestThresh;
       node.gain = bestGain;

       node.left = this.buildTree(X, grad, hess, bestLeftIndices, depth + 1);
       node.right = this.buildTree(X, grad, hess, bestRightIndices, depth + 1);

       if (!node.left || !node.right) {
         node.isLeaf = true;
       }
    } else {
      node.isLeaf = true;
    }

    return node;
  }

  private sigmoid(x: number) {
    return 1 / (1 + Math.exp(-x));
  }

  getFeatureImportance(): number[] {
     const importance = new Map<number, number>();
     const traverse = (n: Node | null) => {
       if (!n || n.isLeaf) return;
       const current = importance.get(n.featureIndex) || 0;
       importance.set(n.featureIndex, current + n.gain);
       traverse(n.left);
       traverse(n.right);
     };
     for(const t of this.trees) traverse(t);

     const maxIdx = Math.max(...importance.keys());
     if (maxIdx < 0) return [];
     const res = new Array(maxIdx + 1).fill(0);
     importance.forEach((val, key) => res[key] = val);
     return res;
  }
}

// --- Time Series & Feature Engineering Tools ---

/**
 * Generates a multivariate time series matrix (N x M).
 * features defines the list of generators.
 */
export function generateMultivariateSeries(
  length: number,
  generators: ((t: number) => number)[]
): number[][] {
  const data: number[][] = [];
  for (let t = 0; t < length; t++) {
    const row = generators.map(gen => gen(t));
    data.push(row);
  }
  return data;
}

/**
 * Prepares dataset for predicting ONE feature using:
 * - Lags of ALL features (history).
 * - Current values of OTHER features (concurrent context), if enabled.
 *
 * @param data Matrix N x M (N time steps, M features)
 * @param targetFeatureIndex Index of the feature to predict
 * @param lags Number of past steps to include
 * @param useConcurrentOthers If true, includes values of other features at time t
 */
export function prepareMultivariateDataset(
  data: number[][],
  targetFeatureIndex: number,
  lags: number,
  useConcurrentOthers: boolean = true
): { X: number[][], y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  const n = data.length;
  const m = data[0].length;

  // We need at least 'lags' history
  for (let i = lags; i < n; i++) {
    const features: number[] = [];

    // 1. History (Lags) of ALL features
    // Order: Lag 1 (all features), Lag 2 (all features)...
    for (let k = 1; k <= lags; k++) {
       for (let f = 0; f < m; f++) {
         features.push(data[i - k][f]);
       }
    }

    // 2. Concurrent Others (Current time step values of non-target features)
    if (useConcurrentOthers) {
      for (let f = 0; f < m; f++) {
        if (f !== targetFeatureIndex) {
          features.push(data[i][f]);
        }
      }
    }

    X.push(features);
    y.push(data[i][targetFeatureIndex]);
  }

  return { X, y };
}

// --- Metrics ---
export function calcMetrics(pred: number[], actual: number[]) {
  const n = pred.length;
  if (n === 0) return { mae:0, rmse:0, bias:0, r2:0 };
  let sumAbs = 0, sumSq = 0, sumErr = 0;
  let sumActual = 0;

  for(let i=0; i<n; i++) {
    const err = pred[i] - actual[i];
    sumAbs += Math.abs(err);
    sumSq += err*err;
    sumErr += err;
    sumActual += actual[i];
  }

  const meanActual = sumActual / n;
  let sst = 0;
  for(let i=0; i<n; i++) {
      sst += Math.pow(actual[i] - meanActual, 2);
  }
  const sse = sumSq;
  const r2 = sst > 1e-9 ? 1 - (sse/sst) : 0;

  return {
    mae: sumAbs / n,
    rmse: Math.sqrt(sumSq / n),
    bias: sumErr / n,
    r2: r2
  };
}
