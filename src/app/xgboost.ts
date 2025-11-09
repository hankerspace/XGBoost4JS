// Simplified XGBoost-like gradient boosting for binary classification in TypeScript
// Ported from the issue description and adapted for Angular usage (ES modules)

class Node {
  featureIndex: number | null = null;
  threshold: number | null = null;
  left: Node | null = null;
  right: Node | null = null;
  value: number | null = null; // leaf weight
  isLeaf = false;
}

export type TaskType = 'binary' | 'regression';

export interface XGBoostParams {
  learningRate?: number;
  maxDepth?: number;
  minChildWeight?: number;
  numRounds?: number;
  task?: TaskType; // 'binary' par défaut; 'regression' pour séries temporelles
  lambda?: number; // L2 regularization on leaf weights
  gamma?: number; // Minimum split loss reduction to make a split
  subsample?: number; // Row subsampling rate per tree
  colsampleByTree?: number; // Column subsampling rate per tree
  seed?: number; // RNG seed for reproducibility
}

export class XGBoost {
  learningRate: number;
  maxDepth: number;
  minChildWeight: number;
  numRounds: number;
  task: TaskType;
  trees: { root: Node }[] = [];
  lambda: number;
  gamma: number;
  subsample: number;
  colsampleByTree: number;
  seed: number;
  private rng!: () => number;
  private baseScore = 0; // raw score (before link)
  private nFeatures = 0;
  private importanceGain: number[] = [];
  private importanceCount: number[] = [];

  constructor(params: XGBoostParams = {}) {
    this.learningRate = params.learningRate ?? 0.3;
    this.maxDepth = params.maxDepth ?? 4;
    this.minChildWeight = params.minChildWeight ?? 1;
    this.numRounds = params.numRounds ?? 100;
    this.task = params.task ?? 'binary';
    this.lambda = params.lambda ?? 1;
    this.gamma = params.gamma ?? 0;
    this.subsample = params.subsample ?? 1;
    this.colsampleByTree = params.colsampleByTree ?? 1;
    this.seed = params.seed ?? 1337;
    this.rng = mulberry32(this.seed);
  }

  fit(X: number[][], y: number[]): void {
    if (!X.length || !X[0]?.length || X.length !== y.length) return;
    this.trees = [];
    this.nFeatures = X[0].length;
    this.importanceGain = new Array(this.nFeatures).fill(0);
    this.importanceCount = new Array(this.nFeatures).fill(0);

    // Base score initialization
    if (this.task === 'binary') {
      const meanY = Math.min(1 - 1e-6, Math.max(1e-6, y.reduce((a, b) => a + b, 0) / y.length));
      this.baseScore = Math.log(meanY / (1 - meanY)); // logit
    } else {
      this.baseScore = y.reduce((a, b) => a + b, 0) / y.length;
    }

    // Raw predictions start at baseScore
    let raw = new Array(y.length).fill(this.baseScore) as number[];

    for (let round = 0; round < this.numRounds; round++) {
      // Compute first and second order gradients
      const grad: number[] = new Array(y.length);
      const hess: number[] = new Array(y.length);
      if (this.task === 'binary') {
        for (let i = 0; i < y.length; i++) {
          const p = sigmoid(raw[i]);
          grad[i] = p - y[i]; // d/dz of logistic loss
          hess[i] = Math.max(1e-6, p * (1 - p));
        }
      } else {
        for (let i = 0; i < y.length; i++) {
          grad[i] = raw[i] - y[i];
          hess[i] = 1; // squared error
        }
      }

      // Row subsampling
      const rowIdx: number[] = [];
      const rate = Math.max(0, Math.min(1, this.subsample));
      if (rate >= 1) {
        for (let i = 0; i < y.length; i++) rowIdx.push(i);
      } else {
        for (let i = 0; i < y.length; i++) if (this.rng() < rate) rowIdx.push(i);
        if (rowIdx.length === 0) rowIdx.push(Math.floor(this.rng() * y.length));
      }

      // Column subsampling per tree
      const featIdx: number[] = [];
      const colRate = Math.max(0, Math.min(1, this.colsampleByTree));
      if (colRate >= 1) {
        for (let f = 0; f < this.nFeatures; f++) featIdx.push(f);
      } else {
        const shuffled = Array.from({ length: this.nFeatures }, (_, i) => i).sort(() => this.rng() - 0.5);
        const k = Math.max(1, Math.round(colRate * this.nFeatures));
        for (let i = 0; i < k; i++) featIdx.push(shuffled[i]);
      }

      // Build matrices for the subsampled rows
      const Xsub = rowIdx.map((i) => X[i]);
      const gsub = rowIdx.map((i) => grad[i]);
      const hsub = rowIdx.map((i) => hess[i]);

      const tree = this._buildTree(Xsub, gsub, hsub, 0, featIdx);
      this.trees.push({ root: tree });

      // Update raw predictions for ALL rows with shrinkage
      for (let i = 0; i < X.length; i++) {
        raw[i] += this.learningRate * this._predictRaw(X[i], tree);
      }
    }
  }

  private _buildTree(
    X: number[][],
    grad: number[],
    hess: number[],
    depth: number,
    featureIndices: number[]
  ): Node {
    const node = new Node();
    const G = grad.reduce((a, b) => a + b, 0);
    const H = hess.reduce((a, b) => a + b, 0);

    // Stop criteria
    if (
      depth >= this.maxDepth ||
      H < (this.minChildWeight ?? 1) ||
      X.length <= 1
    ) {
      node.isLeaf = true;
      node.value = -G / (H + this.lambda);
      return node;
    }

    let bestGain = -Infinity;
    let best: { f: number; thr: number; L: number[]; R: number[] } | null = null;

    const m = X.length;
    for (const f of featureIndices) {
      // Build and sort tuples (value, grad, hess, index)
      const tuples: { v: number; g: number; h: number; i: number }[] = new Array(m);
      for (let i = 0; i < m; i++) tuples[i] = { v: X[i][f], g: grad[i], h: hess[i], i };
      tuples.sort((a, b) => a.v - b.v);

      let GL = 0;
      let HL = 0;
      let GR = G;
      let HR = H;
      for (let i = 0; i < m - 1; i++) {
        const t = tuples[i];
        GL += t.g;
        HL += t.h;
        GR -= t.g;
        HR -= t.h;
        // skip equal feature values to avoid zero-width split
        if (tuples[i].v === tuples[i + 1].v) continue;
        if (HL < this.minChildWeight || HR < this.minChildWeight) continue;

        const gain = 0.5 * (GL * GL) / (HL + this.lambda) +
                     0.5 * (GR * GR) / (HR + this.lambda) -
                     0.5 * (G * G) / (H + this.lambda) - this.gamma;

        if (gain > bestGain) {
          bestGain = gain;
          const thr = (tuples[i].v + tuples[i + 1].v) / 2;
          const L: number[] = [];
          const R: number[] = [];
          for (let k = 0; k < m; k++) (X[tuples[k].i][f] <= thr ? L : R).push(tuples[k].i);
          best = { f, thr, L, R };
        }
      }
    }

    if (!best || bestGain <= 0) {
      node.isLeaf = true;
      node.value = -G / (H + this.lambda);
      return node;
    }

    node.featureIndex = best.f;
    node.threshold = best.thr;
    // Update feature importance
    if (this.importanceGain.length) {
      this.importanceGain[best.f] += Math.max(0, bestGain);
      this.importanceCount[best.f] += 1;
    }

    const leftX = best.L.map((idx) => X[idx]);
    const leftG = best.L.map((idx) => grad[idx]);
    const leftH = best.L.map((idx) => hess[idx]);
    node.left = this._buildTree(leftX, leftG, leftH, depth + 1, featureIndices);

    const rightX = best.R.map((idx) => X[idx]);
    const rightG = best.R.map((idx) => grad[idx]);
    const rightH = best.R.map((idx) => hess[idx]);
    node.right = this._buildTree(rightX, rightG, rightH, depth + 1, featureIndices);

    return node;
  }

  predictSingle(x: number[]): number {
    let raw = this.baseScore;
    for (const tree of this.trees) raw += this.learningRate * this._predictRaw(x, tree.root);
    return this.task === 'binary' ? sigmoid(raw) : raw;
  }

  predictBatch(X: number[][]): number[] {
    return X.map((x) => this.predictSingle(x));
  }

  private _predictRaw(x: number[], node: Node): number {
    if (node.isLeaf) return node.value ?? 0;
    if ((node.featureIndex ?? 0) >= x.length || node.threshold == null) return 0; // safety
    return x[node.featureIndex!] <= node.threshold ? this._predictRaw(x, node.left!) : this._predictRaw(x, node.right!);
  }

  getFeatureImportance(): number[] {
    // Return total gain per feature (closer to xgboost's importance=Gain)
    if (!this.importanceGain.length) return [];
    return this.importanceGain.slice();
  }

  getFeatureImportanceCounts(): number[] {
    return this.importanceCount.slice();
  }

  private _getNumFeatures(node: Node | null): number {
    if (!node) return 0;
    if (node.isLeaf) return 0;
    return Math.max((node.featureIndex ?? -1) + 1, this._getNumFeatures(node.left), this._getNumFeatures(node.right));
  }

  private _traverseTreeForImportance(node: Node | null, importance: number[]): void {
    if (!node || node.isLeaf) return;
    if (node.featureIndex != null) importance[node.featureIndex]++;
    this._traverseTreeForImportance(node.left, importance);
    this._traverseTreeForImportance(node.right, importance);
  }

  toJSON(): { trees: { root: Node }[]; params: XGBoostParams; meta: { baseScore: number; nFeatures: number; importanceGain: number[]; importanceCount: number[] } } {
    return {
      trees: this.trees,
      params: {
        learningRate: this.learningRate,
        maxDepth: this.maxDepth,
        minChildWeight: this.minChildWeight,
        numRounds: this.numRounds,
        task: this.task,
        lambda: this.lambda,
        gamma: this.gamma,
        subsample: this.subsample,
        colsampleByTree: this.colsampleByTree,
        seed: this.seed,
      },
      meta: { baseScore: this.baseScore, nFeatures: this.nFeatures, importanceGain: this.importanceGain, importanceCount: this.importanceCount }
    };
  }

  static fromJSON(json: { trees: { root: Node }[]; params: XGBoostParams; meta?: { baseScore: number; nFeatures: number; importanceGain: number[]; importanceCount: number[] } }): XGBoost {
    const model = new XGBoost(json.params ?? {});
    model.trees = json.trees ?? [];
    if (json.meta) {
      model['baseScore'] = json.meta.baseScore ?? 0;
      model['nFeatures'] = json.meta.nFeatures ?? 0;
      model['importanceGain'] = json.meta.importanceGain ?? [];
      model['importanceCount'] = json.meta.importanceCount ?? [];
    }
    return model;
  }
}

export type DatasetType = 'binary' | 'nonlinear' | 'noisy';

export function generateDataset(type: DatasetType = 'binary', n = 1000): { X: number[][]; y: number[] } {
  switch (type) {
    case 'binary':
      return generateBinaryData(n);
    case 'nonlinear':
      return generateNonlinearData(n);
    case 'noisy':
      return generateNoisyData(n);
  }
}

function generateBinaryData(n: number): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = Math.random() * 10;
    const x2 = Math.random() * 10;
    X.push([x1, x2]);
    y.push(x1 + x2 > 10 ? 1 : 0);
  }
  return { X, y };
}

function generateNonlinearData(n: number): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = Math.random() * 8 - 4;
    const x2 = Math.random() * 8 - 4;
    X.push([x1, x2]);
    y.push(x1 * x1 + x2 * x2 < 9 ? 1 : 0);
  }
  return { X, y };
}

function generateNoisyData(n: number): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = Math.random() * 10;
    const x2 = Math.random() * 10;
    X.push([x1, x2]);
    const trueLabel = x1 + x2 > 10 ? 1 : 0;
    y.push(Math.random() < 0.1 ? 1 - trueLabel : trueLabel);
  }
  return { X, y };
}

export function metrics(predictions: number[], actual: number[]) {
  const binaryPreds = predictions.map((p) => (p >= 0.5 ? 1 : 0));
  let tp = 0,
    fp = 0,
    tn = 0,
    fn = 0;
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] === 1 && binaryPreds[i] === 1) tp++;
    if (actual[i] === 0 && binaryPreds[i] === 1) fp++;
    if (actual[i] === 0 && binaryPreds[i] === 0) tn++;
    if (actual[i] === 1 && binaryPreds[i] === 0) fn++;
  }
  const accuracy = (tp + tn) / (tp + tn + fp + fn);
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = (2 * (precision * recall)) / (precision + recall) || 0;
  return { accuracy, precision, recall, f1, cm: { tp, fp, tn, fn }, binaryPreds };
}

// ===== Helpers Séries Temporelles =====
export type SeriesType =
  | 'sine'
  | 'cosine'
  | 'square'
  | 'sawtooth'
  | 'trend'
  | 'trend_sine'
  | 'randomwalk';

export function generateTimeSeries(
  type: SeriesType,
  total: number,
  options?: {
    amplitude?: number;
    frequency?: number; // nombre de cycles sur [0..total)
    phase?: number; // déphasage en radians
    noise?: number; // bruit uniforme ±noise
    trendSlope?: number; // pente (variation totale sur l'intervalle [0..1])
    drift?: number; // dérive par pas pour random walk
  }
): number[] {
  const A = options?.amplitude ?? 1;
  const f = options?.frequency ?? 1;
  const phase = options?.phase ?? 0;
  const noise = options?.noise ?? 0;
  const slope = options?.trendSlope ?? 0; // utilisé par trend / trend_sine
  const drift = options?.drift ?? 0; // utilisé par randomwalk

  const data: number[] = [];
  if (type === 'randomwalk') {
    let v = 0;
    for (let t = 0; t < total; t++) {
      const step = (Math.random() * 2 - 1) * (A || 1) + drift;
      v += step;
      const n = noise > 0 ? (Math.random() * 2 - 1) * noise : 0;
      data.push(v + n);
    }
    return data;
  }

  for (let t = 0; t < total; t++) {
    const x = (2 * Math.PI * f * t) / total + phase;
    let base = 0;
    switch (type) {
      case 'sine':
        base = Math.sin(x);
        break;
      case 'cosine':
        base = Math.cos(x);
        break;
      case 'square':
        base = Math.sin(x) >= 0 ? 1 : -1;
        break;
      case 'sawtooth': {
        // valeur dans [-1,1] basée sur la phase fractionnaire
        const frac = (x / (2 * Math.PI)) - Math.floor((x / (2 * Math.PI)) + 0.5);
        base = 2 * frac; // ~[-1,1]
        break;
      }
      case 'trend':
        base = 0; // pas d'onde, uniquement la tendance
        break;
      case 'trend_sine':
        base = Math.sin(x);
        break;
    }
    const trend = slope * (t / Math.max(1, total - 1));
    const wave = (type === 'trend' ? 0 : A * base);
    const n = noise > 0 ? (Math.random() * 2 - 1) * noise : 0;
    data.push(wave + trend + n);
  }
  return data;
}

export function windowedSupervised(series: number[], lag: number): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = lag; i < series.length; i++) {
    X.push(series.slice(i - lag, i));
    y.push(series[i]);
  }
  return { X, y };
}

export function recursiveForecast(model: XGBoost, seed: number[], steps: number): number[] {
  const lag = seed.length;
  const buf = seed.slice();
  const preds: number[] = [];
  for (let s = 0; s < steps; s++) {
    const next = model.predictSingle(buf.slice(-lag));
    preds.push(next);
    buf.push(next);
  }
  return preds;
}

// ===== Utilities =====
function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  } else {
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }
}

function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
