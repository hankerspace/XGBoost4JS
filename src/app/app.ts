import { Component, ElementRef, ViewChild, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { XGBoostService } from './xgboost.service';
import type { XGBoostParams } from './xgboost';
import type { SeriesType } from './xgboost';
import { Chart } from 'chart.js/auto';

// Local helper types for strongly-typed SVG scales and scatter data
type Pair = readonly [number, number];
type TrainPoint = { x: number; y: number; c: number };
type TestPoint = { x: number; y: number; c: number; p: number; a: number };
interface Scatter<TPoint> {
  points: TPoint[];
  domain: { x: Pair; y: Pair };
}

@Component({
  selector: 'app-root',
  imports: [CommonModule, FormsModule],
  templateUrl: './timeseries.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('XGBoost4JS — Time Series');

  // Etat UI séries
  seriesType: SeriesType = 'sine';
  amplitude = 1;
  frequency = 3; // nombre de cycles sur la longueur totale
  noise = 0.05;
  phase = 0;
  trendSlope = 0;
  drift = 0;
  totalLen = 400;
  trainLen = 300;
  lag = 20;
  
  // Timestamp-related properties
  useTimestamps = true; // Toggle to use timestamp features
  generateCustomFeatures = false; // Toggle to generate additional custom features
  startDate = new Date('2024-01-01T00:00:00Z'); // Starting date for the series
  intervalHours = 1; // Time interval between points in hours

  params: XGBoostParams = {
    learningRate: 0.1,
    maxDepth: 4,
    minChildWeight: 1,
    numRounds: 150,
    task: 'regression',
  };

  series: number[] = [];
  timestamps: Date[] = [];
  customFeatures: number[][] = [];
  preds: number[] = [];
  truth: number[] = [];
  mae = 0;
  rmse = 0;
  mse = 0;
  mape = 0;
  r2 = 0;
  bias = 0;
  maxAe = 0;
  nPreds = 0;

  @ViewChild('rawChart') rawChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('predChart') predChartRef?: ElementRef<HTMLCanvasElement>;
  private rawChart?: Chart;
  private predChart?: Chart;
  // Loading state for training/prediction
  isLoading = false;
  // Loading state for hyperparameter tuning
  isTuning = false;
  // Progress state for hyperparameter tuning
  tuneTotal = 0;
  tuneDone = 0;

  // Hyperparameter tuning candidates (comma-separated)
  tuneLr = '0.05,0.1,0.2';
  tuneDepth = '3,4,5';
  tuneMinChildWeight = '1,2,5';
  tuneRounds = '100,200';

  tuneResults: { params: XGBoostParams; mae: number; rmse: number; bias: number }[] = [];
  // Range for heatmap coloring
  tuneMinRmse = 0;
  tuneMaxRmse = 0;

  constructor(private svc: XGBoostService) {}

  ngAfterViewInit() {
    this.generateSeries();
    this.initRawChart();
    this.initPredChart();
  }

  private initRawChart() {
    if (!this.rawChartRef) return;
    const ctx = this.rawChartRef.nativeElement.getContext('2d');
    if (!ctx) return;
    this.rawChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Train (brut)', data: [], borderColor: '#2563eb', tension: 0.2, pointRadius: 0 },
          { label: 'Test (brut)', data: [], borderColor: '#0ea5e9', borderDash: [6,4], tension: 0.2, pointRadius: 0 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 't' } },
          y: { title: { display: true, text: 'valeur' } },
        },
        plugins: {
          legend: { position: 'top' },
        },
      },
    });
    this.updateRawChart();
  }

  private initPredChart() {
    if (!this.predChartRef) return;
    const ctx = this.predChartRef.nativeElement.getContext('2d');
    if (!ctx) return;
    this.predChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Vérité terrain', data: [], borderColor: '#16a34a', tension: 0.2, pointRadius: 0 },
          { label: 'Prévision', data: [], borderColor: '#f97316', tension: 0.2, pointRadius: 0 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 't' } },
          y: { title: { display: true, text: 'valeur' } },
        },
        plugins: { legend: { position: 'top' } },
      },
    });
    this.updatePredChart();
  }

  private updateRawChart() {
    if (!this.rawChart) return;
    const full = this.series.slice(0, this.totalLen);
    const labels = Array.from({ length: full.length }, (_, i) => i);
    const trainData = full.map((v, i) => (i < this.trainLen ? v : null));
    const testData = full.map((v, i) => (i >= this.trainLen ? v : null));
    this.rawChart.data.labels = labels as any;
    this.rawChart.data.datasets[0].data = trainData as any;
    this.rawChart.data.datasets[1].data = testData as any;
    this.rawChart.update();
  }

  private updatePredChart() {
    if (!this.predChart) return;
    const labels = Array.from({ length: this.truth.length }, (_, i) => this.trainLen + i);
    this.predChart.data.labels = labels as any;
    this.predChart.data.datasets[0].data = this.truth as any;
    this.predChart.data.datasets[1].data = this.preds as any;
    this.predChart.update();
  }

  generateSeries() {
    if (this.useTimestamps) {
      // Generate with timestamps
      const totalNeeded = this.totalLen;
      const intervalMs = this.intervalHours * 3600000; // Convert hours to milliseconds
      const result = this.svc.generateSeriesWithTimestamps(
        this.seriesType,
        totalNeeded,
        this.startDate,
        intervalMs,
        {
          amplitude: this.amplitude,
          frequency: this.frequency,
          noise: this.noise,
          phase: this.phase,
          trendSlope: this.trendSlope,
          drift: this.drift,
          generateCustomFeatures: this.generateCustomFeatures,
        }
      );
      
      this.series = result.values;
      this.timestamps = result.timestamps;
      this.customFeatures = result.customFeatures || [];
    } else {
      // Generate without timestamps (legacy mode)
      const totalNeeded = this.totalLen + this.lag;
      this.series = this.svc.generateSeries(this.seriesType, totalNeeded, {
        amplitude: this.amplitude,
        frequency: this.frequency,
        noise: this.noise,
        phase: this.phase,
        trendSlope: this.trendSlope,
        drift: this.drift,
      });
      this.timestamps = [];
      this.customFeatures = [];
    }
    
    // Réinitialiser sorties
    this.preds = [];
    this.truth = this.series.slice(this.trainLen, this.totalLen);
    this.mae = 0;
    this.rmse = 0;
    this.updateRawChart();
    this.updatePredChart();
  }

  async trainAndForecast() {
    // show loading spinner and yield to UI before heavy sync work
    this.isLoading = true;
    await new Promise<void>((resolve) => setTimeout(resolve));
    try {
      if (this.useTimestamps && this.timestamps.length > 0) {
        // Use timestamp-based training
        const customFeat = this.customFeatures.length > 0 ? this.customFeatures : undefined;
        const { preds, truth, mae, rmse, mse, mape, r2, bias, maxAe } = this.svc.trainAndForecastWithTimestamps(
          this.params,
          this.timestamps,
          this.series,
          this.trainLen,
          customFeat
        );
        this.preds = preds;
        this.truth = truth;
        this.mae = mae;
        this.rmse = rmse;
        this.mse = mse;
        this.mape = mape;
        this.r2 = r2;
        this.bias = bias;
        this.maxAe = maxAe;
        this.nPreds = truth.length;
      } else {
        // Use legacy lag-based training
        const forecastLen = this.totalLen - this.trainLen;
        const { preds, truth, mae, rmse, mse, mape, r2, bias, maxAe } = this.svc.trainAndForecastTS(
          this.params,
          this.series,
          this.lag,
          this.trainLen,
          forecastLen
        );
        this.preds = preds;
        this.truth = truth;
        this.mae = mae;
        this.rmse = rmse;
        this.mse = mse;
        this.mape = mape;
        this.r2 = r2;
        this.bias = bias;
        this.maxAe = maxAe;
        this.nPreds = truth.length;
      }
      this.updatePredChart();
    } finally {
      this.isLoading = false;
    }
  }

  reset() {
    this.series = [];
    this.preds = [];
    this.truth = [];
    this.mae = 0;
    this.rmse = 0;
    this.updateRawChart();
    this.updatePredChart();
  }

  private parseNumbers(list: string): number[] {
    return list
      .split(',')
      .map((s) => parseFloat(s.trim()))
      .filter((v) => Number.isFinite(v));
  }

  async tuneHyperparams() {
    this.isTuning = true;
    // Laisser le temps d'afficher l'overlay
    await new Promise<void>((r) => setTimeout(r));
    try {
      const space = {
        learningRates: this.parseNumbers(this.tuneLr),
        maxDepths: this.parseNumbers(this.tuneDepth),
        minChildWeights: this.parseNumbers(this.tuneMinChildWeight),
        rounds: this.parseNumbers(this.tuneRounds),
      };

      // Calculer le nombre total de simulations attendues (borné par le maxCombos du service)
      const rawTotal =
        (space.learningRates.length || 0) *
        (space.maxDepths.length || 0) *
        (space.minChildWeights.length || 0) *
        (space.rounds.length || 0);
      const maxCombos = 120;
      this.tuneTotal = Math.min(rawTotal, maxCombos);
      this.tuneDone = 0;

      this.tuneResults = await this.svc.tuneHyperparamsTS(
        space,
        this.series,
        this.lag,
        this.trainLen,
        maxCombos,
        (done, total) => {
          this.tuneDone = done;
          this.tuneTotal = total;
        }
      );

      // Mettre à jour l'échelle pour le gradient (rmse min->max)
      if (this.tuneResults.length) {
        this.tuneMinRmse = Math.min(...this.tuneResults.map((r) => r.rmse));
        this.tuneMaxRmse = Math.max(...this.tuneResults.map((r) => r.rmse));
      } else {
        this.tuneMinRmse = this.tuneMaxRmse = 0;
      }
    } finally {
      this.isTuning = false;
    }
  }

  applyBestParams() {
    if (!this.tuneResults?.length) return;
    const best = this.tuneResults[0];
    this.params = { ...this.params, ...best.params };
  }

  applyParamsFrom(row: { params: XGBoostParams }) {
    if (!row) return;
    this.params = { ...this.params, ...row.params };
  }

}
