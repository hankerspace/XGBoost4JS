import { Component, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Chart, registerables } from 'chart.js';
import { XGBoostService } from './xgboost.service';
import { XGBoostParams } from './xgboost';

Chart.register(...registerables);

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App implements AfterViewInit {
  // Charts
  @ViewChild('mainChart') mainChartRef!: ElementRef<HTMLCanvasElement>;
  chart!: Chart;

  // State
  data: number[][] = []; // N x M
  featuresCount = 3;
  dataLength = 200;

  // Params
  targetIndex = 0;
  lag = 5;
  trainRatio = 0.8;

  xgbParams: XGBoostParams = {
    learningRate: 0.1,
    maxDepth: 3,
    minChildWeight: 1,
    numRounds: 50,
    gamma: 0,
    regLambda: 1
  };

  // Results
  metrics: any = null;
  importance: number[] = [];
  isLoading = false;

  constructor(private svc: XGBoostService) {}

  ngAfterViewInit() {
    this.initChart();
    this.generate();
  }

  initChart() {
    const ctx = this.mainChartRef.nativeElement.getContext('2d');
    if (!ctx) return;

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: []
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: { position: 'top' },
          tooltip: { enabled: true }
        },
        scales: {
          x: { display: true },
          y: { display: true }
        }
      }
    });
  }

  generate() {
    // Generators for 3 features
    // F0: Sine Wave + Noise (Target)
    // F1: Cosine Wave (Correlated)
    // F2: Random Trend (Distractor or Helper)
    const generators = [
      (t: number) => Math.sin(t * 0.1) + (Math.random() - 0.5) * 0.2,
      (t: number) => Math.cos(t * 0.1) + (Math.random() - 0.5) * 0.1,
      (t: number) => (Math.random() - 0.5) * 2 // Random Noise / Walk
    ];

    this.data = this.svc.generateMultivariate(this.dataLength, generators);
    this.updateChartData();
    this.metrics = null;
  }

  updateChartData() {
    if (!this.chart) return;

    const labels = Array.from({length: this.dataLength}, (_, i) => i);

    this.chart.data.labels = labels;
    this.chart.data.datasets = [
      {
        label: 'F0 (Target: Sine)',
        data: this.data.map(row => row[0]),
        borderColor: 'blue',
        borderWidth: 1,
        pointRadius: 0
      },
      {
        label: 'F1 (Cosine)',
        data: this.data.map(row => row[1]),
        borderColor: 'green',
        borderWidth: 1,
        pointRadius: 0,
        hidden: true // Hide by default to reduce clutter
      },
      {
        label: 'F2 (Random)',
        data: this.data.map(row => row[2]),
        borderColor: 'gray',
        borderWidth: 1,
        pointRadius: 0,
        hidden: true
      }
    ];
    this.chart.update();
  }

  async train() {
    this.isLoading = true;
    // Yield to UI
    await new Promise(r => setTimeout(r, 50));

    try {
      const res = this.svc.trainAndEvaluate(
        this.xgbParams,
        this.data,
        this.targetIndex,
        this.lag,
        this.trainRatio
      );

      this.metrics = res.metrics;
      this.importance = res.importance;

      // Update Chart with Predictions
      // Predictions correspond to the Test Set
      // Test Set starts at splitIdx = floor(N * ratio)
      // Actually, prepareDataset starts at index 'lag'.
      // The Service handles split. We need to map predictions back to time indices.

      // We'll reconstruct where the test set is.
      // The service does: X is from [lag..N-1].
      // Train is first X_train items. Test is the rest.
      // Let's calculate indices to plot correctly.

      // Since service doesn't return indices, we'll deduce:
      // Total valid points = N - lag.
      // Split index = floor((N - lag) * ratio).
      // Test points start at: lag + splitIdx.

      const totalValid = this.dataLength - this.lag;
      const splitIdx = Math.floor(totalValid * this.trainRatio);
      const testStartIndex = this.lag + splitIdx;

      // Create Prediction Dataset padded with nulls
      const predData = new Array(this.dataLength).fill(null);
      res.testResults.predicted.forEach((val, i) => {
        if (testStartIndex + i < this.dataLength) {
          predData[testStartIndex + i] = val;
        }
      });

      // Add or Update Prediction Dataset
      const predLabel = 'Prediction (F' + this.targetIndex + ')';
      const existingIdx = this.chart.data.datasets.findIndex(d => d.label?.startsWith('Prediction'));

      const dataset = {
        label: predLabel,
        data: predData,
        borderColor: 'red',
        borderWidth: 2,
        pointRadius: 0
      };

      if (existingIdx >= 0) {
        this.chart.data.datasets[existingIdx] = dataset;
      } else {
        this.chart.data.datasets.push(dataset);
      }

      this.chart.update();

    } catch (e) {
      console.error(e);
      alert(e);
    } finally {
      this.isLoading = false;
    }
  }
}
