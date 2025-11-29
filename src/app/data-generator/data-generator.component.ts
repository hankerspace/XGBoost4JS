import { Component, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Chart, registerables } from 'chart.js';
import { XGBoostService } from '../xgboost.service';
import { XGBoostParams } from 'xgboost-js';
import { MetricsDisplayComponent } from '../components/metrics-display/metrics-display.component';

Chart.register(...registerables);

@Component({
  selector: 'app-data-generator',
  standalone: true,
  imports: [CommonModule, FormsModule, MetricsDisplayComponent],
  templateUrl: './data-generator.component.html',
  styleUrls: ['./data-generator.component.css']
})
export class DataGeneratorComponent implements AfterViewInit {
  // Charts
  @ViewChild('mainChart') mainChartRef!: ElementRef<HTMLCanvasElement>;
  chart!: Chart;

  // State
  data: number[][] = []; // N x M
  featuresCount = 3;
  dataLength = 744;

  // Generation Modes
  generationMode = 'simple';
  generationModes = [
    { value: 'simple', label: 'Simple (Sin/Cos)' },
    { value: 'trend', label: 'Trend + Seasonality' },
    { value: 'break', label: 'Structural Break' },
    { value: 'chaos', label: 'Chaotic (Interaction)' }
  ];

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
  namedImportance: { name: string; value: number }[] = [];
  testDataRows: { timestamp: string; target: number; prediction: number; features: number[] }[] = [];
  rawFeatureNames: string[] = [];
  isLoading = false;
  timestamps: string[] = [];
  startDate: string = '';

  constructor(private svc: XGBoostService) {
    const now = new Date();
    now.setMinutes(0, 0, 0);
    this.startDate = this.toLocalISOString(now);
  }

  toLocalISOString(d: Date): string {
    const offset = d.getTimezoneOffset() * 60000; // offset in milliseconds
    const localDate = new Date(d.getTime() - offset);
    return localDate.toISOString().slice(0, 16);
  }

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
    let generators: ((t: number) => number)[];

    switch (this.generationMode) {
      case 'trend':
        // Trend + Seasonality (Daily cycle assuming t=hours)
        generators = [
          (t: number) => (t / 500) + Math.sin(t * 2 * Math.PI / 24) + (Math.random() - 0.5) * 0.2, // F0: Trend + Season
          (t: number) => (t / 500) + (Math.random() - 0.5) * 0.1, // F1: Trend only
          (t: number) => Math.sin(t * 2 * Math.PI / 24) + (Math.random() - 0.5) * 0.1 // F2: Season only
        ];
        break;

      case 'break':
        // Structural Break halfway
        const breakPoint = Math.floor(this.dataLength / 2);
        generators = [
          (t: number) => (t > breakPoint ? 2 : 0) + Math.sin(t * 0.1) + (Math.random() - 0.5) * 0.2, // F0
          (t: number) => (t > breakPoint ? 1 : 0), // F1: Indicator
          (t: number) => Math.cos(t * 0.1) + (Math.random() - 0.5) * 0.1 // F2
        ];
        break;

      case 'chaos':
        // Multiplicative / Non-linear (F0 = F1 * F2)
        generators = [
          (t: number) => (Math.sin(t * 0.05) * Math.cos(t * 0.15)) + (Math.random() - 0.5) * 0.1, // F0
          (t: number) => Math.sin(t * 0.05), // F1
          (t: number) => Math.cos(t * 0.15) // F2
        ];
        break;

      case 'simple':
      default:
        generators = [
          (t: number) => Math.sin(t * 0.1) + (Math.random() - 0.5) * 0.2,
          (t: number) => Math.cos(t * 0.1) + (Math.random() - 0.5) * 0.1,
          (t: number) => (Math.random() - 0.5) * 2
        ];
        break;
    }

    this.data = this.svc.generateMultivariate(this.dataLength, generators);

    // Generate Timestamps & Features
    const now = new Date(this.startDate);

    this.timestamps = [];

    for (let i = 0; i < this.dataLength; i++) {
       const d = new Date(now.getTime() + i * 3600000); // 1 hour step

       // Formatted string for chart
       const dateStr = d.toISOString().slice(5, 16).replace('T', ' '); // MM-DD HH:mm
       this.timestamps.push(dateStr);

    }

    this.rawFeatureNames = [
      'F0 (Sine)', 'F1 (Cos)', 'F2 (Rnd)'
    ];

    this.updateChartData();
    this.metrics = null;
    this.testDataRows = [];
    this.namedImportance = [];
  }

  updateChartData() {
    if (!this.chart) return;

    this.chart.data.labels = this.timestamps;
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
      const res: any = this.svc.trainAndEvaluate(
        this.xgbParams,
        this.data,
        this.targetIndex,
        this.lag,
        this.trainRatio
      );

      this.metrics = res.metrics;
      this.importance = res.importance;

      // Map Importance to Names
      const featureNames = res.featureNames || [];
      this.namedImportance = res.importance.map((gain: number, i: number) => ({
        name: featureNames[i] || `Feat ${i}`,
        value: gain
      })).sort((a: any, b: any) => b.value - a.value);

      // Update Chart with Predictions
      const totalValid = this.dataLength - this.lag;
      const splitIdx = Math.floor(totalValid * this.trainRatio);
      const testStartIndex = this.lag + splitIdx;

      // Create Prediction Dataset padded with nulls
      const predData = new Array(this.dataLength).fill(null);

      // Prepare Table Rows (Test Set)
      this.testDataRows = [];

      res.testResults.predicted.forEach((val: number, i: number) => {
        const idx = testStartIndex + i;
        if (idx < this.dataLength) {
          predData[idx] = val;

          this.testDataRows.push({
             timestamp: this.timestamps[idx],
             target: res.testResults.actual[i],
             prediction: val,
             features: this.data[idx]
          });
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
