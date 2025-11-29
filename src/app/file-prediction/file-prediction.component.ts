import { Component, ViewChild, ElementRef, AfterViewInit, ChangeDetectorRef } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Chart, registerables } from 'chart.js';
import { XGBoostService } from '../xgboost.service';
import { XGBoostParams } from '../xgboost';
import * as XLSX from 'xlsx';
import { MetricsDisplayComponent } from '../components/metrics-display/metrics-display.component';

Chart.register(...registerables);

@Component({
  selector: 'app-file-prediction',
  standalone: true,
  imports: [CommonModule, FormsModule, MetricsDisplayComponent],
  templateUrl: './file-prediction.component.html',
  styleUrls: ['./file-prediction.component.css']
})
export class FilePredictionComponent implements AfterViewInit {
  // Charts
  @ViewChild('mainChart') mainChartRef!: ElementRef<HTMLCanvasElement>;
  chart!: Chart;

  // State
  data: number[][] = []; // N x M (Features only, without timestamp)
  timestamps: string[] = [];
  dataLength = 0;

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

  constructor(private svc: XGBoostService, private cdr: ChangeDetectorRef) {}

  ngAfterViewInit() {
    // Chart init deferred until data is loaded or empty chart?
    // We can init empty chart
    this.initChart();
  }

  initChart() {
    if (!this.mainChartRef) return;
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

  onFileChange(evt: any) {
    const target: DataTransfer = <DataTransfer>(evt.target);
    if (target.files.length !== 1) throw new Error('Cannot use multiple files');

    const reader: FileReader = new FileReader();
    reader.onload = (e: any) => {
      const bstr: string = e.target.result;
      const wb: XLSX.WorkBook = XLSX.read(bstr, { type: 'binary' });

      const wsname: string = wb.SheetNames[0];
      const ws: XLSX.WorkSheet = wb.Sheets[wsname];

      // Parse as array of arrays
      const data: any[][] = XLSX.utils.sheet_to_json(ws, { header: 1 });

      if (data.length < 2) {
        alert('File is too short or empty');
        return;
      }

      // Process Header
      const header = data[0];
      // Col 0 is Timestamp, Col 1.. are features
      this.rawFeatureNames = header.slice(1).map((h: any) => String(h));

      // Process Data
      this.data = [];
      this.timestamps = [];

      for (let i = 1; i < data.length; i++) {
        const row = data[i];
        if (!row || row.length === 0) continue;

        // Timestamp
        let ts = row[0];
        // Handle Excel dates if necessary, but often they come as strings or numbers
        // If number (Excel serial date), convert?
        // For simplicity, assume string or standard format.
        // If it's a number and looks like excel date:
        if (typeof ts === 'number' && ts > 20000 && ts < 60000) {
             // Simple rough check for excel date serial
             // Not implementing full conversion unless needed.
             ts = String(ts);
        }
        this.timestamps.push(String(ts));

        // Features
        const feats = row.slice(1).map((v: any) => Number(v));
        // Check for NaNs
        if (feats.some((f: number) => isNaN(f))) {
          // Handle missing? skip or fill 0
           // For now, fill 0
           this.data.push(feats.map((f: number) => isNaN(f) ? 0 : f));
        } else {
           this.data.push(feats);
        }
      }

      this.dataLength = this.data.length;
      this.targetIndex = 0; // Reset target

      // Update UI
      this.cdr.detectChanges();
      this.initChart();
      this.updateChartData();
    };
    reader.readAsBinaryString(target.files[0]);
  }

  updateChartData() {
    if (!this.chart) return;
    if (this.data.length === 0) return;

    this.chart.data.labels = this.timestamps;

    // Plot target feature and maybe a few others
    // If too many features, just plot target
    const targetData = this.data.map(row => row[this.targetIndex]);

    const datasets: any[] = [
        {
            label: this.rawFeatureNames[this.targetIndex] || `Feature ${this.targetIndex}`,
            data: targetData,
            borderColor: 'blue',
            borderWidth: 1,
            pointRadius: 0
        }
    ];

    // Optional: Plot other features hidden
    this.rawFeatureNames.forEach((name, i) => {
        if (i === this.targetIndex) return;
        datasets.push({
            label: name,
            data: this.data.map(row => row[i]),
            borderColor: 'gray',
            borderWidth: 1,
            pointRadius: 0,
            hidden: true
        });
    });

    this.chart.data.datasets = datasets;
    this.chart.update();
  }

  async train() {
    if (this.data.length === 0) return;

    this.isLoading = true;
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
      const serviceFeatureNames = res.featureNames || [];

      this.namedImportance = res.importance.map((gain: number, i: number) => {
        let name = serviceFeatureNames[i] || `Feat ${i}`;
        // Replace F0, F1 with real names if possible
        name = name.replace(/F(\d+)/g, (match: string, index: string) => {
             const idx = parseInt(index, 10);
             return this.rawFeatureNames[idx] || match;
        });

        return {
            name: name,
            value: gain
        };
      }).sort((a: any, b: any) => b.value - a.value);

      // Update Chart with Predictions
      const totalValid = this.dataLength - this.lag;
      const splitIdx = Math.floor(totalValid * this.trainRatio);
      const testStartIndex = this.lag + splitIdx;

      const predData = new Array(this.dataLength).fill(null);
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

      // Add/Update Prediction Dataset
      if (!this.chart) return;
      const predLabel = 'Prediction (' + (this.rawFeatureNames[this.targetIndex] || `F${this.targetIndex}`) + ')';
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
