import { Component, signal, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { XGBoostService } from './services/xgboost.service';
import { TrainingDataPoint } from './models/xgboost.interface';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  protected readonly title = signal('XGBoost4JS');
  
  // Demo data
  protected trained = false;
  protected predictionResult: number | null = null;
  protected modelInfo = '';
  
  // User inputs
  protected selectedDate = new Date().toISOString().slice(0, 16);
  protected customFeatureValue = 25;
  protected targetValue = 50;

  constructor(private xgboostService: XGBoostService) {}

  ngOnInit() {
    // Auto-train with demo data on initialization
    this.trainModel();
  }

  /**
   * Train the model with sample data
   */
  trainModel() {
    // Create sample training data: energy consumption prediction
    // Pattern: higher consumption during weekdays and during work hours
    const trainingData: TrainingDataPoint[] = [
      // Weekday mornings (high consumption)
      { timestamp: new Date(2024, 0, 15, 8, 0, 0), features: { temperature: 15 }, target: 75 },
      { timestamp: new Date(2024, 0, 16, 9, 0, 0), features: { temperature: 16 }, target: 80 },
      { timestamp: new Date(2024, 0, 17, 8, 30, 0), features: { temperature: 14 }, target: 78 },
      
      // Weekday afternoons (high consumption)
      { timestamp: new Date(2024, 0, 15, 14, 0, 0), features: { temperature: 20 }, target: 85 },
      { timestamp: new Date(2024, 0, 16, 15, 0, 0), features: { temperature: 22 }, target: 90 },
      { timestamp: new Date(2024, 0, 17, 16, 0, 0), features: { temperature: 21 }, target: 88 },
      
      // Weekday nights (low consumption)
      { timestamp: new Date(2024, 0, 15, 22, 0, 0), features: { temperature: 12 }, target: 30 },
      { timestamp: new Date(2024, 0, 16, 23, 0, 0), features: { temperature: 11 }, target: 25 },
      { timestamp: new Date(2024, 0, 17, 21, 0, 0), features: { temperature: 13 }, target: 35 },
      
      // Weekend (lower consumption)
      { timestamp: new Date(2024, 0, 13, 10, 0, 0), features: { temperature: 18 }, target: 40 },
      { timestamp: new Date(2024, 0, 14, 11, 0, 0), features: { temperature: 19 }, target: 45 },
      { timestamp: new Date(2024, 0, 20, 14, 0, 0), features: { temperature: 20 }, target: 42 },
    ];

    this.xgboostService.train(trainingData, {
      numTrees: 20,
      maxDepth: 4,
      learningRate: 0.1
    });

    this.trained = true;
    const summary = this.xgboostService.getModelSummary();
    this.modelInfo = `Model trained with ${summary.numTrees} trees. Features: ${summary.featureNames.join(', ')}`;
  }

  /**
   * Make a prediction based on user inputs
   */
  makePrediction() {
    if (!this.trained) {
      return;
    }

    const timestamp = new Date(this.selectedDate);
    const prediction = this.xgboostService.predict({
      timestamp,
      features: { temperature: this.customFeatureValue }
    });

    this.predictionResult = Math.round(prediction * 100) / 100;
  }

  /**
   * Get formatted date string
   */
  protected getFormattedDate(): string {
    const date = new Date(this.selectedDate);
    return date.toLocaleString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }
}
