import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-metrics-display',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './metrics-display.component.html',
})
export class MetricsDisplayComponent implements OnChanges {
  @Input() metrics: any;

  // Circle properties
  radius = 56;
  circumference = 2 * Math.PI * this.radius;
  strokeDashoffset = this.circumference;
  r2Score = 0;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['metrics'] && this.metrics) {
      this.r2Score = this.metrics.r2;
      // Clamp r2Score between 0 and 1 for the gauge (visual only)
      // R2 can be negative, we'll treat negative as 0 for the gauge progress
      const progress = Math.max(0, Math.min(1, this.r2Score));
      this.strokeDashoffset = this.circumference - progress * this.circumference;
    }
  }
}
