import { Routes } from '@angular/router';
import { DataGeneratorComponent } from './data-generator/data-generator.component';
import { FilePredictionComponent } from './file-prediction/file-prediction.component';

export const routes: Routes = [
  { path: '', component: DataGeneratorComponent },
  { path: 'file-prediction', component: FilePredictionComponent }
];
