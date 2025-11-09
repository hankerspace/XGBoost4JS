import { TestBed } from '@angular/core/testing';
import { App } from './app';
import { XGBoostService } from './services/xgboost.service';
import { TimestampFeatureService } from './services/timestamp-feature.service';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [XGBoostService, TimestampFeatureService]
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should render title', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('XGBoost4JS');
  });

  it('should train model on initialization', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    fixture.detectChanges();
    expect(app['trained']).toBe(true);
  });
});
