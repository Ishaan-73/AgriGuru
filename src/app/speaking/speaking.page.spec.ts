import { ComponentFixture, TestBed } from '@angular/core/testing';
import { SpeakingPage } from './speaking.page';

describe('SpeakingPage', () => {
  let component: SpeakingPage;
  let fixture: ComponentFixture<SpeakingPage>;

  beforeEach(async(() => {
    fixture = TestBed.createComponent(SpeakingPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }));

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
