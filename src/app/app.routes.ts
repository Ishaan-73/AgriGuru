import { Routes } from '@angular/router';
import { AuthGuard } from './auth.guard';


export const routes: Routes = [
  {
    path: 'home',
    loadComponent: () => import('./home/home.page').then((m) => m.HomePage),
  },
  {
    path: '',
    redirectTo: 'selection',
    pathMatch: 'full',
  },
  // {
  //   path: 'start',
  //   loadComponent: () => import('./start/start.page').then( m => m.StartPage)
  // },
  {
    path: 'login',
    loadComponent: () => import('./login/login.page').then( m => m.LoginPage)
  },
  // {
  //   path: 'verify-login',
  //   loadComponent: () => import('./verify-login/verify-login.page').then( m => m.VerifyLoginPage)
  // },
  {
    path: 'selection',
    loadComponent: () => import('./selection/selection.page').then( m => m.SelectionPage)
  },
  {
    path: 'speaking',
    loadComponent: () => import('./speaking/speaking.page').then( m => m.SpeakingPage)
  },
  {
    path: 'result',
    loadComponent: () => import('./result/result.page').then( m => m.ResultPage)
  },
  {
    path: 'check',
    loadComponent: () => import('./check/check.page').then( m => m.CheckPage)
  },
  
];
