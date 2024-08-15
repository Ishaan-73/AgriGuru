import { Injectable } from '@angular/core';
import { ActivatedRoute, ActivatedRouteSnapshot, CanActivate, CanActivateFn, CanLoad, Resolve, Router, RouterStateSnapshot, UrlTree } from '@angular/router';
import { Observable } from 'rxjs';
import { Constants } from './constants';

@Injectable({
  providedIn: 'root'
})
export class AuthGuard {
  // canActivate(
  //   route: ActivatedRouteSnapshot,
  //   state: RouterStateSnapshot): Observable<boolean | UrlTree> | Promise<boolean | UrlTree> | boolean | UrlTree {
  //   return true;
  // }
  constructor(private router: Router,private activatedRoute: ActivatedRoute) {}

  public canActivate: CanActivateFn = (
    route: ActivatedRouteSnapshot,
    state: RouterStateSnapshot
  ): Observable<boolean | UrlTree> | Promise<boolean | UrlTree> | boolean | UrlTree => {
    

  //   let currentRoute = "";
  //   this.activatedRoute.url.subscribe(urlSegments => {
  //     currentRoute = urlSegments.join('/')
  //   });
  //   if (localStorage.getItem(Constants.ISLOGGEDIN) == "true") {
  //     if (currentRoute == "speaking")
  //       return true;
  //     else if (currentRoute == "result")
  //       return true;
  //     else if (currentRoute == "selection")
  //       return true;
  //     else
  //       this.router.navigate(['/selection']);
  //     return true;
  //   } 
  //   else {
  //     if (currentRoute !== "start" && currentRoute !== "")
  //       this.router.navigate(['/login']);
  //     return false;
  //   }
  if (localStorage.getItem(Constants.ISLOGGEDIN) == "false" || localStorage.getItem(Constants.ISLOGGEDIN) == null){
    this.router.navigate(['/start']);
    return false;
  }
  
  return true;
  };
  
}
