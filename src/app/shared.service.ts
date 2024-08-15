import { ElementRef, Injectable } from '@angular/core';
import { HttpClient, HttpClientModule, HttpHeaders } from "@angular/common/http";
import { Observable } from 'rxjs';
import { OtpRequest } from './otp-request';
import { OtpResponse } from './otpResponse';
import { CustomerOtpVerificationInterface } from './otp-verification';
import { GarageInfo } from './garage-info';
import { Constants } from './constants';
import { VehicleBrand } from './vehicle-brand';
import { VehicleData } from './vehicle-data';
import { CountryData } from './country-data';
import { CustomerData } from './customer-data';
import { TranslateHttpLoader } from '@ngx-translate/http-loader'
import { TranslateService } from '@ngx-translate/core';
import { WebcamImage, WebcamModule} from 'ngx-webcam';
import { WebcamInitError, WebcamUtil} from 'ngx-webcam';
import { Data } from '@angular/router';
// import { GarageInfo } from './garage-info';


@Injectable({
 providedIn: 'root',


})




export class SharedService {


  private mobileNumber : string | undefined;
  private email : string| undefined;
  public brandName: string | undefined;
  public modelName: string | undefined;
  private message : string | undefined;

 constructor(private http: HttpClient,
  private translate: TranslateService
   ) { }


 
 private access_token: any | undefined;
 public garage_ID: any | undefined;

  setDetails(emailRef: string, mobile_numberRef: string){
   this.mobileNumber = mobile_numberRef;
   this.email = emailRef;
   localStorage.setItem(Constants.MOBILE_NUMBER,mobile_numberRef);
   localStorage.setItem(Constants.EMAIL,emailRef);
 }

 setAccessToken(accessToken: any){
  this.access_token = "Bearer "+accessToken;
  localStorage.setItem(Constants.ACCESS_TOKEN,this.access_token);
  console.log(this.access_token);
 }






createTranslateLoader(http: HttpClient) {
  return new TranslateHttpLoader(http, "assets/language/", ".json");
}
// }

getAppliedLanguageTranslateService(language_code: any) {
  this.translate.use(language_code)
  return this.translate;
}


}