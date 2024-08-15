import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IonSelect, IonicModule, LoadingController, NavController, ToastController } from '@ionic/angular';
import { Router } from '@angular/router';
import { SharedService } from '../shared.service';
import { OtpRequest } from '../otp-request';
import { TranslateLoader, TranslateModule, TranslateService } from '@ngx-translate/core';


import { StoreModule } from '@ngrx/store';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { CountryData } from '../country-data';
import { Constants } from '../constants';


@Component({
 selector: 'app-login',
 templateUrl: './login.page.html',
 styleUrls: ['./login.page.scss'],
 standalone: true,
 imports: [IonicModule, CommonModule, FormsModule,StoreModule,HttpClientModule,TranslateModule
          
        ],
 providers: [SharedService,TranslateService]
})
export class LoginPage implements OnInit{


 isChecked: boolean = false;
 isSelected: boolean = false;
 isSelected1: boolean = false;
 isSelected2: boolean = false;
 email: string | undefined;
 mobileNumber: string | undefined;
 otpRequest: OtpRequest = {
   mobile_number: "",
   email: "",
   country_phone_code: "",
   user_type: "GARAGE_OWNER"
 }
 translate: TranslateService | undefined;
allCountries: CountryData[] = [];
selectedCountry: string = "";
storedLang = localStorage.getItem(Constants.LANGUAGE);


 @ViewChild("email",{read: ElementRef}) emailRef : ElementRef | undefined;
 @ViewChild("password",{read: ElementRef}) passwordRef : ElementRef | undefined;
//  @ViewChild("country") selectCountryRef: ElementRef | undefined;


 constructor(private router: Router,
   private service: SharedService,
   private http: HttpClient,
   private toastControl: ToastController,
   private navController: NavController,
   private loadingController: LoadingController,
   ) {
  }

   ngOnInit(){
    if (this.storedLang){
      if (this.storedLang == " Arabic ")
          this.translate  = this.service.getAppliedLanguageTranslateService('ar');
      else if (this.storedLang == " English ")
          this.translate  = this.service.getAppliedLanguageTranslateService('en');
  }
     
   }

 checkTerms(){
   this.isChecked = !this.isChecked;
   console.log(this.isChecked);
}

changeisSelected1(){
  if (this.passwordRef?.nativeElement.value)
    this.isSelected1 = true;
  else
    this.isSelected1 = false;
}

changeisSelected2(){
  if (this.emailRef?.nativeElement.value)
    this.isSelected2 = true;
  else
    this.isSelected2 = false;
}

backToGetStarted(){
  this.navController.navigateBack(['/start']);
}
sendOTP(){

}



}
