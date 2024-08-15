import { Component } from '@angular/core';
import { AlertController, IonicModule } from '@ionic/angular';
import { Constants } from './constants';
import { ActivatedRoute, Router } from '@angular/router';
import { TranslateLoader, TranslateModule, TranslateService } from '@ngx-translate/core';
import { HttpClient } from '@angular/common/http';
import { TranslateHttpLoader } from '@ngx-translate/http-loader';
import { SharedService } from './shared.service';
import { CommonModule } from '@angular/common';



@Component({
  selector: 'app-root',
  templateUrl: 'app.component.html',
  styleUrls: ['app.component.scss'],
  standalone: true,
  imports: [IonicModule,TranslateModule,CommonModule],
  providers: [SharedService]
})
export class AppComponent {
  constructor(private alertController: AlertController,
              private router: Router,
              private service: SharedService,
              private actRoute: ActivatedRoute
              ) {
                // let currentRoute = "";
                //   this.actRoute.url.subscribe(urlSegments => {
                //     currentRoute = urlSegments.join('/')
                //   });
                // console.log(currentRoute);
                // this.translate  = this.service.getAppliedLanguageTranslateService('en');
                // if (this.storedLang){
                //   if (this.storedLang == " Arabic " || this.storedLang == "Arabic")
                //       this.translate  = this.service.getAppliedLanguageTranslateService('ar');
                //   else if (this.storedLang == " English " || this.storedLang == "English")
                //       this.translate  = this.service.getAppliedLanguageTranslateService('en');
                // }
                // console.log(localStorage.getItem(Constants.ISLOGGEDIN));
                // if (localStorage.getItem(Constants.ISLOGGEDIN) == "true" && currentRoute !== ""){
                //   // let currentRoute = "";
                //   // this.actRoute.url.subscribe(urlSegments => {
                //   //   currentRoute = urlSegments.join('/')
                //   // });
                //   // if (currentRoute !== "speaking")
                //     this.router.navigate(['/selection']);
                // }
                
              }
  
  
  // name =  localStorage.getItem(Constants.OWNERNAME);
  
  // firstLetter = this.name?.charAt(0);
  // email = localStorage.getItem(Constants.EMAIL);
  // phone = localStorage.getItem(Constants.MOBILE_NUMBER);
  // // translate: TranslateService;
  // storedLang = localStorage.getItem(Constants.LANGUAGE);
  // languages = ["English","Arabic"];
  // cancel = "";
  // OK = "";
  // Conf = "";
  // Sure = "";

  // public logoutButtons = [
  //   {
  //     text: this.cancel,
  //     role: 'cancel',
  //   },
  //   {
  //     text: this.OK,
  //     role: 'confirm',
  //     handler: () => {
  //       localStorage.setItem(Constants.ISLOGGEDIN,"false");
  //       this.router.navigate(['/login']);
  //     }
  //   },
  // ];

  // async LOGOUT(){
  //   console.log("LOGOUT");
  //   this.cancel = this.translate?.instant("Cancel");
  //   this.OK = this.translate?.instant("OK");
  //   this.Conf = this.translate?.instant("Confirmation");
  //   this.Sure = this.translate?.instant("Are you sure you want to logout?");
  //   const alert = await this.alertController.create({
  //     header: this.Conf,
  //     message: this.Sure,
  //     buttons: [
  //       {
  //         text: this.cancel,
  //         role: 'cancel',
  //       },
  //       {
  //         text: this.OK,
  //         role: 'confirm',
  //         handler: () => {
  //           localStorage.setItem(Constants.ISLOGGEDIN,"false");
  //           localStorage.removeItem( "ACCESS_TOKEN");
  //           localStorage.removeItem( "GARAGE_ID");
  //           localStorage.removeItem( "USER_ID");
  //           localStorage.removeItem( "EMAIL" );
  //           localStorage.removeItem( "PHONE_NUMBER" );
  //           localStorage.removeItem( "BRAND" );
  //           localStorage.removeItem( "MODEL" );
  //           localStorage.removeItem( "QUESTION" );
  //           localStorage.removeItem( "ANSWER" );
  //           localStorage.removeItem( "PHONECODE" );
  //           localStorage.removeItem( "COUNTRYCODE" );
  //           localStorage.removeItem( "OWNERNAME" );
  //           console.log("LogOut");
  //           this.router.navigate(['/start']);
  //         }
  //       },
  //     ],
  //   });

  //   await alert.present();
  // }

  // changeLang(lang: any){
  //   console.log(lang,this.translate);
  //   if (this.translate){
  //     if (lang == this.translate?.instant(" English ") || lang == this.translate?.instant("English")){
  //       // this.translate = this.service.getAppliedLanguageTranslateService('en');
  //       console.log("in1");
  //       this.translate?.use('en');
  //       localStorage.setItem(Constants.LANGUAGE,"English");

  //       // window.location.reload();
  //       }
  //     else if (lang == this.translate?.instant(" Arabic ") || lang == this.translate?.instant("Arabic")){
  //         // this.translate = this.service.getAppliedLanguageTranslateService('ar');
  //         console.log("in2");
  //         this.translate?.use('ar');
  //         localStorage.setItem(Constants.LANGUAGE,lang);

  //     // window.location.reload();
  //     }
  //   }
  //   else{
      
  //   }
  
  
  // }

  clickSelection(){
    this.router.navigate(['/selection']);
  }

  clickSpeaking(){
    this.router.navigate(['/speaking']);
  }

  clickCheck(){
    this.router.navigate(['/check']);
  }
    
  
}
