import { ChangeDetectorRef, Component, ElementRef, Input, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AlertController, IonTextarea, IonicModule, LoadingController, NavController, ToastController } from '@ionic/angular';
import { DomSanitizer } from '@angular/platform-browser';
// import * as RecordRTC from 'recordrtc';
import { SpeechRecognition } from "@capacitor-community/speech-recognition";
import { SharedService } from '../shared.service';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { VoiceRecognitionService } from '../voice-recognition.service';
import { ActivatedRoute, Router } from '@angular/router';
import { Constants } from '../constants';
// import { Result } from 'range-parser';
import { Annyang } from 'annyang';
import { TranslateModule, TranslateService } from '@ngx-translate/core';
// 
// var annyang = require('annyang');
// import { MediaCapture,MediaFile,CaptureAudioOptions } from '@awesome-cordova-plugins/media-capture/ngx';
// import { FileOpener } from '@awesome-cordova-plugins/file-opener/ngx';
// import { CaptureError } from '@awesome-cordova-plugins/media-capture/ngx';
// // import { invokeSaveAsDialog } from 'recordrtc';
// import { Plugins } from '@capacitor/core';

@Component({
  selector: 'app-speaking',
  templateUrl: './speaking.page.html',
  styleUrls: ['./speaking.page.scss'],
  standalone: true,
  imports: [IonicModule, CommonModule, FormsModule,HttpClientModule,TranslateModule],
  providers: [SharedService,VoiceRecognitionService]
})
export class SpeakingPage implements OnInit {

  message: string = "";
  question: string = "";
  flag: boolean = false;
  clicked: boolean = false;
  brandName: string | undefined;
  modelName: string | undefined;
  text: string = "";
  recording = false;
  user_id: string | undefined;
  access_token: string | undefined;
  public logoutButtons = [
    {
      text: 'Cancel',
      role: 'cancel',
    },
    {
      text: 'OK',
      role: 'confirm',
      handler: () => {
        localStorage.setItem(Constants.ISLOGGEDIN,"false");
        this.router.navigate(['/login']);
      }
    },
  ];;
  // response: any | undefined;
//  annyang: Annyang | undefined;

  // @ViewChild("queryRef",{read: ElementRef}) queryRef : ElementRef | undefined;
  @ViewChild("queryRef",{read: ElementRef}) ionTextArea: IonTextarea | undefined;
  answer: string ="";
  brand: string = localStorage.getItem(Constants.BRAND) as string;
  model: string = localStorage.getItem(Constants.MODEL) as string;
  translate: TranslateService | undefined;
  storedLang = localStorage.getItem(Constants.LANGUAGE);
  async LOGOUT(){
    console.log("LOGOUT");
    const alert = await this.alertController.create({
      header: 'Confirmation',
      message: 'Are you sure you want to logout?',
      buttons: this.logoutButtons,
    });

    await alert.present();
  }
  constructor(private navControl: NavController,
              private servie: SharedService,
              public ser: VoiceRecognitionService,
              private router: Router,
              private toastControl: ToastController,
              private ref: ChangeDetectorRef,
              private alertController: AlertController,
              private loadingController: LoadingController,
              private activatedRoute: ActivatedRoute,
              private http: HttpClient
              ) { this.ser.ngOnInit();
                this.translate  = this.servie.getAppliedLanguageTranslateService('en');
                if (this.storedLang){
                  if (this.storedLang == " Arabic " || this.storedLang == "Arabic")
                      this.translate  = this.servie.getAppliedLanguageTranslateService('ar');
                  else if (this.storedLang == " English " || this.storedLang == "English")
                      this.translate  = this.servie.getAppliedLanguageTranslateService('en');
                }
              }

  ngOnInit() {
    let currentRoute = "";
      this.activatedRoute.url.subscribe(urlSegments => {
        currentRoute = urlSegments.join('/')
      });
      console.log(currentRoute);
  }
  



  recordAudio(){
    
    SpeechRecognition.requestPermissions()
    .then(() => {
      SpeechRecognition.start({
        language: "en-US",
        maxResults: 10,
        prompt: "Say something",
        partialResults: true,
        popup: true
      })
      .then((response) => {
        console.log(response);
        let l = 0;
        for (let i of response.matches){
          if (i.length > l){
            this.text = i;
            l=i.length;
          }
        }
        
        console.log(this.text);
        SpeechRecognition.removeAllListeners();
  
        SpeechRecognition.stop();
        this.message += " " + this.text;
        this.flag = true;
      });
    })

    
    
    
  }



  Q1(){
    this.message = this.translate?.instant("What is Tomato Yellow Leaf Curl Virus (TYLCV)?");
    localStorage.setItem(Constants.QUESTION,this.message);
    this.http.post('http://127.0.0.1:5000/hello2', { question: this.message }).subscribe((data2: any) => {
      localStorage.setItem(Constants.ANSWER,data2.message);
      this.navControl.navigateForward(['/result']);
    });
  }

  Q2(){
    this.message = this.translate?.instant("How is TYLCV spread?");
    localStorage.setItem(Constants.QUESTION,this.message);
    this.http.post('http://127.0.0.1:5000/hello2', { question: this.message }).subscribe((data2: any) => {
      localStorage.setItem(Constants.ANSWER,data2.message);
      this.navControl.navigateForward(['/result']);
    });
  }

  Q3(){
    this.message = this.translate?.instant("What are the primary causes of Tomato leaf curl disease?");
    localStorage.setItem(Constants.QUESTION,this.message);
    this.http.post('http://127.0.0.1:5000/hello2', { question: this.message }).subscribe((data2: any) => {
      localStorage.setItem(Constants.ANSWER,data2.message);
      this.navControl.navigateForward(['/result']);
    });
  }

  Q4(){
    this.message = this.translate?.instant("What are the symptoms of Bacterial Spot in tomatoes?");
    localStorage.setItem(Constants.QUESTION,this.message);
    this.http.post('http://127.0.0.1:5000/hello2', { question: this.message }).subscribe((data2: any) => {
      localStorage.setItem(Constants.ANSWER,data2.message);
      this.navControl.navigateForward(['/result']);
    });
  }

  Q5(){
    this.message = this.translate?.instant("How do you control bacterial spot of tomatoes?");
    localStorage.setItem(Constants.QUESTION,this.message);
    this.http.post('http://127.0.0.1:5000/hello2', { question: this.message }).subscribe((data2: any) => {
      localStorage.setItem(Constants.ANSWER,data2.message);
      this.navControl.navigateForward(['/result']);
    });
  }

  Q6(){
    this.message = this.translate?.instant("What is the causative agent of bacterial leaf spot in tomatoes?");
    localStorage.setItem(Constants.QUESTION,this.message);
    this.http.post('http://127.0.0.1:5000/hello2', { question: this.message }).subscribe((data2: any) => {
      localStorage.setItem(Constants.ANSWER,data2.message);
      this.navControl.navigateForward(['/result']);
    });
  }

  Q7(){
    this.message = this.translate?.instant("How does bacterial spot develop on tomatoes?");
    localStorage.setItem(Constants.QUESTION,this.message);
    this.http.post('http://127.0.0.1:5000/hello2', { question: this.message }).subscribe((data2: any) => {
      localStorage.setItem(Constants.ANSWER,data2.message);
      this.navControl.navigateForward(['/result']);
    });
  }


  async submitQuery(){

    let loading= await this.loadingController.create({
      message: '',
      cssClass:'ion-loading-class',
      translucent: true,
      showBackdrop: true,
      spinner: 'lines',
      keyboardClose: true,
      duration:1000,
  });
     await loading.present();

    this.clicked = true;
    localStorage.setItem(Constants.QUESTION,this.message);

    this.http.post('http://127.0.0.1:5000/hello2', { question: this.message }).subscribe((data2: any) => {
      console.log(data2.message);
      const resultString = data2.message.substring(2, data2.message.length - 2);
      localStorage.setItem(Constants.ANSWER,resultString);
      this.navControl.navigateForward(['/result']);
    });
    


    
  }
  

}
