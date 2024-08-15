import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AlertController, IonicModule, LoadingController, NavController, ToastController } from '@ionic/angular';
import { ActivatedRoute, Router } from '@angular/router';
import { Constants } from '../constants';
import { TtsService } from '../tts.service';
import { SpeechRecognition } from '@capacitor-community/speech-recognition';
import { SharedService } from '../shared.service';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Location } from '@angular/common';
import { TranslateModule, TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'app-result',
  templateUrl: './result.page.html',
  styleUrls: ['./result.page.scss'],
  standalone: true,
  imports: [IonicModule, CommonModule, FormsModule,HttpClientModule,TranslateModule],
  providers: [TtsService,SharedService]
})
export class ResultPage implements OnInit {

  public question: string | undefined = "";
  public answer: string | undefined;
  speaking: boolean = false;
  innerHtml: string = "";
  text: string = "";
  flag = false;
  message = "";
  message2 = "";
  brand: string = localStorage.getItem(Constants.BRAND) as string;
  model: string = localStorage.getItem(Constants.MODEL) as string;

  translate: TranslateService | undefined;
  storedLang = localStorage.getItem(Constants.LANGUAGE);
  
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
  ];

  // say = require('say');

  constructor(private navControl: NavController,
              private router: Router,
              private ttsService: TtsService,
              private alertController: AlertController,
              private toastControl: ToastController,
              private servie: SharedService,
               private location: Location,
               private activatedRoute: ActivatedRoute,
               private loadingController: LoadingController,
               private http: HttpClient
              ) { }

  ngOnInit() {
    let currentRoute = "";
    this.activatedRoute.url.subscribe(urlSegments => {
      currentRoute = urlSegments.join('/')
    });
    console.log(currentRoute);
      this.message2 = localStorage.getItem(Constants.QUESTION) as string ;
      this.innerHtml = localStorage.getItem(Constants.ANSWER) as string;
      this.answer = this.innerHtml;
      console.log(this.innerHtml);
    
    if (this.storedLang){
      if (this.storedLang == " Arabic ")
          this.translate  = this.servie.getAppliedLanguageTranslateService('ar');
      else if (this.storedLang == " English ")
          this.translate  = this.servie.getAppliedLanguageTranslateService('en');

  }
  }


  // async LOGOUT(){
  //   console.log("LOGOUT");
  //   const alert = await this.alertController.create({
  //     header: 'Confirmation',
  //     message: 'Are you sure you want to logout?',
  //     buttons: this.logoutButtons,
  //   });

  //   await alert.present();
  // }

  recordAudio(){
    
    SpeechRecognition.requestPermissions()
    .then(() => {
      SpeechRecognition.start({
        language: localStorage.getItem(Constants.LANGUAGE) as string,
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

  anotherQuestion(){
    this.navControl.navigateBack(['/speaking']);
  }

  backToSpeaking(){
    this.navControl.navigateBack(['/speaking']);
  }

  textToSpeech(){
    if (this.speaking === false){
      this.speaking = true;
      console.log("IN");
      var myDiv = document.getElementById("response");
      var unorderedList = myDiv?.querySelector("ul");
      var listItems = unorderedList?.getElementsByTagName("li");
      var contents = [];
      if (listItems)
        for (var i = 0; i < listItems.length; i++) {
          var listItem = listItems[i];
          contents.push(listItem.textContent);
        }
      var plainText = contents.join("\n");
      console.log(plainText);
      this.ttsService.speak(plainText);
      // this.say.speak(localStorage.getItem(Constants.ANSWER));
    }
    else{
      this.speaking = false;
      console.log("OUT");
      this.ttsService.stop();
      // this.say.stop();
    }
  }

  async submitQuery(){

    let loading= await this.loadingController.create({
      message: '',
      cssClass:'ion-loading-class',
      translucent: true,
      showBackdrop: true,
      spinner: 'lines',
      keyboardClose: true,
      duration:5000,
  });
     await loading.present();


     localStorage.setItem(Constants.QUESTION,this.message);

    this.http.post('http://127.0.0.1:5000/hello2', { question: this.message }).subscribe((data2: any) => {
      console.log(data2.message);
      const resultString = data2.message.substring(2, data2.message.length - 2);
      localStorage.setItem(Constants.ANSWER,resultString);
      window.location.reload();
    });


    
    
    
  }

}
