import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IonicModule, NavController } from '@ionic/angular';
import { TranslateModule, TranslateService } from '@ngx-translate/core';
import { SharedService } from '../shared.service';
import { Constants } from '../constants';
import { ActivatedRoute, Router } from '@angular/router';

@Component({
  selector: 'app-start',
  templateUrl: './start.page.html',
  styleUrls: ['./start.page.scss'],
  standalone: true,
  imports: [IonicModule, CommonModule, FormsModule,TranslateModule]
})
export class StartPage implements OnInit {

  translate: TranslateService | undefined;
  constructor(private navController: NavController,
              private service: SharedService,
              private router: Router,
              private activatedRoute: ActivatedRoute) { 
}

  languages = ["English","Arabic"];
  storedLang = localStorage.getItem(Constants.LANGUAGE);
  lang: string | undefined;


  ngOnInit() {
    let currentRoute = "";
      this.activatedRoute.url.subscribe(urlSegments => {
        currentRoute = urlSegments.join('/')
      });
      console.log(currentRoute);
    console.log(this.storedLang);
    this.translate  = this.service.getAppliedLanguageTranslateService('en');
    if (this.storedLang){
      if (this.storedLang == " Arabic " || this.storedLang == "Arabic")
          this.translate  = this.service.getAppliedLanguageTranslateService('ar');
      else if (this.storedLang == " English " || this.storedLang == "English")
          this.translate  = this.service.getAppliedLanguageTranslateService('en');
    }
                
  }

  getStartedClicked(){
    this.navController.navigateForward(['/login']);
  }

  langSelected(lang: string){
    this.lang = lang;
    console.log(lang);
    if (this.translate){
      if (lang == this.translate?.instant(" English ") || lang == this.translate?.instant("English")){
        // this.translate = this.service.getAppliedLanguageTranslateService('en');
        console.log("in1");
        this.translate?.use('en');
        localStorage.setItem(Constants.LANGUAGE,lang);

        // window.location.reload();
        }
      else if (lang == this.translate?.instant(" Arabic ") || lang == this.translate?.instant("Arabic")){
          // this.translate = this.service.getAppliedLanguageTranslateService('ar');
          console.log("in2");
          this.translate?.use('ar');
          localStorage.setItem(Constants.LANGUAGE,lang);

      // window.location.reload();
      }
    }
    




  }

}
