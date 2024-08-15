import { Injectable } from '@angular/core';
import { TextToSpeech } from '@capacitor-community/text-to-speech';
import { Constants } from './constants';

@Injectable({
  providedIn: 'root'
})
export class TtsService {
  async speak(answer: string){
      console.log("in");
      console.log(localStorage.getItem(Constants.LANGUAGE));
      await TextToSpeech.speak({
        text: answer,
        lang: 'en',
        rate: 0.7,
        pitch: 1.0,
        volume: 1.0,
        category: 'ambient',
      });
    
  }

  async stop(){
    
      await TextToSpeech.stop();
    };
  
}
