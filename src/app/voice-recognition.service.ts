import { Injectable, OnInit } from '@angular/core';

declare var webkitSpeechRecognition: any;

@Injectable({
  providedIn: 'root'
})
export class VoiceRecognitionService implements OnInit{
  recognition = new webkitSpeechRecognition();
  isStopped = false;
  public text = "";
  tempWords : string | undefined;

  constructor() { }

  ngOnInit(){
    this.recognition.interimResults = true;
    this.recognition.lang = "en-US";
    this.recognition.addEventListener('result',(e: { results: Iterable<unknown> | ArrayLike<unknown>; }) => {
      const transcript = Array.from(e.results)
      .map((result: any) => result[0])
      .map((result: { transcript: any; }) => result.transcript)
      .join('');

      this.tempWords = transcript as unknown as string;
    })
  }

  start(){
    this.isStopped = false;
    this.recognition.start();
    this.recognition.addEventListener('end',(condition: any) =>{
      if (this.isStopped){
        this.recognition.stop();
      }
      else{
        this.text = this.text + ' ' + this.tempWords + '.';
        this.tempWords = '';
        this.recognition.stop();
      }
    })
  }

  stop(){
    this.isStopped = true;
    this.text = this.text + ' ' + this.tempWords + '.';
    this.tempWords = '';
    this.recognition.stop();
  }
  
}
