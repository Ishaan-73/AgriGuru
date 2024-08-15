import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IonicModule } from '@ionic/angular';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-check',
  templateUrl: './check.page.html',
  styleUrls: ['./check.page.scss'],
  standalone: true,
  imports: [IonicModule, CommonModule, FormsModule]
})
export class CheckPage implements OnInit {

  constructor(private http: HttpClient) { }



  ngOnInit() {
    this.func();
  }

  public temperature = "";
  public humidity = "";

  public func() {
    this.http.get('http://127.0.0.1:5000/hello4').subscribe((data: any) => {
      this.temperature = data['temperature'];
      this.humidity = data['humidity'];
      console.log(this.temperature,this.humidity);
    });
  }

}
