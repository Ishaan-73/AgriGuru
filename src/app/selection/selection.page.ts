import { Component, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AlertController, IonSelect, IonicModule, NavController } from '@ionic/angular';
import { ActivatedRoute, Router } from '@angular/router';
import { SharedService } from '../shared.service';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Constants } from '../constants';
import { VehicleBrand } from '../vehicle-brand';
import { VehicleData } from '../vehicle-data';
import { TranslateModule, TranslateService } from '@ngx-translate/core';
import { WebcamImage, WebcamModule} from 'ngx-webcam';
import { WebcamInitError, WebcamUtil} from 'ngx-webcam';
import { Subject, Observable } from 'rxjs';
@Component({
  selector: 'app-selection',
  templateUrl: './selection.page.html',
  styleUrls: ['./selection.page.scss'],
  standalone: true,
  imports: [IonicModule, CommonModule, FormsModule,HttpClientModule,TranslateModule,WebcamModule,],
  providers: [SharedService]
})
export class SelectionPage implements OnInit {
  
  image: WebcamImage | undefined;


  allBrands: VehicleBrand[] = [];
  allModels: VehicleData[] = [];

  translate: TranslateService | undefined;
  storedLang = localStorage.getItem(Constants.LANGUAGE);
  
  


  constructor(private service: SharedService,
              private router: Router,
              private alertController: AlertController,
              private activateRoute: ActivatedRoute,
              private http: HttpClient) { 
                console.log("Hello");
                // this.fetchMessage();
              //   if(this.router.getCurrentNavigation()) {
              //     if(this.router.getCurrentNavigation()?.extras !== undefined) {
              //       if(this.router.getCurrentNavigation()?.extras.state !== undefined) {
              //           if(this.router.getCurrentNavigation()?.extras?.state?.['reset']) {
              //              this['reset']();
              //           }
              //       }
              //     }
              // }
                
              }
  
  

  public showWebcam = false;
  public submitted = false;
  public picTaken = false;
  public allowCameraSwitch = true;
  public multipleWebcamsAvailable = false;
  public deviceId: string | undefined;
  public videoOptions: MediaTrackConstraints = {
    // width: {ideal: 1024},
    // height: {ideal: 576}
  };
  public errors: WebcamInitError[] = [];

  // latest snapshot
  public webcamImage: WebcamImage | undefined;

  // webcam snapshot trigger
  private trigger: Subject<void> = new Subject<void>();
  // switch to next / previous / specific webcam; true/false: forward/backwards, string: deviceId
  private nextWebcam: Subject<boolean|string> = new Subject<boolean|string>();

  public ngOnInit(): void {
    WebcamUtil.getAvailableVideoInputs()
      .then((mediaDevices: MediaDeviceInfo[]) => {
        this.multipleWebcamsAvailable = mediaDevices && mediaDevices.length > 1;
      });
  }

  public triggerSnapshot(): void {
    this.trigger.next();
  }

  public toggleWebcam(): void {
    this.showWebcam = !this.showWebcam;
    this.picTaken = false;

  }

  public handleInitError(error: WebcamInitError): void {
    this.errors.push(error);
  }

  public showNextWebcam(directionOrDeviceId: boolean|string): void {
    // true => move forward through devices
    // false => move backwards through devices
    // string => move to device with given deviceId
    this.nextWebcam.next(directionOrDeviceId);
  }
  message2="";
  message3="";
  public handleImage(webcamImage: WebcamImage): void {
    console.info('received webcam image', webcamImage);
    this.webcamImage = webcamImage;
    console.log(typeof this.webcamImage.imageAsDataUrl);
    this.picTaken = true;
    this.imageUrl = this.webcamImage.imageAsDataUrl;
    // this.service.getLabel(this.webcamImage.imageAsDataUrl);
  }

  public cameraWasSwitched(deviceId: string): void {
    console.log('active device: ' + deviceId);
    this.deviceId = deviceId;
  }

  public get triggerObservable(): Observable<void> {
    return this.trigger.asObservable();
  }

  public get nextWebcamObservable(): Observable<boolean|string> {
    return this.nextWebcam.asObservable();
  }

  public imageUrl: any | undefined;

  public submitImage() {
    this.submitted = true;
    this.http.post('http://127.0.0.1:5000/hello', { imageUrl: this.imageUrl }).subscribe((data: any) => {
      this.message2 = data.message;
      console.log(this.message2);
      this.func();
    });

    
  }
  public func() {
    this.http.post('http://127.0.0.1:5000/hello3', { DiseaseName: this.message2 }).subscribe((data: any) => {
      this.message3 = data.message;
      console.log(this.message3);
    });
  }

  clickExpert() {
    this.router.navigate(['speaking']);
  }

  public imageSrc: string | ArrayBuffer | undefined;

  onFileSelected(event: any) {
    const file: File = <File>event.target.files[0];

    const reader = new FileReader();
    reader.onload = e => {
      this.imageSrc = reader.result as string | ArrayBuffer | undefined;
      console.log(this.imageSrc);
      this.imageUrl = this.imageSrc;
    }


    if (file) {
      reader.readAsDataURL(file);
    }
  }

  // message = '';
  // fetchMessage() {
  //   this.http.get('http://127.0.0.1:5000/hello').subscribe((data: any) => {
  //     this.message = data.message;
  //   });
  // }

}
