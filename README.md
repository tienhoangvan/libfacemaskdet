I use libfacedetection as base code. Thanks Dr. ShiqiYu, Prof.    
Compared with libfacedetection, I remove the face landmark and add the facemask. Although it has low accuracy, it can compile the source code under any platform with a C++ compiler, and there are some method that we can improve the mAP:    
- [ ] more training data
- [ ] change the net 
- [ ] hard minging
- [ ] FPN
- [ ] Focal Loss


Some popular ligthweight face detections can see: 
- libfacedetection
- Ultra-Light-Fast-Generic-Face-Detector-1MB
- A-Light-and-Fast-Face-Detector-for-Edge-Devices
- CenterFace
- DBFace
- RetinaFace MobileNet0.25


