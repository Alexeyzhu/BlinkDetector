# Blink Detector

Test task for Summer Internship 2020 in EORA.
 
System that detects and counts eye blink of a person in real-time mode and shows alert if eyes closed more than 2 seconds 

## How to run


#### Prerequisites
* [Python](https://www.python.org/downloads/)  3.6 and higher
* [Git](https://git-scm.com/downloads)

#### Clone project and setup

* Clone repository

```bash
   git clone https://github.com/Alexeyzhu/BlinkDetector.git
```

* Install **requirements.txt**
```bash
   pip install -r requirements.txt
```

You can face problems with installing dlib on Windows. In this case, try to install [Cmake](https://cmake.org/install/) firstly

#### Run program

You can run program with 2 kind of source: video file or a stream from a web camera. 
In folder you can find 2 video samples for testing: *demo.mp4* and *promo.mp4*

* To run system with video file as a source:

```bash
   python blink-detector.py --shape-predictor shape_predictor_68_face_landmarks.dat --video demo.mp4
```
or
```bash
   python blink-detector.py --shape-predictor shape_predictor_68_face_landmarks.dat --video promo.mp4
```

* To run system with web camera as a source:

```bash
   python blink-detector.py --shape-predictor shape_predictor_68_face_landmarks.dat
```