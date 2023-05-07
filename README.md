# [Yolov5] Scene Classification & Object Detection
Yolov5로 (1) Scene Classification (2) Object Detection 수행해 scene tag & object tag infer

## (1) Scene Classification
### ✅ dataset
- 링크
- class 설명

### ✅ train 
- 구글 코랩 이용
- <h4>.ipynb</h4> : 100epoch ...

## (2) Scene Classification
### ✅ dataset
- 링크
- class 설명

### ✅ train _ 첫번째 시도
- 구글 코랩 이용
- <h4>.ipynb</h4> : 20epoch ...

### ✅ train _ 두번째 시도 (use this)
- 텐센트 클라우드 서버 이용

## (1)(2)를 이용해 tag infer
- <h4>[방법1] ipynb로 실행 -> yolov5.ipynb</h4> 
: 설명

- <h4>[방법2] 모듈화된 .py 파일 실행 -> yolov5폴더</h4> 
: yolov5폴더에 필요한 모든 파일 들어 있음.
from yolov5.run_yolov5 import *
images = run_yolov5(images)
으로 실행가능.

## 이용한 코드
- [yolov5 github](https://github.com/ultralytics/yolov5)

