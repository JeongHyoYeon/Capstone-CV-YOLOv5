<div align="center">
  <h1>
    Project
  </h1>
</div>

<div>
  <p>
    <b>GitHub</b> : <a href="https://github.com/JeongHyoYeon">AfterTripGithub</a>
  </p>
  
  <p>
    <b>AfterTrip</b> : 여행 후 그룹별 사진 공유의 불편함을 해결하기 위해 얼굴인식AI와 객체인식AI를 이용해 폴더를 분류하는 사진 공유용 모바일 웹
  </p>
</div>

<br>
<div align="center">
  <h1>
    [Yolov5] Scene Classification
  </h1>
</div>
<img src="https://github.com/JeongHyoYeon/Capstone-CV-YOLOv5/assets/90602936/7ece8b56-71fd-44ba-851a-5603234966ad">
<p>
  <b>dataset</b> : <a href="https://universe.roboflow.com/m3-ytsk5/m3finalclass"> link </a>
</p>
<p>
  우선 풍경tag를 붙이기 위해 자연, 해안, 숲, 도심 등 총 8개의 class로 구성된 custom dataset으로 
  yolov5를 200epoch train시켰습니다.
</p>

<p>
  이렇게 fine tuning한 모델로 scene classification을 진행합니다.
</p>

<p>
  그렇게 분류된 top5 class중에 probability가 0.6이상인 class의 tag를 붙이게 됩니다.
</p>


<br>
<div align="center">
  <h1>
    [Yolov5] Object Detection
  </h1>
</div>
<img src="https://github.com/JeongHyoYeon/Capstone-CV-YOLOv5/assets/90602936/2e6e5eab-9983-4f2a-a25b-0028d530b51d">
<p>
  객체tag를 붙이기 위해
  80개의 class로 구성된 coco dataset으로 pre-trained된 
  yolov5 m checkpoint를 이용하여 object detection을 진행합니다.
</p>
<p>
  정확도를 나타내는 mean average precision인 mAP 가 44.5로 좋은 성능을 보여줍니다.
</p>

<img src="https://github.com/JeongHyoYeon/Capstone-CV-YOLOv5/assets/90602936/bb565feb-029a-4cce-9bef-d490d7c29d8a">

이렇게 scene classification과 object detection을 수행해 최대 88개의 폴더를 생성하게 됩니다.
그 결과로 생성된 폴더 내부를 살펴보면, 식탁 폴더엔 식탁 위 사진들이, 컵 폴더엔 컵이 들어간 사진들이, 비행기 폴더엔 공항에서 찍은 사진이 잘 들어간 것을 확인할 수 있습니다.

이러한 객체 인식 AI는 사진 100장에 1분 미만의 시간이 소요됩니다. (Tesla T4 GPU 이용)

<br>
<div align="center">
  <h1>
    실행 방법
  </h1>
</div>

```
from main import *

images = run_yolov5(images)
```
```
pretrained yolov5 모델로
(1) scene classification 수행해 [scene tag]를
(2) obejct detectoin 수행해 [object tag]를
infer하는 함수.

Args:
        images (list) : [{
                        "id" (int) : DB에서 이미지 id
                        "url" (str) : S3에서 생성한 url
                      }]
Returns:
        images (list) : [{
                        "id" (int) : DB에서 이미지 id
                        "url" (str) : S3에서 생성한 url
                        "yolo_tag" (str list) : (1) pretrained yolov5로 scene classification [scene tag]
                                                (2) pretrained yolov5로 obejct detectoin [object tag]
                        "yolo_detail" (list) :  (1)의 경우, scene classification했을때 해당 tag에 대한 probability
                                                (2)의 경우, object detection 했을때 해당 tag가 몇번 등장했는지
                      }]
```

<br>
<div align="center">
  <h1>
    참고자료
  </h1>
</div>
<p> <b>YOLOv5</b> : <a href="https://github.com/ultralytics/yolov5">github</a> </p>

<br>
<div align="center">
  <h1>
    기술블로그
  </h1>
</div>
<p> <b>기술블로그</b> :  <a href="https://deardus00.tistory.com/22">[2023-1 졸업프로젝트] FaceRecognition & YOLOv5</a> </p>
