# 이미지 저장
import urllib.request
import os
import shutil
# yolo 기본
#import utils
#display = utils.notebook_init()  # checks
# yolo Classification, ObjectDetection
from yolov5.yolov5.yolov5_scene_classification import run_yolov5_scene
from yolov5.yolov5.yolov5_object_detection import run_yolov5_object

def download_images(images, images_folder_path, re_download):
  # 폴더 생성
  if os.path.exists(images_folder_path):
    if re_download:
      shutil.rmtree(images_folder_path)
      os.mkdir(images_folder_path)
  else :
    os.mkdir(images_folder_path)

  # 이미지 저장
  if re_download:
    for idx, image in enumerate(images):
      urllib.request.urlretrieve(image["url"], images_folder_path + str(image["id"]) + ".jpg")

def print_images_dict(images):
    for i, img in enumerate(images):
        print("idx = ", i)
        print(img)
        print(" ")

def run_yolov5(images):
    """
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
    """
    print("==========START YOLOV5==========\n")

    re_download = True

    base_path = "/content/drive/MyDrive/Capstone_Yolov5_Test/yolov5/"
    yolov5_path = base_path + "yolov5/"
    images_folder_path = base_path + "images/"

    # [STEP 0. url로 이미지 다운받아 images 폴더에 저장]
    print("\n\n")
    print("Step0. download images at folder\n")
    download_images(images, images_folder_path, re_download)

    # [STEP 1. yolov5 : scene classification [scene tag] ]
    print("\n\n")
    print("Step1. scene classification\n")
    images = run_yolov5_scene(
        images,
        base_path,
        yolov5_path,
        classification_threshold=0.4,
        weights="yolov5/checkpoint/yolov5_scene_best.pt",  # model.pt path(s)
        source="yolov5/images/*.jpg",  # file/dir/URL/glob/screen/0(webcam)
        nosave=True,  # do not save images/videos
    )

    print_images_dict(images)

    # [STEP 2. yolov5 : object detection [object tag] ]
    print("\n\n")
    print("Step2. object detection\n")
    images = run_yolov5_object(
        images,
        base_path,
        yolov5_path,
        weights="yolov5/checkpoint/yolov5_object_best.pt",  # model.pt path(s)
        source="yolov5/images/*.jpg",  # file/dir/URL/glob/screen/0(webcam)
        nosave=True,  # do not save images/videos
    )

    print_images_dict(images)

    print("==========END YOLOV5==========\n")