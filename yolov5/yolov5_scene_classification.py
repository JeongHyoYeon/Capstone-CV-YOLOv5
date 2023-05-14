import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
YOLO_PATH = FILE.parents[0]
YOLO_CLONE_PATH = os.path.join(YOLO_PATH, "yolov5/")

if str(YOLO_CLONE_PATH) not in sys.path:
    sys.path.append(str(YOLO_CLONE_PATH))  # add YOLO_CLONE_PATH to PATH

YOLO_PATH = Path(YOLO_PATH)
YOLO_CLONE_PATH = Path(YOLO_CLONE_PATH)

from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, print_args, strip_optimizer)
from utils.plots import Annotator
from utils.torch_utils import select_device, smart_inference_mode


# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

def run_yolov5_scene(
        images,
        base_path,
        yolov5_path,
        classification_threshold=0.4,  # probability threshold

        weights=YOLO_CLONE_PATH / 'yolov5s-cls.pt',  # model.pt path(s)
        source=YOLO_CLONE_PATH / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        nosave=False,  # do not save images/videos

        data=YOLO_CLONE_PATH / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(224, 224),  # inference size (height, width)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=YOLO_CLONE_PATH / 'runs/predict-cls',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    """
    yolov5 classification을 run시켜서 images에 scene tag를 추가하는 함수로
    Ultralytics의 yolov5/classify/predict.py를 변형한 함수.

    Args:
      images (list) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                    }]

    Returns:
      images (list) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                      "yolo_tag" :  (1) pretrained yolov5로 scene classification [scene tag]
                      "yolo_scene_prob" : scene classification했을 때의 probability
                    }]
    """
    # check
    check_requirements(exclude=('tensorboard', 'thop'))

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    if not nosave:
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    #model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    #model = torch.hub.load('path/to/yolov5', 'custom', path='path/to/best.pt', source='local')
    model = torch.hub.load( YOLO_CLONE_PATH, 
                           "custom", 
                           path = os.path.join(YOLO_PATH, "checkpoint/yolov5_scene_best.pt"),  
                           source = "local" )
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    images_idx = 0  # (YeonWoo)

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.Tensor(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            results = model(im)

        # Post-process
        with dt[2]:
            pred = F.softmax(results, dim=1)  # probabilities

        # Process predictions
        for i, prob in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt

            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, example=str(names), pil=True)

            # Print results
            top5i = prob.argsort(0, descending=True)[:5].tolist()  # top 5 indices
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "

            # (YeonWoo) tag_temp, prob_temp 구하기
            tag_temp = []
            prob_temp = []

            for j in top5i:
                if (prob[j] >= classification_threshold):
                    tag_temp.append(names[j])
                    prob_temp.append(round(float(prob[j]), 3))  # tensor -> float -> 반올림

            # (YeonWoo) images에 tag & prob 추가
            # (images id 구하는 코드. 필요없었음. images idx만 구하면됨.)
            # file_name = os.path.split(path)[1] # 전체 경로에서 file name만 parsing
            # images_idx = int(file_name.split('.')[0]) # 45.jpg에서 "45" parsing -> int로 변경경
            images[images_idx]["yolo_tag"] = tag_temp
            images[images_idx]["yolo_detail"] = prob_temp
            images_idx = images_idx + 1

            '''
            # Write results
            text = '\n'.join(f'{prob[j]:.2f} {names[j]}' for j in top5i)
            if save_img or view_img:  # Add bbox to image
                annotator.text((32, 32), text, txt_color=(255, 255, 255))
            if save_txt:  # Write to file
                with open(f'{txt_path}.txt', 'a') as f:
                    f.write(text + '\n')

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
              '''

        # Print time (inference-only)
        LOGGER.info(f'{s}{dt[1].dt * 1E3:.1f}ms')

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    return images
