import torch
import cv2
import numpy as np
import argparse
import sys
from copy import deepcopy
import tqdm

def mark_detections(cap, model, distance_thres, disp_img = True):
    
    frames_arr = []
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1

    while(cap.isOpened()):
        #cap.set(cv2.CAP_PROP_FPS, 10)
        n_cars = 0
        n_trucks = 0
        n_bikes = 0
        ret, frame = cap.read()
        
        if not ret:
            break
        
        height, width, channels = frame.shape
        
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = model(frame_rgb)
        
        bboxes = preds.xyxy[0].detach().cpu().numpy()
        #print(bboxes)
        frame_cpy = frame_rgb.copy()
        frame_cpy = cv2.line(frame_cpy, (400, distance_thres), (900, distance_thres), (0,0,255), 2)
        for box in bboxes:
            if box[1] >= distance_thres:
                if box[5] == 2:
                    n_cars += 1
                    frame_cpy = cv2.rectangle(frame_cpy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
                elif box[5] == 7:
                    n_trucks += 1
                    frame_cpy = cv2.rectangle(frame_cpy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                elif box[5] == 3:
                    n_bikes += 1
                    frame_cpy = cv2.rectangle(frame_cpy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

        frame_cpy = cv2.putText(frame_cpy, 'Cars: ' + str(n_cars), (40,40), font, fontScale, (255,0,0), 2, cv2.LINE_AA)
        frame_cpy = cv2.putText(frame_cpy, 'Trucks: ' + str(n_trucks), (40,80), font, fontScale, (255,0,0), 2, cv2.LINE_AA)
        frame_cpy = cv2.putText(frame_cpy, 'Bikes: ' + str(n_bikes), (40,120), font, fontScale, (255,0,0), 2, cv2.LINE_AA)
        
        if disp_img:
            cv2.imshow('Traffic Feed', frame_cpy[:,:,::-1])
            if cv2.waitKey(10) == ord('q'):
                break
        else:
            frames_arr.append(frame_cpy[:,:,::-1])

    cap.release()
    cv2.destroyAllWindows()
    
    return frames_arr, (width, height)


def disp_detections(video, conf_thres, iou_thres, distance_thres):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.iou = iou_thres
    model.conf = conf_thres
    
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Current FPS of video:", fps)
    
    _, _ = mark_detections(cap, model, distance_thres, disp_img=True)

def save_video_output(video, conf_thres, iou_thres, save_path, distance_thres):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.iou = iou_thres
    model.conf = conf_thres
    
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Current FPS of video:", fps)
    
    print("Processing inferences on all frames...")
    frames_arr, dims = mark_detections(cap, model, distance_thres, disp_img=False)    
    
    vid_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, dims)
    pbar = tqdm.tqdm(total=len(frames_arr), desc='Frame', position=0)
    for frame in frames_arr:
        vid_out.write(frame)
        pbar.update(1)
    
    vid_out.release()
    print("Detection video processing completed.")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='traffic_short.mp4', help='MP4 file with traffic video sequence')
    parser.add_argument('--conf_thres', type=float, default=0.20, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.15, help='NMS IoU threshold')
    parser.add_argument('--show_detections', action='store_true', help='Display sequence of detections with instance count')
    parser.add_argument('--save_path', type=str, default='', help='MP4 file with detection output')
    parser.add_argument('--distance_thres', default=300, type=int, help='Threshold for distance beyond which bboxes will not be counted')
    opt = parser.parse_args()
    return opt

def main(opt):
    if opt.show_detections:
        opt_dt = deepcopy(opt)
        del opt_dt.show_detections, opt_dt.save_path
        disp_detections(**vars(opt_dt))
    
    if len(opt.save_path):
        opt_dt = deepcopy(opt)
        del opt_dt.show_detections
        save_video_output(**vars(opt_dt))
        
    

if __name__=='__main__':
    opt = parse_opt()
    main(opt)