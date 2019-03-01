from darknet_py import detect
from darknet_py import load_net_custom, load_meta
from moviepy.editor import VideoFileClip
from motion_profile import MPGenerator
from skimage import draw, io
import os
from skimage import draw
import numpy as np
import time
configPath = './cfg/yolov3.cfg'
weightPath = 'yolov3.weights'
metaPath= './data/coco.data'
thresh = 0.25

if __name__ == '__main__':
    video_name = 'data/firstsecond/FILE0033.MOV'
    video = VideoFileClip(video_name)
    low_bound = 360
    up_bound = 600
    mpGen = MPGenerator(video_name)
    # todo: motion profile is static and save for env and save as '0033_mp.jpg'
    mp_img = mpGen.generate(low_bound, up_bound, '0033_mp.jpg')
    blank_img = np.zeros(mp_img.shape, dtype=np.uint8)

    netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
    metaMain = load_meta(metaPath.encode("ascii"))
    altNames = None
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    time_idx = 0
    pre_detect = [] # middle x pos of detected objects in last frame
    sample_gap = 15
    for frame in video.iter_frames():
        if time_idx % sample_gap != 0:
            time_idx += 1
            continue
        print('Object detected at frame: '+ str(time_idx) + ':')
        # Get yolo object prediction result
        t1 = time.time()
        res = detect(netMain, metaMain, frame, thresh, altNames=altNames)
        t2 = time.time()
        print('Cost Time: %f' % (t2-t1))
        showImage = False
        # todo: show detection for every img
        if showImage:
            try:
                from skimage import io, draw
                import numpy as np

                image = frame
                print("*** " + str(len(res)) + " Results, color coded by confidence ***")
                imcaption = []
                detections = res
                for detection in detections:
                    label = detection[0]
                    if label != 'person':
                        continue
                    confidence = detection[1]
                    pstring = label + ": " + str(np.rint(100 * confidence)) + "%"
                    imcaption.append(pstring)
                    print(pstring)
                    bounds = detection[2]
                    shape = image.shape
                    # x = shape[1]
                    # xExtent = int(x * bounds[2] / 100)
                    # y = shape[0]
                    # yExtent = int(y * bounds[3] / 100)
                    yExtent = int(bounds[3])
                    xEntent = int(bounds[2])
                    # Coordinates are around the center
                    xCoord = int(bounds[0] - bounds[2] / 2)
                    yCoord = int(bounds[1] - bounds[3] / 2)
                    boundingBox = [
                        [xCoord, yCoord],
                        [xCoord, yCoord + yExtent],
                        [xCoord + xEntent, yCoord + yExtent],
                        [xCoord + xEntent, yCoord]
                    ]
                    # Wiggle it around to make a 3px border
                    rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox],
                                                    shape=shape)
                    rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox],
                                                      shape=shape)
                    rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox],
                                                      shape=shape)
                    rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox],
                                                      shape=shape)
                    rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox],
                                                      shape=shape)
                    boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                    draw.set_color(image, (rr, cc), boxColor, alpha=0.8)
                    draw.set_color(image, (rr2, cc2), boxColor, alpha=0.8)
                    draw.set_color(image, (rr3, cc3), boxColor, alpha=0.8)
                    draw.set_color(image, (rr4, cc4), boxColor, alpha=0.8)
                    draw.set_color(image, (rr5, cc5), boxColor, alpha=0.8)

                io.imshow(image)
                io.show()
                detections = {
                    "detections": detections,
                    "image": image,
                    "caption": "\n<br/>".join(imcaption)
                }
            except Exception as e:
                print("Unable to show image: " + str(e))

        curr_detect = []
        same_obj_thresh = 50 # max distance of middle x between two frames
        for obj in res:
            tag, prob, pos = obj[0], obj[1], obj[2]

            # Paint the trajectory of objects which are in the sampling area of motion profile
            # todo: log 存取 所有detection数据
            center_x, center_y, w, h = pos[0], pos[1], pos[2], pos[3]
            try:
                if low_bound < center_y < up_bound and prob > 0.5 and time_idx < mp_img.shape[0]:
                    curr_detect.append([center_x, w, tag])
                    if pre_detect is not None:
                        pre_x, pre_w, pre_tag = pre_detect[0][0], pre_detect[0][1], pre_detect[0][2]
                        if abs(center_x - pre_x) < same_obj_thresh and tag == pre_tag:
                            w_offset = (w - pre_w) / same_obj_thresh
                            x_offset = (center_x - pre_x) / same_obj_thresh
                            for add_idx in range(1, same_obj_thresh):
                                add_center = int(add_idx * x_offset + pre_x)
                                add_w = int(add_idx * w_offset + pre_w)
                                add_x1 = pre_x - 0.5 * add_w if pre_x - 0.5 * add_w > 0 else 0
                                add_x2 = pre_x + 0.5 * add_w if pre_x + 0.5 * add_w < mp_img.shape[1] else mp_img.shape[1]
                                add_time_idx = time_idx - sample_gap + add_idx
                                _rr, _cc = draw.line(add_time_idx, int(add_x1), add_time_idx, int(add_x2))
                                if pre_tag == 'car':
                                    blank_img[_rr, _cc, 0] = 255
                                elif pre_tag == 'person':
                                    blank_img[_rr, _cc, 1] = 255
                            pre_detect.pop(0)
                        elif abs(center_x - pre_x) >= same_obj_thresh and center_x > pre_x:
                            pre_detect.pop(0)

                    x1 = center_x - 0.5 * w if center_x - 0.5 * w > 0 else 0
                    x2 = center_x + 0.5 * w if center_x + 0.5 * w < mp_img.shape[1] else mp_img.shape[1]
                    rr, cc = draw.line(time_idx, int(x1), time_idx, int(x2))
                    # tag lei
                    if tag == 'car':
                        blank_img[rr, cc, 0] = 255
                    elif tag == 'person':
                        blank_img[rr, cc, 1] = 255
            except IndexError as e:
                print(e)

        pre_detect = curr_detect
        time_idx += 1

    io.imsave('file0033_yolo_trace.jpg', blank_img)

    io.imshow(blank_img)
    io.show()
