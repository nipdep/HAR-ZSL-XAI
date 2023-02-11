import os
from random import random

import cv2
import numpy as np


def gen_random_video(model, dataset, epoch, batch_size, model_name, unique_name, epoch_vids, device, mapping_l):
    ind = random.randint(0, len(dataset) - 1)
    batch_sel = random.randint(0, batch_size - 1)

    in_seq, tar_seq, action, vid_size = dataset[ind]
    pred_seq, _ = model(in_seq.repeat(batch_size, 1, 1).to(device))

    os.makedirs(f"{epoch_vids}/{model_name}/{unique_name}/{epoch}", exist_ok=True)
    gen_video(pred_seq[batch_sel].squeeze().cpu().detach().numpy(),
              f"{epoch_vids}/{model_name}/{unique_name}/{epoch}/pred.mp4", int(vid_size[0]), int(vid_size[1]),
              mapping_list=mapping_l)
    gen_video(in_seq.detach().numpy(), f"{epoch_vids}/{model_name}/{unique_name}/{epoch}/true.mp4", int(vid_size[0]),
              int(vid_size[1]), mapping_list=mapping_l)


def gen_skeleton(frame, connections, height, width):
    img_3 = np.zeros([height, width, 3], dtype=np.uint8)
    img_3.fill(255)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # add circles
    for i, coord in enumerate(frame):
        x, y = int(width * coord[0]), int(height * coord[1])
        img_3 = cv2.circle(img_3, center=(x, y), radius=3, color=(255, 0, 0), thickness=6)
        img_3 = cv2.putText(img_3, str(i), (x, y + 5), font,
                            fontScale, color, thickness, cv2.LINE_AA)

    for line in connections:
        i, j = line
        st = frame[i, :]
        start_point = (int(width * st[0]), int(height * st[1]))

        en = frame[j, :]
        end_point = (int(width * en[0]), int(height * en[1]))

        img_3 = cv2.line(img_3, start_point, end_point, color=(0, 0, 0), thickness=3)

    return img_3


def gen_skeletons(frame1, frame2, connections, height, width):
    img_3 = np.zeros([height, width, 3], dtype=np.uint8)
    img_3.fill(255)

    # add circles for the frame-set-1
    for coord in frame1:
        x, y = int(width * coord[0]), int(height * coord[1])
        img_3 = cv2.circle(img_3, center=(x, y), radius=1, color=(255, 0, 0), thickness=6)

    # add circles for the frame-set-2
    for coord in frame2:
        x, y = int(width * coord[0]), int(height * coord[1])
        img_3 = cv2.circle(img_3, center=(x, y), radius=1, color=(0, 255, 0), thickness=6)

    for line in connections:
        i, j = line

        st = frame1[i, :]
        start_point = (int(width * st[0]), int(height * st[1]))
        en = frame1[j, :]
        end_point = (int(width * en[0]), int(height * en[1]))
        img3_ = cv2.line(img_3, start_point, end_point, color=(250, 0, 0), thickness=3)

        st = frame2[i, :]
        start_point = (int(width * st[0]), int(height * st[1]))
        en = frame2[j, :]
        end_point = (int(width * en[0]), int(height * en[1]))
        img3_ = cv2.line(img_3, start_point, end_point, color=(0, 250, 0), thickness=3)

    return img_3


def gen_video(points, save_file, frame_h, frame_w, mapping_list=None):
    # make 3D if points are flatten
    if len(points.shape) == 2:
        fts = points.shape[1]
        x_cds = list(range(0, fts, 3))
        y_cds = list(range(1, fts, 3))
        z_cds = list(range(2, fts, 3))
        points = np.transpose(np.array([points[:, x_cds], points[:, y_cds], points[:, z_cds]]), (1, 2, 0))

    size = (frame_w, frame_h)
    result = cv2.VideoWriter(save_file,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size)

    for __id, frame in enumerate(points):
        skel_image = gen_skeleton(frame, mapping_list, frame_h, frame_w)
        result.write(skel_image)

    result.release()


def gen_cmp_video(points1, points2, save_file, frame_h, frame_w, mapping_list=None):
    # make 3D if points are flatten
    if len(points1.shape) == 2:
        fts = points1.shape[1]
        x_cds = list(range(0, fts, 3))
        y_cds = list(range(1, fts, 3))
        z_cds = list(range(2, fts, 3))
        points1 = np.transpose(np.array([points1[:, x_cds], points1[:, y_cds], points1[:, z_cds]]), (1, 2, 0))

    if len(points2.shape) == 2:
        fts = points2.shape[1]
        x_cds = list(range(0, fts, 3))
        y_cds = list(range(1, fts, 3))
        z_cds = list(range(2, fts, 3))
        points2 = np.transpose(np.array([points2[:, x_cds], points2[:, y_cds], points2[:, z_cds]]), (1, 2, 0))

    size = (frame_w, frame_h)
    result = cv2.VideoWriter(save_file,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size)

    # mapping_list = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (12, 24), (11, 23), (23, 24), (24, 26), (26, 28), (23, 25), (25, 27)]
    if not mapping_list:
        # add lines
        mapping_list = [(0, 1), (1, 3), (3, 5), (0, 2), (2, 4), (0, 6), (1, 7), (6, 7), (6, 8), (7, 9), (8, 10),
                        (9, 11)]
    for __id, frame_1 in enumerate(points1):
        frame_2 = points2[__id]
        skel_image = gen_skeletons(frame_1, frame_2, mapping_list, frame_h, frame_w)
        result.write(skel_image)

    result.release()
