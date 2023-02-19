import os
import numpy as np
import random


def normalize_array(arr, video_size):
    arr[:, :, 0] = arr[:, :, 0] / video_size[1]
    arr[:, :, 1] = arr[:, :, 1] / video_size[0]


def normalize_arrays(arrays, video_size):
    for arr in arrays:
        normalize_array(arr, video_size)

    return arrays


def break_into_time_frames(coordinates, time_window=50, stride=40):
    shape = coordinates.shape

    # [x,y,z] is the format
    if shape[0] < time_window:
        return None

    frame_points = list(range(0, shape[0] - time_window, stride))
    arrays = []
    for i in frame_points:
        arrays.append(coordinates[i:i + time_window])
    return arrays


def random_break_into_time_frames(coordinates, time_window=60, select=50, stride=40):
    shape = coordinates.shape

    # [x,y,z] is the format
    if shape[0] < time_window:
        # print(shape[0])
        return []

    frame_points = list(range(0, shape[0] - time_window, stride))
    arrays = []
    for i in frame_points:
        frame_no = sorted(random.sample(range(i, i + time_window), select))
        arrays.append(coordinates[frame_no])
    return arrays


def save_arrays(data_dir, file_name, class_name, arrays, vid_size):
    for __id, each_array in enumerate(arrays):
        np.savez(os.path.join(data_dir, f"{class_name}_cls_{file_name}_{__id}.npz"), coords=each_array,
                 video_size=vid_size)


def classname_id(class_name_list):
    id2classname = {k: v for k, v in zip(list(range(len(class_name_list))), class_name_list)}
    classname2id = {v: k for k, v in id2classname.items()}
    return id2classname, classname2id


def break_from_middle(coords, time_window=50):
    coords_mid_point = coords.shape[0] // 2
    time_window_half = time_window // 2

    if coords_mid_point - time_window_half < 0:
        return []
    else:
        return [coords[coords_mid_point - time_window_half:coords_mid_point + time_window_half, :, :]]


def split_array_from_builder(refined_data: str, shapes, each_file):
    file_id = each_file.filepath.split(os.path.sep)[-1].split(".")[0]
    num_frame, body_data = each_file.load_data()
    orig_vid_size = shapes[file_id]
    for frame_data in body_data:
        if frame_data["body_count"] != 1:
            return {
                "status": False,
                "Reason": "Body Count is higher than 1",
                "file_id": file_id
            }
    skel_data = []
    for frame_data in body_data:
        frame_jd = []
        for jd in frame_data["bodies"][0]["joint_details"]:
            x = jd["colorX"] / orig_vid_size[1]
            y = jd["colorY"] / orig_vid_size[0]

            frame_jd.append([x, y])

        skel_data.append(frame_jd)

    skel_data = np.asarray(skel_data)

    if (skel_data[:, :, 0] < 0).sum() > 0 or (skel_data[:, :, 1] < 0).sum() > 0:
        return {
            "status": False,
            "Reason": "Negative Values",
            "file_id": file_id
        }

    f_point = break_from_middle(skel_data, time_window=50)

    class_n = str(int(file_id.split("A")[-1]))
    file_id = file_id

    if len(f_point) > 0:
        save_arrays(refined_data, file_id, class_n, f_point, orig_vid_size)
        return {
            "status": True,
            "Reason": None,
            "file_id": file_id
        }
    else:
        return {
            "status": False,
            "Reason": "Array Empty",
            "file_id": file_id
        }


def split_array_main(id2shapes: dict, refined_data: str, each_file: dict):
    try:
        coords, vid_size = each_file["keypoint"][0], id2shapes[each_file["frame_dir"]]
        # if (coords[:, :, 0] > 1).sum() > 0 or (coords[:, :, 1] > 1).sum() > 0:
        #    return each_file["frame_dir"]

        if (coords[:, :, 0] < 0).sum() > 0 or (coords[:, :, 1] < 0).sum() > 0:
            return "Negative Values", each_file["frame_dir"]

        f_point = random_break_into_time_frames(coords, time_window=70)

        # print(path_parts)
        class_n = each_file["label"]
        file_id = each_file["frame_dir"]

        if len(f_point) > 0:
            f_point = normalize_arrays(f_point, vid_size)
            # gen_video(f_point[0], "check_vid.mp4", 400, 400)
            save_arrays(refined_data, file_id, class_n, f_point, vid_size)
            return each_file["frame_dir"]

    except IndexError:
        return "Index Error", each_file["frame_dir"]
