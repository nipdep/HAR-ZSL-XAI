import os.path

import numpy as np
from .data import SkeletonFile, NTU120SkeletonFile


def convert_numpy_array_single_person(skel_file: SkeletonFile) -> str:
    SAVE_FOL = "E:\\FYP_Data\\NTU120\\skel\\nturgbd_skeletons_s001_to_s032\\npz_files"

    num_frames, frames = skel_file.load_data()
    coordinate_array = []
    for f in frames:
        joint_array = []
        for j in f["bodies"][0]["joint_details"]:
            joint_array.append([j["x"], j["y"], j["z"]])

        coordinate_array.append(joint_array)

    coordinate_array = np.asarray(coordinate_array, dtype=np.float64)
    file_id = skel_file.filepath.split(os.path.sep)[-1].split(".")[0]

    os.makedirs(SAVE_FOL, exist_ok=True)
    np.savez(os.path.join(SAVE_FOL, f"{file_id}.npz"), coord=coordinate_array)
    return os.path.join(SAVE_FOL, f"{file_id}.npz")


if __name__ == "__main__":
    file = NTU120SkeletonFile(
        "E:\\FYP_Data\\NTU120\skel\\nturgbd_skeletons_s001_to_s032\\nturgb+d_skeletons\\S001C001P001R001A002.skeleton")

    convert_numpy_array_single_person(file)
