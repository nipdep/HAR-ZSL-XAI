import os

from tqdm import tqdm

from abc import ABC, abstractmethod

from typing import List


class SkeletonFile(ABC):

    @abstractmethod
    def load_data(self):
        pass


class NTU120SkeletonFile(SkeletonFile):
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        try:
            with open(self.filepath, "r") as f0:
                num_frame = int(f0.readline())
                frames = []
                for frame_no in range(num_frame):
                    frame_d = {"body_count": int(f0.readline()), "bodies": []}
                    for n_body in range(frame_d["body_count"]):
                        val = f0.readline().split()
                        d_line = list(map(float, val))
                        body_details = {"bodyID": val[0],
                                        "clipedEdges": int(d_line[1]),
                                        "handLeftConfidence": int(d_line[2]),
                                        "handLeftState": int(d_line[3]),
                                        "handRightConfidence": int(d_line[4]),
                                        "handRightState": int(d_line[5]),
                                        "isResticted": int(d_line[6]),
                                        "leanX": d_line[7],
                                        "leanY": d_line[8],
                                        "trackingState": int(d_line[9]),
                                        "joint_details": []
                                        }

                        num_joints = int(f0.readline())
                        for n_j in range(num_joints):
                            d_line = list(map(float, f0.readline().split()))
                            joint_details = {
                                "x": d_line[0],
                                "y": d_line[1],
                                "z": d_line[2],
                                "depthX": d_line[3],
                                "depthY": d_line[4],
                                "colorX": d_line[5],
                                "colorY": d_line[6],
                                "orientationW": d_line[7],
                                "orientationX": d_line[8],
                                "orientationY": d_line[9],
                                "orientationZ": d_line[10],
                                "trackingState": int(d_line[11])
                            }

                            body_details["joint_details"].append(joint_details)

                        frame_d["bodies"].append(body_details)
                    frames.append(frame_d)
            return num_frame, frames
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found:- {self.filepath}")


class SkeletonFileBuilder(object):
    def __init__(self, folder_path: list = None, ignore_files: list = None, file_names: set = None):
        if not folder_path:
            folder_path = "E:\\FYP_Data\\NTU120\\skel\\nturgbd_skeletons_s001_to_s032\\nturgb+d_skeletons"

        if not ignore_files:
            ignore_files = ("E:\\FYP_Data\\NTU120\\skel\\NTU_RGBD_samples_with_missing_skeletons.txt",
                            "E:\\FYP_Data\\NTU120\\skel\\NTU_RGBD120_samples_with_missing_skeletons.txt")

        self.folder_path = folder_path
        self.ignore_files = ignore_files
        self.file_names = file_names
        self.ignore_files = self.create_ignore_files()
        self.files2build = list(self.get_read_files())

    def __len__(self):
        return len(self.files2build)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.files2build):
            x = NTU120SkeletonFile(self.files2build[self.i])
            self.i += 1
            return x
        else:
            raise StopIteration

    def create_ignore_files(self) -> set:
        file_names = set()
        for __f in self.ignore_files:
            with open(__f, "r") as f0:
                file_names = file_names.union(set(f0.read().split("\n")))

        return file_names

    def get_read_files(self) -> set:
        if not self.file_names:
            self.file_names = set([os.path.join(self.folder_path, x) for x in set(os.listdir(self.folder_path))])

        self.ignore_files = set([os.path.join(self.folder_path, x) for x in self.ignore_files])
        return self.file_names.difference(self.ignore_files)

    def build(self) -> List[SkeletonFile]:
        data = []
        for __f in tqdm(self.files2build):
            data.append(NTU120SkeletonFile(__f))

        return data


class FileListIterator(object):
    def __init__(self, numpy_files):
        self.numpy_files = numpy_files

    def __len__(self):
        return len(self.numpy_files)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.numpy_files):
            x = self.numpy_files[self.i]
            self.i += 1
            return x
        else:
            raise StopIteration


if __name__ == "__main__":
    # file = NTU120SkeletonFile(
    #    "E:\\FYP_Data\\NTU120\\skel\\nturgbd_skeletons_s001_to_s017\\nturgb+d_skeletons\\S001C001P001R001A005.skeleton")

    # file.load_data()
    # print(file.frames)

    # builder = SkeletonFileBuilder()
    # files = builder.build()
    # iterator = iter(builder)
    # for x in iterator:
    #    print(x)
    #    break

    with open("E:\\FYP_Data\\NTU120\\files_containing_only_1_person.txt", "r") as f0:
        file_list = f0.read().split("\n")

    builder = SkeletonFileBuilder(file_names=set(file_list))
    file_iterator = iter(builder)
