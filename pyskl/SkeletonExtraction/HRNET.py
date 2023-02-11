import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

import decord
import numpy as np
import os
import os.path as osp

import glob

try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    import mmpose
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

default_mmdet_root = "D:/FYP/HAR-ZSL-XAI/pyskl/mmdetection"
default_mmpose_root = "D:/FYP/HAR-ZSL-XAI/pyskl/mmpose"
default_det_config = (
    f'{default_mmdet_root}/configs/faster_rcnn/'
    'faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py')
default_det_ckpt = (
    'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
    'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth')
default_pose_config = (
    f'{default_mmpose_root}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
    'coco/hrnet_w32_coco_256x192.py')
default_pose_ckpt = (
    'https://download.openmmlab.com/mmpose/top_down/hrnet/'
    'hrnet_w32_coco_256x192-c78dce93_20200708.pth')

det_model = init_detector(default_det_config, default_det_ckpt, 'cuda')
pose_model = init_pose_model(default_pose_config, default_pose_ckpt, 'cuda')


def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]


def detection_inference(model, frames):
    results = []
    for frame in frames:
        result = inference_detector(model, frame)
        results.append(result)
    return results


def pose_inference(model, frames, det_results):
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frames, det_results)):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
    return kp


def gen_video_list(file_path: str):
    annotations = []
    for __path in glob.glob(os.path.join(file_path, "*", "*.mp4")):
        p_parts = __path.split(os.path.sep)
        cls_name = p_parts[-2]

        annotations.append(dict(
            frame_dir=osp.basename(__path).split('.')[0],
            file_path=__path,
            filename=p_parts[-1],
            label=cls_name
        ))

    return annotations





def HRNetExtraction(anno, det_score_thr=0.7, det_area_thr=1600):
    frames = extract_frame(anno['file_path'])
    det_results = detection_inference(det_model, frames)
    # * Get detection results for human
    det_results = [x[0] for x in det_results]
    for i, res in enumerate(det_results):
        # * filter boxes with small scores
        res = res[res[:, 4] >= det_score_thr]
        # * filter boxes with small areas
        box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
        assert np.all(box_areas >= 0)
        res = res[box_areas >= det_area_thr]
        det_results[i] = res

    pose_results = pose_inference(pose_model, frames, det_results)
    shape = frames[0].shape[:2]
    anno['img_shape'] = anno['original_shape'] = shape
    anno['total_frames'] = len(frames)
    anno['num_person_raw'] = pose_results.shape[0]
    anno['keypoint'] = pose_results[..., :2].astype(np.float16)
    anno['keypoint_score'] = pose_results[..., 2].astype(np.float16)
    return anno


def inference_detectorV2(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results
