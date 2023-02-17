import os
import json
import random
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Adam

from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


class PAMAP2Reader(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.readPamap2()

    def readFile(self, file_path, cols):
        all_data = {"data": {}, "target": {}, 'collection': []}
        prev_action = -1
        starting = True
        # action_seq = []
        action_ID = 0

        for l in open(file_path).readlines():
            s = l.strip().split()
            if s[1] != "0":
                if (prev_action != int(s[1])):
                    if not (starting):
                        df = pd.DataFrame(action_seq)
                        intep_df = df.interpolate(method='linear', limit_direction='both', axis=0)
                        intep_data = intep_df.values
                        all_data['data'][action_ID] = np.array(intep_data)
                        all_data['target'][action_ID] = prev_action
                        action_ID += 1
                    action_seq = []
                else:
                    starting = False
                intm_data = s[3:]
                data_seq = np.array(intm_data)[cols].astype(np.float16)
                # data_seq[np.isnan(data_seq)] = 0
                action_seq.append(data_seq)
                prev_action = int(s[1])
                # print(prev_action)
                all_data['collection'].append(data_seq)
        else:
            if len(action_seq) > 1:
                df = pd.DataFrame(action_seq)
                intep_df = df.interpolate(method='linear', limit_direction='backward', axis=0)
                intep_data = intep_df.values
                all_data['data'][action_ID] = np.array(intep_data)
                all_data['target'][action_ID] = prev_action
        return all_data

    def readPamap2Files(self, filelist, cols, labelToId):
        data = []
        labels = []
        collection = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            fpath = os.path.join(self.root_path, filename)
            file_data = self.readFile(fpath, cols)
            data.extend(list(file_data['data'].values()))
            labels.extend(list(file_data['target'].values()))
            collection.extend(file_data['collection'])
        return np.asarray(data), np.asarray(labels, dtype=int), np.array(collection)

    def readPamap2(self):
        files = ['subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat', 'subject105.dat',
                 'subject106.dat', 'subject107.dat', 'subject108.dat', 'subject109.dat', 'subject110.dat',
                 'subject111.dat', 'subject112.dat', 'subject113.dat', 'subject114.dat']

        label_map = [
            # (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'Nordic walking'),
            (9, 'watching TV'),
            (10, 'computer work'),
            (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            (18, 'folding laundry'),
            (19, 'house cleaning'),
            (20, 'playing soccer'),
            (24, 'rope jumping')
        ]
        labelToId = {x[0]: i for i, x in enumerate(label_map)}
        # print "label2id=",labelToId
        idToLabel = [x[1] for x in label_map]
        # print "id2label=",idToLabel
        cols = [1, 2, 3, 7, 8, 9, 10, 11, 12, 17, 18, 19, 23, 24, 25, 26, 27, 28, 33, 34, 35, 39, 40, 41, 42, 43, 44]
        self.cols = cols
        # print "cols",cols
        self.data, self.targets, self.all_data = self.readPamap2Files(files, cols, labelToId)
        # self.data = self.data[:, :, cols]
        # print(self.data)
        # nan_perc = np.isnan(self.data).astype(int).mean()
        # print("null value percentage ", nan_perc)
        # f = lambda x: labelToId[x]
        self.targets = np.array([labelToId[i] for i in list(self.targets)])
        self.label_map = label_map
        self.idToLabel = idToLabel
        # return data, idToLabel

    def resample(self, signal, freq=10):
        step_size = int(100 / freq)
        seq_len, _ = signal.shape
        resample_indx = np.arange(0, seq_len, step_size)
        resampled_sig = signal[resample_indx, :]
        return resampled_sig

    def windowing(self, signal, window_len, overlap):
        seq_len = int(window_len * 100)  # 100Hz compensation
        overlap_len = int(overlap * 100)  # 100Hz
        l, _ = signal.shape
        if l > seq_len:
            windowing_points = np.arange(start=0, stop=l - seq_len, step=seq_len - overlap_len, dtype=int)[:-1]

            windows = [signal[p:p + seq_len, :] for p in windowing_points]
        else:
            windows = []
        return windows

    def normalize(self, signal):
        feat_max = signal.astype(np.float64).max(axis=0)
        feat_min = signal.astype(np.float64).min(axis=0)
        norm_feat = (signal - feat_min) / (feat_max - feat_min + 10e-6)
        return norm_feat

    def resampling(self, data, targets, window_size, window_overlap, resample_freq):
        assert len(data) == len(targets), "# action data & # action labels are not matching"
        all_data, all_ids, all_labels = [], [], []
        for i, d in enumerate(data):
            # print(">>>>>>>>>>>>>>>  ", np.isnan(d).mean())
            label = targets[i]
            d = self.normalize(d)
            windows = self.windowing(d, window_size, window_overlap)
            for w in windows:
                # print(np.isnan(w).mean(), label, i)
                resample_sig = self.resample(w, resample_freq)
                # print(np.isnan(resample_sig).mean(), label, i)
                all_data.append(resample_sig)
                all_ids.append(i + 1)
                all_labels.append(label)

        return all_data, all_ids, all_labels

    def generate(self, unseen_classes, window_size=5.21, window_overlap=1, resample_freq=10, smoothing=False,
                 normalize=False, seen_ratio=0.2, unseen_ratio=0.8):

        def smooth(x, window_len=11, window='hanning'):
            if x.ndim != 1:
                raise Exception('smooth only accepts 1 dimension arrays.')
            if x.size < window_len:
                raise Exception("Input vector needs to be bigger than window size.")
            if window_len < 3:
                return x
            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise Exception("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
            s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
            if window == 'flat':  # moving average
                w = np.ones(window_len, 'd')
            else:
                w = eval('np.' + window + '(window_len)')
            y = np.convolve(w / w.sum(), s, mode='same')
            return y[window_len:-window_len + 1]

        # assert all([i in list(self.label_map.keys()) for i in unseen_classes]), "Unknown Class label!"
        seen_classes = [i for i in range(len(self.idToLabel)) if i not in unseen_classes]
        unseen_mask = np.in1d(self.targets, unseen_classes)

        # build seen dataset
        seen_data = self.data[np.invert(unseen_mask)]
        seen_targets = self.targets[np.invert(unseen_mask)]

        # build unseen dataset
        unseen_data = self.data[unseen_mask]
        unseen_targets = self.targets[unseen_mask]

        # resampling seen and unseen datasets
        seen_data, seen_ids, seen_targets = self.resampling(seen_data, seen_targets, window_size, window_overlap,
                                                            resample_freq)
        unseen_data, unseen_ids, unseen_targets = self.resampling(unseen_data, unseen_targets, window_size,
                                                                  window_overlap, resample_freq)

        seen_data, seen_targets = np.array(seen_data), np.array(seen_targets)
        unseen_data, unseen_targets = np.array(unseen_data), np.array(unseen_targets)

        if normalize:
            a, b, nft = seen_data.shape
            intm_sdata = seen_data.reshape((-1, nft))
            intm_udata = unseen_data.reshape((-1, nft))

            scaler = MinMaxScaler()
            norm_sdata = scaler.fit_transform(intm_sdata)
            norm_udata = scaler.transform(intm_udata)

            seen_data = norm_sdata.reshape(seen_data.shape)
            unseen_data = norm_udata.reshape(unseen_data.shape)

        if smoothing:
            seen_data = np.apply_along_axis(smooth, axis=1, arr=seen_data)
            unseen_data = np.apply_along_axis(smooth, axis=1, arr=unseen_data)
        # train-val split
        seen_index = list(range(len(seen_targets)))
        random.shuffle(seen_index)
        split_point = int((1 - seen_ratio) * len(seen_index))
        fst_index, sec_index = seen_index[:split_point], seen_index[split_point:]
        # print(type(fst_index), type(sec_index), type(seen_data), type(seen_targets))
        X_seen_train, X_seen_val, y_seen_train, y_seen_val = seen_data[fst_index, :], seen_data[sec_index, :], \
        seen_targets[fst_index], seen_targets[sec_index]

        data = {'train': {
            'X': X_seen_train,
            'y': y_seen_train
        },
            'eval-seen': {
                'X': X_seen_val,
                'y': y_seen_val
            },
            'test': {
                'X': unseen_data,
                'y': unseen_targets
            },
            'seen_classes': seen_classes,
            'unseen_classes': unseen_classes
        }

        return data

# build PAMAP2 dataset data reader
class PAMAP2ReaderV1(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.readPamap2()

    def readFile(self, file_path):
        all_data = {"data": {}, "target": {}, 'collection': []}
        prev_action = -1
        starting = True
        # action_seq = []
        action_ID = 0

        for l in open(file_path).readlines():
            s = l.strip().split()
            if s[1] != "0":
                if (prev_action != int(s[1])):
                    if not (starting):
                        df = pd.DataFrame(action_seq)
                        intep_df = df.interpolate(method='linear', limit_direction='backward', axis=0)
                        intep_data = intep_df.values
                        all_data['data'][action_ID] = np.array(intep_data)
                        all_data['target'][action_ID] = prev_action
                        action_ID += 1
                    action_seq = []
                else:
                    starting = False
                data_seq = np.array(s[3:]).astype(np.float16)
                # data_seq[np.isnan(data_seq)] = 0
                action_seq.append(data_seq)
                prev_action = int(s[1])
                # print(prev_action)
                all_data['collection'].append(data_seq)
        else:
            if len(action_seq) > 1:
                df = pd.DataFrame(action_seq)
                intep_df = df.interpolate(method='linear', limit_direction='backward', axis=0)
                intep_data = intep_df.values
                all_data['data'][action_ID] = np.array(intep_data)
                all_data['target'][action_ID] = prev_action
        return all_data

    def clean_data(self):
        pass

    def readPamap2Files(self, filelist, cols, labelToId):
        data = []
        labels = []
        collection = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            fpath = os.path.join(self.root_path, filename)
            file_data = self.readFile(fpath)
            data.extend(list(file_data['data'].values()))
            labels.extend(list(file_data['target'].values()))
            collection.extend(file_data['collection'])

        # for i, arr in enumerate(data):
        #    if np.isinf(arr).sum() > 0 or np.isnan(arr).sum() > 0:
        #        print(i, "inf", np.isinf(arr).sum(), "nan", np.isnan(arr).sum())

        data = np.array(data)

        labels = np.asarray(labels, dtype=int)
        collection = np.array(collection).astype(np.float32)

        return data, labels, collection

    def readPamap2(self):
        files = ['subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat', 'subject105.dat',
                 'subject106.dat', 'subject107.dat', 'subject108.dat', 'subject109.dat', 'subject110.dat',
                 'subject111.dat', 'subject112.dat', 'subject113.dat', 'subject114.dat']

        label_map = [
            # (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'Nordic walking'),
            (9, 'watching TV'),
            (10, 'computer work'),
            (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            (18, 'folding laundry'),
            (19, 'house cleaning'),
            (20, 'playing soccer'),
            (24, 'rope jumping')
        ]
        labelToId = {x[0]: i for i, x in enumerate(label_map)}
        # print "label2id=",labelToId
        idToLabel = [x[1] for x in label_map]
        # print "id2label=",idToLabel
        cols = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53
        ]
        # print "cols",cols
        self.data, self.targets, self.all_data = self.readPamap2Files(files, cols, labelToId)
        # print(self.data)
        # nan_perc = np.isnan(self.data).astype(int).mean()
        # print("null value percentage ", nan_perc)
        # f = lambda x: labelToId[x]
        print(np.unique(self.targets))
        self.targets = np.array([labelToId[i] for i in list(self.targets)])
        print(np.unique(self.targets))
        self.label_map = label_map
        self.idToLabel = idToLabel
        # return data, idToLabel

    def aggregate(self, signal):
        means = signal.mean(axis=0)
        # print(means.shape)
        stds = signal.std(axis=0)
        # print(stds.shape)
        if np.isinf(means).sum():
            means[np.isinf(means)] = 0

        if np.isinf(stds).sum():
            stds[np.isinf(stds)] = 0
        mergered = np.vstack((means, stds)).reshape((-1,), order='F')
        return mergered

    def windowing(self, signal, window_len, overlap):
        seq_len = int(window_len * 100)  # 100Hz compensation
        overlap_len = int(overlap * 100)  # 100Hz
        l, _ = signal.shape
        if l > seq_len:
            windowing_points = np.arange(start=0, stop=l - seq_len, step=seq_len - overlap_len, dtype=int)[:-1]
            # windowing_points = windowing_points-overlap_len
            # windowing_points[0] = 0

            windows = [signal[p:p + seq_len, :] for p in windowing_points]
        else:
            windows = []
        return windows

    def resampling(self, data, targets, window_size, window_overlap, resample_freq):
        assert len(data) == len(targets), "# action data & # action labels are not matching"
        all_data, all_ids, all_labels = [], [], []
        for i, d in enumerate(data):
            # print(">>>>>>>>>>>>>>>  ", np.isnan(d).mean())
            label = targets[i]
            windows = self.windowing(d, window_size, window_overlap)
            for w in windows:
                # print(np.isnan(w).mean(), label, i)
                resample_sig = self.aggregate(w)
                # print(np.isnan(resample_sig).mean(), label, i)
                all_data.append(resample_sig)
                all_ids.append(i + 1)
                all_labels.append(label)

        return all_data, all_ids, all_labels

    def generate(self, unseen_classes, resampling=True, window_size=5.21, window_overlap: float = 1, resample_freq=10,
                 seen_ratio=0.2, unseen_ratio=0.8):
        # assert all([i in list(self.label_map.keys()) for i in unseen_classes]), "Unknown Class label!"
        seen_classes = [i for i in range(len(self.idToLabel)) if i not in unseen_classes]
        unseen_mask = np.in1d(self.targets, unseen_classes)

        s = np.unique(self.targets, return_counts=True)
        print("per class count : ", dict(zip([self.idToLabel[i] for i in s[0]], s[1])))

        # build seen dataset
        seen_data = self.data[np.invert(unseen_mask)]
        seen_targets = self.targets[np.invert(unseen_mask)]

        # build unseen dataset
        unseen_data = self.data[unseen_mask]
        unseen_targets = self.targets[unseen_mask]

        # resampling seen and unseen datasets
        seen_data, seen_ids, seen_targets = self.resampling(seen_data, seen_targets, window_size, window_overlap,
                                                            resample_freq)
        unseen_data, unseen_ids, unseen_targets = self.resampling(unseen_data, unseen_targets, window_size,
                                                                  window_overlap, resample_freq)

        seen_data, seen_targets = np.array(seen_data).astype(np.float32), np.array(seen_targets)
        unseen_data, unseen_targets = np.array(unseen_data).astype(np.float32), np.array(unseen_targets)
        # train-val split
        seen_index = list(range(len(seen_targets)))
        random.shuffle(seen_index)
        split_point = int((1 - seen_ratio) * len(seen_index))
        fst_index, sec_index = seen_index[:split_point], seen_index[split_point:]
        print(type(fst_index), type(sec_index), type(seen_data), type(seen_targets))
        X_seen_train, X_seen_val, y_seen_train, y_seen_val = seen_data[fst_index, :], seen_data[sec_index, :], \
            seen_targets[fst_index], seen_targets[sec_index]

        # val-test split
        unseen_index = list(range(len(unseen_targets)))
        random.shuffle(unseen_index)
        split_point = int((1 - unseen_ratio) * len(unseen_index))
        fst_index, sec_index = unseen_index[:split_point], unseen_index[split_point:]

        X_unseen_val, X_unseen_test, y_unseen_val, y_unseen_test = unseen_data[fst_index, :], unseen_data[sec_index, :], \
            unseen_targets[fst_index], unseen_targets[sec_index]

        data = {'train': {
            'X': X_seen_train,
            'y': y_seen_train
        },
            'eval-seen': {
                'X': X_seen_val,
                'y': y_seen_val
            },
            'test': {
                'X': unseen_data,
                'y': unseen_targets
            },
            'seen_classes': seen_classes,
            'unseen_classes': unseen_classes
        }

        return data


class PAMAP2ReaderV2(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.readPamap2()

    def readFile(self, file_path, cols):
        all_data = {"data": {}, "target": {}, 'collection': []}
        prev_action = -1
        starting = True
        # action_seq = []
        action_ID = 0

        for l in open(file_path).readlines():
            s = l.strip().split()
            if s[1] != "0":
                if (prev_action != int(s[1])):
                    if not (starting):
                        df = pd.DataFrame(action_seq)
                        intep_df = df.interpolate(method='linear', limit_direction='backward', axis=0)
                        intep_data = intep_df.values
                        all_data['data'][action_ID] = np.array(intep_data,dtype=np.float64)
                        all_data['target'][action_ID] = prev_action
                        action_ID += 1
                    action_seq = []
                else:
                    starting = False
                data_seq = np.array(s[3:])[cols].astype(np.float64)
                # data_seq[np.isnan(data_seq)] = 0
                action_seq.append(data_seq)
                prev_action = int(s[1])
                # print(prev_action)
                all_data['collection'].append(data_seq)
        else:
            if len(action_seq) > 1:
                df = pd.DataFrame(action_seq)
                intep_df = df.interpolate(method='linear', limit_direction='backward', axis=0)
                intep_data = intep_df.values
                all_data['data'][action_ID] = np.array(intep_data)
                all_data['target'][action_ID] = prev_action
        return all_data

    def readPamap2Files(self, filelist, cols, labelToId):
        data = []
        labels = []
        collection = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            fpath = os.path.join(self.root_path, filename)
            file_data = self.readFile(fpath, cols)
            data.extend(list(file_data['data'].values()))
            labels.extend(list(file_data['target'].values()))
            collection.extend(file_data['collection'])
        return np.asarray(data), np.asarray(labels, dtype=int), np.array(collection)

    def readPamap2(self):
        files = ['subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat', 'subject105.dat',
                 'subject106.dat', 'subject107.dat', 'subject108.dat', 'subject109.dat', 'subject110.dat',
                 'subject111.dat', 'subject112.dat', 'subject113.dat', 'subject114.dat']

        label_map = [
            #(0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'Nordic walking'),
            (9, 'watching TV'),
            (10, 'computer work'),
            (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            (18, 'folding laundry'),
            (19, 'house cleaning'),
            (20, 'playing soccer'),
            (24, 'rope jumping')
        ]
        labelToId = {x[0]: i for i, x in enumerate(label_map)}
        # print "label2id=",labelToId
        idToLabel = [x[1] for x in label_map]
        # print "id2label=",idToLabel
        cols = [1, 2, 3, 7, 8, 9, 17, 18, 19, 23, 24, 25, 33, 34, 35, 39, 40, 41]
        # print "cols",cols
        self.data, self.targets, self.all_data = self.readPamap2Files(files, cols, labelToId)
        # print(self.data)
        # nan_perc = np.isnan(self.data).astype(int).mean()
        # print("null value percentage ", nan_perc)
        # f = lambda x: labelToId[x]
        print(np.unique(self.targets))
        self.targets = np.array([labelToId[i] for i in list(self.targets)])
        print(np.unique(self.targets))
        self.label_map = label_map
        self.idToLabel = idToLabel
        # return data, idToLabel

    def aggregate(self, signal):
        means = signal.mean(axis=0)
        stds = signal.std(axis=0)

        #if np.isinf(means).sum():
        #    means[np.isinf(means)] = 0

        #if np.isinf(stds).sum():
        #    stds[np.isinf(stds)] = 0

        mergered = np.vstack((means, stds)).reshape((-1,), order='F')
        # print(signal.shape, means.shape, stds.shape, mergered.shape)
        return mergered

    def windowing(self, signal, window_len, overlap):
        seq_len = int(window_len * 100)  # 100Hz compensation
        overlap_len = int(overlap * 100)  # 100Hz
        l, _ = signal.shape
        if l > seq_len:
            windowing_points = np.arange(start=0, stop=l - seq_len, step=seq_len - overlap_len, dtype=int)[:-1]
            # windowing_points = windowing_points-overlap_len
            # windowing_points[0] = 0

            windows = [signal[p:p + seq_len, :] for p in windowing_points]
        else:
            windows = []
        return windows

    def resampling(self, data, targets, window_size, window_overlap, resample_freq):
        assert len(data) == len(targets), "# action data & # action labels are not matching"
        all_data, all_ids, all_labels = [], [], []
        for i, d in enumerate(data):
            # print(">>>>>>>>>>>>>>>  ", np.isnan(d).mean())
            label = targets[i]
            windows = self.windowing(d, window_size, window_overlap)
            for w in windows:
                # print(np.isnan(w).mean(), label, i)
                resample_sig = self.aggregate(w)
                # print(np.isnan(resample_sig).mean(), label, i)
                all_data.append(resample_sig)
                all_ids.append(i + 1)
                all_labels.append(label)

        return all_data, all_ids, all_labels

    def generate(self, unseen_classes, resampling=True, window_size=5.21, window_overlap: float = 1, resample_freq=10,
                 seen_ratio=0.2, unseen_ratio=0.8):
        # assert all([i in list(self.label_map.keys()) for i in unseen_classes]), "Unknown Class label!"
        seen_classes = [i for i in range(len(self.idToLabel)) if i not in unseen_classes]
        unseen_mask = np.in1d(self.targets, unseen_classes)

        s = np.unique(self.targets, return_counts=True)
        print("per class count : ", dict(zip([self.idToLabel[i] for i in s[0]], s[1])))

        # build seen dataset
        seen_data = self.data[np.invert(unseen_mask)]
        seen_targets = self.targets[np.invert(unseen_mask)]

        # build unseen dataset
        unseen_data = self.data[unseen_mask]
        unseen_targets = self.targets[unseen_mask]

        # resampling seen and unseen datasets
        seen_data, seen_ids, seen_targets = self.resampling(seen_data, seen_targets, window_size, window_overlap,
                                                            resample_freq)
        unseen_data, unseen_ids, unseen_targets = self.resampling(unseen_data, unseen_targets, window_size,
                                                                  window_overlap, resample_freq)

        seen_data, seen_targets = np.array(seen_data), np.array(seen_targets)
        unseen_data, unseen_targets = np.array(unseen_data), np.array(unseen_targets)
        # train-val split
        seen_index = list(range(len(seen_targets)))
        random.shuffle(seen_index)
        split_point = int((1 - seen_ratio) * len(seen_index))
        fst_index, sec_index = seen_index[:split_point], seen_index[split_point:]
        print(type(fst_index), type(sec_index), type(seen_data), type(seen_targets))
        X_seen_train, X_seen_val, y_seen_train, y_seen_val = seen_data[fst_index, :], seen_data[sec_index, :], \
            seen_targets[fst_index], seen_targets[sec_index]

        # val-test split
        unseen_index = list(range(len(unseen_targets)))
        random.shuffle(unseen_index)
        split_point = int((1 - unseen_ratio) * len(unseen_index))
        fst_index, sec_index = unseen_index[:split_point], unseen_index[split_point:]

        X_unseen_val, X_unseen_test, y_unseen_val, y_unseen_test = unseen_data[fst_index, :], unseen_data[sec_index, :], \
            unseen_targets[fst_index], unseen_targets[sec_index]

        data = {'train': {
            'X': X_seen_train,
            'y': y_seen_train
        },
            'eval-seen': {
                'X': X_seen_val,
                'y': y_seen_val
            },
            'test': {
                'X': unseen_data,
                'y': unseen_targets
            },
            'seen_classes': seen_classes,
            'unseen_classes': unseen_classes
        }

        return data


if __name__ == "__main__":
    dataReader = PAMAP2ReaderV2('../../data/PAMAP2_Dataset/Protocol/')
    actionList = dataReader.idToLabel

    fold_classes = [['watching TV', 'house cleaning', 'standing', 'ascending stairs'],
                    ['walking', 'rope jumping', 'sitting', 'descending stairs'],
                    ['playing soccer', 'lying', 'vacuum cleaning', 'computer work'],
                    ['cycling', 'running', 'Nordic walking'], ['ironing', 'car driving', 'folding laundry']]

    fold_cls_ids = [[actionList.index(i) for i in j] for j in fold_classes]

    data_dict = dataReader.generate(unseen_classes=fold_cls_ids[0], seen_ratio=0.2, unseen_ratio=0.8, window_size=5.21,
                                    window_overlap=4.21, resample_freq=20)
