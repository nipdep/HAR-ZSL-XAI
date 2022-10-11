import os
import random 
import numpy as np
import pandas as pd
from scipy.signal import resample

# build PAMAP2 dataset data reader
class PAMAP2Reader(object):
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
                    if not(starting):
                        all_data['data'][action_ID] = np.array(action_seq)
                        all_data['target'][action_ID] = prev_action
                        action_ID+=1
                    action_seq = []
                else:
                    starting = False
                data_seq = np.nan_to_num(np.array(s[3:]), nan=0).astype(np.float16)
                # data_seq[np.isnan(data_seq)] = 0
                action_seq.append(data_seq)
                prev_action = int(s[1])
                all_data['collection'].append(data_seq)
        return all_data

    def readPamap2Files(self, filelist, cols, labelToId):
        data = []
        labels = []
        collection = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i+1, len(filelist)))
            fpath = os.path.join(self.root_path, filename)
            file_data = self.readFile(fpath)
            data.extend(list(file_data['data'].values()))
            labels.extend(list(file_data['target'].values()))
            collection.extend(file_data['collection'])
        return np.asarray(data), np.asarray(labels, dtype=int), np.array(collection)

    def readPamap2(self):
        files = ['subject101.dat', 'subject102.dat','subject103.dat','subject104.dat', 'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat', 'subject109.dat']
            
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
        cols = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53
                ]
        # print "cols",cols
        self.data, self.targets, self.all_data = self.readPamap2Files(files, cols, labelToId)
        # print(self.data)
        # nan_perc = np.isnan(self.data).astype(int).mean()
        # print("null value percentage ", nan_perc)
        # f = lambda x: labelToId[x]
        self.targets = np.array([labelToId[i] for i in list(self.targets)])
        self.label_map = label_map
        self.idToLabel = idToLabel
        # return data, idToLabel

    def resampling(self, data, targets):
        assert len(data) == len(targets), "# action data & # action labels are not matching"
        all_data, all_ids, all_labels = [], [], []
        for i, d in enumerate(data):
            l, _ = d.shape 
            label = targets[i]
            if l < 1200 : # minimum length requirement
                break 
            # generate sampling points
            n_point, _ = divmod(l, 1200)
            sampling_points = np.arange(start=0, stop=l, step=1200)[:-1]
            sampling_points = sampling_points-200
            sampling_points[0] = 0
            # print(l, sampling_points)
            # window sampling 
            for s in sampling_points:
                sub_sample = np.nan_to_num(resample(d[s:s+1200, :], num=120, axis=0), nan=0)
                # print(d[s:s+1200, :])
                nan_perc = np.isnan(sub_sample).astype(int).mean()
                # print("null value percentage ", nan_perc, "id > ", i+1)
                all_data.append(sub_sample)
                all_ids.append(i+1)
                all_labels.append(label)
            
        return all_data, all_ids, all_labels

    def generate(self, unseen_classes, resampling=True, seen_ratio=0.2, unseen_ratio=0.8):
        # assert all([i in list(self.label_map.keys()) for i in unseen_classes]), "Unknown Class label!"
        seen_classes = [i for i in range(len(self.idToLabel)) if i not in unseen_classes]
        unseen_mask = np.in1d(self.targets, unseen_classes)

        # normalize data [without considering ]
        # scaler = MinMaxScaler()
        # scaler.fit(self.all_data)
        # self.data = np.array([scaler.transform(d) for d in self.data])

        # build seen dataset 
        seen_data = self.data[np.invert(unseen_mask)]
        seen_targets = self.targets[np.invert(unseen_mask)]

        # build unseen dataset
        unseen_data = self.data[unseen_mask]
        unseen_targets = self.targets[unseen_mask]

        # resampling seen and unseen datasets 
        seen_data, seen_ids, seen_targets = self.resampling(seen_data, seen_targets)
        unseen_data, unseen_ids, unseen_targets = self.resampling(unseen_data, unseen_targets)

        seen_data, seen_targets = np.array(seen_data), np.array(seen_targets)
        unseen_data, unseen_targets = np.array(unseen_data), np.array(unseen_targets)
        # train-val split
        seen_index = list(range(len(seen_targets)))
        random.shuffle(seen_index)
        split_point = int((1-seen_ratio)*len(seen_index))
        fst_index, sec_index = seen_index[:split_point], seen_index[split_point:]
        print(type(fst_index), type(sec_index), type(seen_data), type(seen_targets))
        X_seen_train, X_seen_val, y_seen_train, y_seen_val = seen_data[fst_index,:], seen_data[sec_index,:], seen_targets[fst_index], seen_targets[sec_index]
        
        # val-test split
        unseen_index = list(range(len(unseen_targets)))
        random.shuffle(unseen_index)
        split_point = int((1-unseen_ratio)*len(unseen_index))
        fst_index, sec_index = unseen_index[:split_point], unseen_index[split_point:]

        X_unseen_val, X_unseen_test, y_unseen_val, y_unseen_test = unseen_data[fst_index,:], unseen_data[sec_index,:], unseen_targets[fst_index], unseen_targets[sec_index]

        data = {'train': {
                        'X': X_seen_train,
                        'y': y_seen_train
                        },
                'eval-seen':{
                        'X': X_seen_val,
                        'y': y_seen_val
                        },
                'eval-unseen':{
                        'X': X_unseen_val,
                        'y': y_unseen_val
                        },
                'test': {
                        'X': X_unseen_test,
                        'y': y_unseen_test
                        },
                'seen_classes': seen_classes,
                'unseen_classes': unseen_classes
                }

        return data
        

class KUHARData(object):
    """KU-HAR dataset implementation"""

    def __init__(self, data_dir, n_proc=1, limit_size=300, config=None, filter_classes=[]):
        self.filter_classes = filter_classes
        self.all_df, self.labels_df = self.load_all(data_dir)
        self.all_IDs = self.all_df.index.unique()
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df[self.feature_names]
        self.class_names = self.labels_df.label.unique()
    
    def load_data(self, data_dir):
        df = pd.read_csv(data_dir)
        return df

    def label_shift(self, l):
        i = 0
        if self.filter_classes != []:
            for j in self.filter_classes:
                if l > j:
                    i+=1
        return i

    def load_all(self, data_dir):
        main_df = self.load_data(data_dir)
        subdf_list = []
        label_dict = {'ID': [], 'label': []}

        for i, r in main_df.iterrows():
            r = r.values 
            acx, acy, acz, gyx, gyy, gyz, label, _, ID = r[:300], r[300:600], r[600:900], r[900:1200], r[1200:1500], r[1500:1800], r[1800], r[1801], r[1802]
            if int(label) not in self.filter_classes:
                sub_df = pd.DataFrame({'accelX': acx, 'accelY': acy, 'accelZ': acz, 'GyroX': gyx, 'GyroY': gyy, 'GyroZ': gyz}, index=[int(ID),]*300)
                label = label-self.label_shift(label)
                label_dict['label'].append(int(label))
                label_dict['ID'].append(int(ID))
                subdf_list.append(sub_df)
                label_df = pd.DataFrame(label_dict)
                label_df.set_index('ID', inplace=True)

        full_df = pd.concat(subdf_list)
        # full_df.reset_index(inplace=True, drop=True)
        return full_df, label_df

