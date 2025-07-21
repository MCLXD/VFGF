""" Implements a dataset object which allows to read representations from LMDB datasets in a multi-modal fashion
The dataset can sample frames for both the anticipation and early recognition tasks."""

import csv
import lmdb
import random
import scipy.io as scio
import numpy as np
from tqdm import tqdm
from torch.utils import data
import pandas as pd
import torch
from torch.nn import functional as F
from datetime import datetime
def read_representations(frames, env, tran=None):
    """ Reads a set of representations, given their frame names and an LMDB environment.
    Applies a transformation to the features if provided"""
    features = []
    # for each frame
    boo = False
    for f in frames:
        # read the current frame
        with env.begin() as e:
            dd = e.get(f.strip().encode('utf-8'))
        if dd is None:
            print(f)
        # convert to numpy array
        data = np.frombuffer(dd, 'float32')
        # append to list
        features.append(data)
    # convert list to numpy array
    features=np.array(features)
    # apply transform if provided
    if tran:
        features=tran(features)
    return features

def read_data(frames, env, tran=None):
    """A wrapper form read_representations to handle loading from more environments.
    This is used for multimodal data loading (e.g., RGB + Flow)"""
    # if env is a list
    if isinstance(env, list):
        # read the representations from all
        l = [read_representations(frames, e, tran) for e in env]
        return l
    else:
        # otherwise, just read the representations
        return read_representations(frames, env, tran)

class SequenceDataset(data.Dataset):
    def __init__(self, path_to_lmdb, path_to_csv, label_type = 'action',
                time_step = 0.25, sequence_length = 14, Rand_length = 0, fps = 30,
                img_tmpl = "frame_{:010d}.jpg",
                transform = None,
                challenge = False,
                past_features = True,
                action_samples = None,
                revision_threshold = None, N=1):
        """
            Inputs:
                path_to_lmdb: path to the folder containing the LMDB dataset
                path_to_csv: path to training/validation csv
                label_type: which label to return (verb, noun, or action)
                time_step: in seconds
                sequence_length: in time steps
                fps: framerate
                img_tmpl: image template to load the features
                tranform: transformation to apply to each sample
                past_features: if past features should be returned
                action_samples: number of frames to be evenly sampled from each action
        """

        self.annotations = pd.read_csv(path_to_csv, header=None, names=['video','start','end','verb','noun','action'])
        self.path_to_lmdb = path_to_lmdb

        self.time_step = time_step
        self.past_features = past_features
        self.action_samples = action_samples
        self.fps=fps
        self.transform = transform
        self.label_type = label_type
        self.sequence_length = sequence_length
        self.Rand_length = Rand_length
        self.img_tmpl = img_tmpl
        self.action_samples = action_samples
        self.challenge = challenge
        self.N = N
        self.revision_threshold = revision_threshold
        self.train_video_len_dict = {}
        # with open('./data/training_videos_length.csv', 'r') as f:#
        #     reader = csv.reader(f)
        #     reader = list(reader)
        #     for sample in reader:
        #         self.train_video_len_dict[sample[0]] = int(sample[1])
        self.random_frames = {}
        #self.random_frames = []
        self.rand_index = []
        #以只读模式打开文件，指定编码为 UTF-8
        # with open('./data/rand.csv', 'r', encoding='utf-8') as csvfile:
        #     # 创建 CSV 读取器对象
        #     reader = csv.reader(csvfile)
        #     # 获取 CSV 文件的总行数
        #     total_lines = sum(1 for _ in csvfile)
        #     # 将文件指针重置到文件开头
        #     csvfile.seek(0)
        #
        #     # 使用 tqdm 为循环添加进度条
        #     for row in tqdm(reader, total=total_lines, desc="Reading CSV"):
        #         # 每一行的内容已经是列表形式
        #         # 去掉换行符
        #         final_result = []
        #         for it in row[1:]:
        #             content = it[1:-1].replace('\n', '')
        #             # 按照单引号进行分割
        #             result = content.split("'")
        #
        #             # 过滤掉空字符串和只有空格的字符串
        #             final_result.append([item.strip() for item in result if item.strip()])
        #         self.random_frames[row[0]] = final_result
        # initialize some lists
        self.ids = [] # action ids
        self.discarded_ids = [] # list of ids discarded (e.g., if there were no enough frames before the beginning of the action
        self.discarded_labels = []
        self.past_frames = [] # names of frames sampled before each action
        self.action_frames = [] # names of frames sampled from each action
        self.labels = [] # labels of each action

        #
        self.__populate_lists()

        # if a list to datasets has been provided, load all of them
        if isinstance(self.path_to_lmdb, list):
            self.env = [lmdb.open(paht_l, readonly=True, lock=False) for paht_l in self.path_to_lmdb]
        else:
            # otherwise, just load the single LMDB dataset
            self.env = lmdb.open(self.path_to_lmdb, readonly=True, lock=False)

    def __get_frames(self, frames, video):
        """ format file names using the image template """

        frames = np.array(list(map(lambda x: video+"_"+self.img_tmpl.format(x), frames)))
        return frames

    def __populate_lists(self):
        """ Samples a sequence for each action and populates the lists. """
        for _, a in tqdm(self.annotations.iterrows(), 'Populating Dataset', total = len(self.annotations)):

            # sample frames before the beginning of the action
            frames, _ = self.__sample_frames_past(a.start, a.video)
            temp_random_frames = []
            self.rand_index.append(a.video + str(a.start))

            # li = [a.video + str(a.start)]
            # for i, temp in enumerate(all_random_frames):
            #     #import ipdb; ipdb.set_trace()
            #     rand_frames = self.__get_frames(temp[0], temp[1])
            #     temp_random_frames.append(rand_frames)
            #     li.append(rand_frames)
            # with open('./data/rand.csv', 'a', encoding='utf-8', newline='') as csvfile:
            #     # 创建 CSV 写入器对象
            #     writer = csv.writer(csvfile)
            #     # 写入单行数据
            #     writer.writerow(li)
            #import ipdb; ipdb.set_trace()
            #self.random_frames.append(np.array(temp_random_frames))

            if self.action_samples:
                # sample frames from the action
                # to sample n frames, we first sample n+1 frames with linspace, then discard the first one
                action_frames = np.linspace(a.start, a.end, self.action_samples+1, dtype=int)[1:]

            # check if there were enough frames before the beginning of the action
            if frames.min()>=1: #if the smaller frame is at least 1, the sequence is valid
                self.past_frames.append(self.__get_frames(frames, a.video))
                self.ids.append(a.name)
                # handle whether a list of labels is required (e.g., [verb, noun]), rather than a single action
                if isinstance(self.label_type, list):
                    # otherwise get the required labels
                    self.labels.append(a[self.label_type].values.astype(int))
                else: #single label version
                    self.labels.append(a[self.label_type])
                if self.action_samples:
                    self.action_frames.append(self.__get_frames(action_frames, a.video))

            else:
                # if the sequence is invalid, do nothing, but add the id to the discarded_ids list
                self.discarded_ids.append(a.name)
                if isinstance(self.label_type, list):
                    if self.challenge:  # if sampling for the challenge, there are no labels, just add -1
                        self.discarded_labels.append(-1)
                    else:
                        # otherwise get the required labels
                        self.discarded_labels.append(a[self.label_type].values.astype(int))
                else:  # single label version
                    if self.challenge:
                        self.discarded_labels.append(-1)
                    else:
                        self.discarded_labels.append(a[self.label_type])

    def __sample_frames_past(self, point, video_name):
        """Samples frames before the beginning of the action "point" """
        # generate the relative timestamps, depending on the requested sequence_length
        # e.g., 2.  , 1.75, 1.5 , 1.25, 1.  , 0.75, 0.5 , 0.25
        # in this case "2" means, sample 2s before the beginning of the action
        time_stamps = np.arange(self.time_step,self.time_step*(self.sequence_length+1),self.time_step)[::-1]
        # compute the time stamp corresponding to the beginning of the action
        end_time_stamp = point/self.fps

        # subtract time stamps to the timestamp of the last frame
        time_stamps = end_time_stamp-time_stamps

        # convert timestamps to frames
        # use floor to be sure to consider the last frame before the timestamp (important for anticipation!)
        # and never sample any frame after that time stamp
        frames = np.floor(time_stamps*self.fps).astype(int)

        # sometimes there are not enough frames before the beginning of the action
        # in this case, we just pad the sequence with the first frame
        # this is done by replacing all frames smaller than 1
        # with the first frame of the sequence
        if frames.max()>=1:
            frames[frames<1]=frames[frames>=1].min()

        #import ipdb; ipdb.set_trace()
        all_random_frames = []

        # 随机选择要删除的元素索引

        indices = random.sample(range(self.sequence_length), self.Rand_length)
        indices.sort(reverse=True)  # 倒序排序，避免索引变化影响
        frames = frames.tolist()
        # 按倒序删除元素，防止索引偏移
        for idx in indices:
            frames.pop(idx)
        return np.array(frames), all_random_frames

    def __len__(self):
        return len(self.ids)
    def my_triu_indices(self, row, col, offset=0):
      indices = []
      for i in range(row):
          for j in range(i + offset, col):
              indices.append([i, j])
      return torch.tensor(indices).t()
    def compute_mse_between_frames_matrix(self, out):
        new_x_norm = []
        negative_feature = []
        if not isinstance(out, list):
            out = [out]
        for i, it in enumerate(out):
            x = F.normalize(torch.from_numpy(it), p=2, dim=1)
            threshold = self.revision_threshold[i]
            assert threshold > 0
            num_frames, dim = x.shape
            # 扩展维度以进行广播
            x_expanded_i = x.unsqueeze(1)
            x_expanded_j = x.unsqueeze(0)
            # 计算差的平方并求均方误差矩阵
            mse_matrix = torch.mean((x_expanded_i - x_expanded_j) ** 2, dim=-1)

            # 只保留上三角部分（不包括对角线）
            upper_tri_indices = self.my_triu_indices(num_frames, num_frames, offset=1)
            mse_matrix = mse_matrix[upper_tri_indices[0], upper_tri_indices[1]]
            #print(mse_matrix)
            # 存储每一行第一个大于阈值的下标，如果没有则存储最大下标
            index_list = []
            row_start = 0
            for i in range(1, num_frames):
                row_end = row_start + (num_frames - i)
                row = mse_matrix[row_start:row_end]
                #print(row)
                #indices = torch.where(row > threshold, ones, zeros).nonzero()[:, 0]
                indices = torch.where(row > threshold)[0]
                index = indices[0].item() + i if len(indices) > 0 else torch.argmax(row, dim=0)+i
                index_list.append(index)
                row_start = row_end
            # 重新构建 new_x_norm
            index_list.append(num_frames - 1)  # 直接添加最后一行的索引
            #print(index_list)
            new_x_norm_t = []
            negative_feature_t = []
            for i in index_list:
                new_x_norm_t.append(x[i, :].unsqueeze(0))
                negative_feature_t.append(torch.cat([x[j, :].unsqueeze(0) for j in range(num_frames) if i != j]))
            new_x_norm.append(torch.stack(new_x_norm_t))
            negative_feature.append(torch.stack(negative_feature_t[6:]))
        return new_x_norm, negative_feature

    def compute_cosine_similarity_between_frames_matrix(self, out):
        new_x_norm = []
        negative_feature = []
        for i, x in enumerate(out):
            threshold = self.revision_threshold[i]
            assert threshold > 0
            num_frames, dim = x.shape
            # 计算余弦相似度矩阵
            x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
            cosine_matrix = torch.mm(x_norm, x_norm.t())

            # 只保留上三角部分（不包括对角线）
            upper_tri_indices = self.my_triu_indices(num_frames, num_frames, offset=1)
            cosine_matrix = cosine_matrix[upper_tri_indices[0], upper_tri_indices[1]]
            #print(cosine_matrix)
            # 存储每一行第一个小于阈值的下标，如果没有则存储最大下标
            index_list = []
            row_start = 0
            for i in range(1, num_frames):
                row_end = row_start + (num_frames - i)
                row = cosine_matrix[row_start:row_end]
                indices = torch.where(row < threshold)[0]
                index = indices[0].item() + i if len(indices) > 0 else torch.argmin(row, dim=0) + i
                index_list.append(index)
                row_start = row_end

            # 重新构建 new_x_norm
            index_list.append(num_frames - 1)  # 直接添加最后一行的索引
            #print(index_list)
            new_x_norm_t = []
            negative_feature_t = []
            for i in index_list:
                new_x_norm_t.append(x[i, :].unsqueeze(0))
                negative_feature_t.append(torch.cat([x[j, :].unsqueeze(0) for j in range(num_frames) if i != j]))

            new_x_norm = torch.stack(new_x_norm_t)
            negative_feature = torch.stack(negative_feature_t[6:])
        return new_x_norm, negative_feature
    def __getitem__(self, index):
        """ sample a given sequence """
        # get past frames
        past_frames = self.past_frames[index]
        if self.action_samples:
            # get action frames
            action_frames = self.action_frames[index]

        # return a dictionary containing the id of the current sequence
        # this is useful to produce the jsons for the challenge
        out = {'id':self.ids[index]}

        if self.past_features:
            # read representations for past frames
            out['past_features'] = read_data(past_frames, self.env, self.transform)
        #out['all_etp_features'], out['all_sample_features'] = self.compute_mse_between_frames_matrix(out['past_features'])
        # get the label of the current sequence
        label = self.labels[index]
        out['label'] = label

        if self.action_samples:
            # read representations for the action samples
            out['action_features'] = read_data(action_frames, self.env, self.transform)
        #import ipdb; ipdb.set_trace()

        out['past_frames'] = list(past_frames)

        #all_random_frames = self.random_frames[index]
        # all_random_frames = self.random_frames[self.rand_index[index]]
        # all_random_features = []
        # for j in range(6, len(all_random_frames)):
        #     all_random_features.append(read_data(all_random_frames[j], self.env, self.transform))
        #
        #
        # #import ipdb; ipdb.set_trace()
        # out['all_sample_features'] = np.array(all_random_features)
        return out

