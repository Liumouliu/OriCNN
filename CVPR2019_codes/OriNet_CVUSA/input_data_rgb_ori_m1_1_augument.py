import cv2
import random
import numpy as np

import scipy.io as sio

noise_level = 360

class InputData:

    # the path of your CVUSA dataset
    img_root = '/media/pan/pan/liu/crossview_localisation/src/CVUSA/'

    yaw_pitch_grd = sio.loadmat('./CVUSA_orientations/yaw_pitch_grd_CVUSA.mat')
    yaw_sat = sio.loadmat('./CVUSA_orientations/yaw_radius_sat_CVUSA.mat')

    def __init__(self):

        self.train_list = self.img_root + 'splits/train-19zl.csv'
        self.test_list = self.img_root + 'splits/val-19zl.csv'

        print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_list.append([data[0], data[1], pano_id])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)
        print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)


        print('InputData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_test_list.append([data[0], data[1], pano_id])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        print('InputData::__init__: load', self.test_list, ' data_size =', self.test_data_size)




    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None, None, None
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            batch_size = self.test_data_size - self.__cur_test_id

        batch_grd = np.zeros([batch_size, 224, 1232, 3], dtype = np.float32)
        batch_sat = np.zeros([batch_size, 512, 512, 3], dtype=np.float32)

        batch_grd_yawpitch = np.zeros([batch_size, 224, 1232, 2], dtype=np.float32)
        batch_sat_yaw = np.zeros([batch_size, 512, 512, 2], dtype=np.float32)

        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # satellite
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][0])
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # img -= 100.0
            img = img / 255.0
            img = img * 2.0 - 1.0
            batch_sat[i, :, :, :] = img


            # ground
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][1])
            img = img.astype(np.float32)
            # img -= 100.0
            img = img / 255.0
            img = img * 2.0 - 1.0
            batch_grd[i, :, :, :] = img


            # orientation of ground, normilze to [-1 1]
            batch_grd_yawpitch[i, :, :, 0] = self.yaw_pitch_grd['orient_mat'][:,:,0].astype(np.float32)/np.pi
            batch_grd_yawpitch[i, :, :, 1] = self.yaw_pitch_grd['orient_mat'][:,:,1].astype(np.float32)/np.pi

            # orientation of aerial

            batch_sat_yaw[i, :, :, 0] = cv2.resize(self.yaw_sat['polor_mat'][:,:,0].astype(np.float32)/np.pi, (512, 512), interpolation=cv2.INTER_AREA)
            batch_sat_yaw[i, :, :, 1] = cv2.resize((self.yaw_sat['polor_mat'][:,:,1].astype(np.float32) - 0.5)*2.0, (512, 512), interpolation=cv2.INTER_AREA)



        self.__cur_test_id += batch_size

        return batch_sat, batch_grd, batch_sat_yaw, batch_grd_yawpitch



    def next_pair_batch(self, batch_size):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.id_idx_list)

        if self.__cur_id + batch_size + 2 >= self.data_size:
            self.__cur_id = 0
            return None, None, None, None

        batch_sat = np.zeros([batch_size, 512, 512, 3], dtype=np.float32)
        batch_grd = np.zeros([batch_size, 224, 1232, 3], dtype=np.float32)

        batch_grd_yawpitch = np.zeros([batch_size, 224, 1232, 2], dtype=np.float32)
        batch_sat_yaw = np.zeros([batch_size, 512, 512, 2], dtype=np.float32)

        i = 0
        batch_idx = 0
        while True:
            if batch_idx >= batch_size or self.__cur_id + i >= self.data_size:
                break

            img_idx = self.id_idx_list[self.__cur_id + i]
            i += 1

            # satellite
            img = cv2.imread(self.img_root + self.id_list[img_idx][0])
            img_yaw = self.yaw_sat['polor_mat'][:,:,0].astype(np.float32)/np.pi
            img_radis = (self.yaw_sat['polor_mat'][:,:,1].astype(np.float32) - 0.5)*2.0

            if img is None or img.shape[0] != img.shape[1]:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][0], i), img.shape)
                continue

            rand_crop = random.randint(1, 748)
            if rand_crop > 512:
                start = int((750 - rand_crop) / 2)
                img = img[start : start + rand_crop, start : start + rand_crop, :]
                img_yaw = img_yaw[start: start + rand_crop, start: start + rand_crop]
                img_radis = img_radis[start: start + rand_crop, start: start + rand_crop]
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            img_yaw = cv2.resize(img_yaw, (512, 512), interpolation=cv2.INTER_AREA)
            img_radis = cv2.resize(img_radis, (512, 512), interpolation=cv2.INTER_AREA)

            rand_rotate = random.randint(0, 4) * 90
            rot_matrix = cv2.getRotationMatrix2D((256, 256), rand_rotate, 1)
            img = cv2.warpAffine(img, rot_matrix, (512, 512))
            img = img.astype(np.float32)

            img_yaw = cv2.warpAffine(img_yaw, rot_matrix, (512, 512))
            img_radis = cv2.warpAffine(img_radis, rot_matrix, (512, 512))

            img = img / 255.0
            img = img * 2.0 - 1.0

            batch_sat[batch_idx, :, :, :] = img
            batch_sat_yaw[batch_idx, :, :, 0] = img_yaw
            batch_sat_yaw[batch_idx, :, :, 1] = img_radis


            # ground
            img = cv2.imread(self.img_root + self.id_list[img_idx][1])
            if img is None or img.shape[0] != 224 or img.shape[1] != 1232:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][1], i), img.shape)
                continue
            img = img.astype(np.float32)
            # img -= 100.0
            img = img / 255.0
            img = img * 2.0 - 1.0


            # generate a random crop and past step
            shift_pixes = int(np.random.uniform(0,noise_level/360.0*(1232.0-1.0)))

            batch_grd[batch_idx,:,0:1232-shift_pixes,:] =  img[:,shift_pixes:1232,:]
            batch_grd[batch_idx,:,1232-shift_pixes:1232,:] =  img[:,0:shift_pixes,:]

            # orientation of ground, normilze to [-1 1]
            batch_grd_yawpitch[batch_idx, :, 0:1232-shift_pixes, :] = self.yaw_pitch_grd['orient_mat'][:,shift_pixes:1232,:].astype(np.float32)/np.pi
            batch_grd_yawpitch[batch_idx, :, 1232-shift_pixes:1232, :] = self.yaw_pitch_grd['orient_mat'][:,0:shift_pixes,:].astype(np.float32)/np.pi

            batch_idx += 1

        self.__cur_id += i

        return batch_sat, batch_grd, batch_sat_yaw, batch_grd_yawpitch


    def get_dataset_size(self):
        return self.data_size

    def get_test_dataset_size(self):
        return self.test_data_size

    def reset_scan(self):
        self.__cur_test_idd = 0

