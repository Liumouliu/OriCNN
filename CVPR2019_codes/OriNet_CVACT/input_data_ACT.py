import cv2
import random
import numpy as np
# load the yaw, pitch angles for the street-view images and yaw angles for the aerial view
import scipy.io as sio

noise_level = 360

class InputData:

    # the path of your CVACT dataset

    img_root = '/media/pan/pan/liu/crossview_localisation/CVACT/ANU_data_small/'

    yaw_pitch_grd = sio.loadmat('./CVACT_orientations/yaw_pitch_grd_CVACT.mat')
    yaw_sat = sio.loadmat('./CVACT_orientations/yaw_radius_sat_CVACT.mat')

    posDistThr = 25
    posDistSqThr = posDistThr*posDistThr

    panoCropPixels = 832 / 2

    panoRows = 224

    panoCols = 1232

    satSize = 512



    def __init__(self):

        self.allDataList = './CVACT_orientations/ACT_data.mat'
        print('InputData::__init__: load %s' % self.allDataList)

        self.__cur_allid = 0  # for training
        self.id_alllist = []
        self.id_idx_alllist = []

        # load the mat
        anuData = sio.loadmat(self.allDataList)


        idx = 0
        for i in range(0,len(anuData['panoIds'])):
            grd_id_ori = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][i] + '_zoom_2.jpg'
            grd_id_align = self.img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.jpg'
            grd_id_ori_sem = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][i] + '_zoom_2_sem.jpg'
            grd_id_align_sem = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][i] + '_zoom_2_aligned_sem.jpg'
            sat_id_ori = self.img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.jpg'
            sat_id_sem = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][i] + '_satView_sem.jpg'
            self.id_alllist.append([grd_id_ori, grd_id_align, grd_id_ori_sem, grd_id_align_sem, sat_id_ori, sat_id_sem, anuData['utm'][i][0], anuData['utm'][i][1]])
            self.id_idx_alllist.append(idx)
            idx += 1
        self.all_data_size = len(self.id_alllist)
        print('InputData::__init__: load', self.allDataList, ' data_size =', self.all_data_size)

        # partion the images into cells

        self.utms_all = np.zeros([2, self.all_data_size], dtype = np.float32)
        for i in range(0, self.all_data_size):
            self.utms_all[0, i] = self.id_alllist[i][6]
            self.utms_all[1, i] = self.id_alllist[i][7]

        self.training_inds = anuData['trainSet']['trainInd'][0][0] - 1

        self.trainNum = len(self.training_inds)

        self.trainList = []
        self.trainIdList = []
        self.trainUTM = np.zeros([2, self.trainNum], dtype = np.float32)
        for k in range(self.trainNum):
            self.trainList.append(self.id_alllist[self.training_inds[k][0]])
            self.trainUTM[:,k] = self.utms_all[:,self.training_inds[k][0]]
            self.trainIdList.append(k)

        self.__cur_id = 0  # for training

        self.val_inds = anuData['valSet']['valInd'][0][0] - 1
        self.valNum = len(self.val_inds)

        self.valList = []
        self.valUTM = np.zeros([2, self.valNum], dtype=np.float32)
        for k in range(self.valNum):
            self.valList.append(self.id_alllist[self.val_inds[k][0]])
            self.valUTM[:, k] = self.utms_all[:, self.val_inds[k][0]]

        self.__cur_test_id = 0


    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.valNum:
            self.__cur_test_id = 0
            return None, None, None, None, None
        elif self.__cur_test_id + batch_size >= self.valNum:
            batch_size = self.valNum - self.__cur_test_id

        batch_grd = np.zeros([batch_size, self.panoRows, self.panoCols, 3], dtype = np.float32)
        batch_sat = np.zeros([batch_size, self.satSize, self.satSize, 3], dtype=np.float32)
        batch_grd_yawpitch = np.zeros([batch_size, self.panoRows, self.panoCols, 2], dtype=np.float32)
        batch_sat_yaw = np.zeros([batch_size, self.satSize, self.satSize, 2], dtype=np.float32)

        # the utm coordinates are used to define the positive sample and negative samples
        batch_utm = np.zeros([batch_size, 2], dtype=np.float32)
        batch_dis_utm = np.zeros([batch_size, batch_size,1], dtype=np.float32)
        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # satellite
            img = cv2.imread(self.valList[img_idx][4])
            img = cv2.resize(img, (self.satSize, self.satSize), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # normalize it to -1 --- 1
            img = img / 255.0
            img = img * 2.0 - 1.0
            batch_sat[i, :, :, :] = img

            # ground
            img = cv2.imread(self.valList[img_idx][1])
            start = int((832 - self.panoCropPixels) / 2)
            img = img[start: start + self.panoCropPixels, :, :]
            img = cv2.resize(img, (self.panoCols, self.panoRows), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # normalize it to -1 --- 1
            img = img / 255.0
            img = img * 2.0 - 1.0
            batch_grd[i, :, :, :] = img

            # orientation of ground, normilze to [-1 1]
            img = self.yaw_pitch_grd['orient_mat'][:, :, 0].astype(np.float32) / np.pi
            img = img[start: start + self.panoCropPixels, :]
            img = cv2.resize(img, (self.panoCols, self.panoRows), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            batch_grd_yawpitch[i, :, :, 0] = img

            img = self.yaw_pitch_grd['orient_mat'][:, :, 1].astype(np.float32) / np.pi
            img = img[start: start + self.panoCropPixels, :]
            img = cv2.resize(img, (self.panoCols, self.panoRows), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            batch_grd_yawpitch[i, :, :, 1] = img

            # orientation of aerial

            batch_sat_yaw[i, :, :, 0] = cv2.resize(self.yaw_sat['polor_mat'][:,:,0].astype(np.float32) / np.pi,
                                                   (self.satSize, self.satSize),
                                                   interpolation=cv2.INTER_AREA)

            batch_sat_yaw[i, :, :, 1] = cv2.resize((self.yaw_sat['polor_mat'][:,:,1].astype(np.float32) - 0.5)*2.0,
                                                   (self.satSize, self.satSize), interpolation=cv2.INTER_AREA)


            batch_utm[i,0] = self.valUTM[0, img_idx]
            batch_utm[i, 1] = self.valUTM[1, img_idx]


        self.__cur_test_id += batch_size

        # compute the batch gps distance
        for ih in range(batch_size):
            for jh in range(batch_size):
                batch_dis_utm[ih,jh,0] = (batch_utm[ih,0] - batch_utm[jh,0])*(batch_utm[ih,0] - batch_utm[jh,0]) + (batch_utm[ih, 1] - batch_utm[jh, 1]) * (batch_utm[ih, 1] - batch_utm[jh, 1])


        return batch_sat, batch_grd, batch_sat_yaw, batch_grd_yawpitch, batch_dis_utm


    def next_pair_batch(self, batch_size):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.trainIdList)

        if self.__cur_id + batch_size + 2 >= self.trainNum:
            self.__cur_id = 0
            return None, None, None, None, None

        batch_sat = np.zeros([batch_size, self.satSize, self.satSize, 3], dtype=np.float32)
        batch_grd = np.zeros([batch_size, self.panoRows, self.panoCols, 3], dtype=np.float32)
        batch_grd_yawpitch = np.zeros([batch_size, self.panoRows, self.panoCols, 2], dtype=np.float32)
        batch_sat_yaw = np.zeros([batch_size, self.satSize, self.satSize, 2], dtype=np.float32)

        # the utm coordinates are used to define the positive sample and negative samples
        batch_utm = np.zeros([batch_size, 2], dtype=np.float32)
        batch_dis_utm = np.zeros([batch_size, batch_size, 1], dtype=np.float32)

        i = 0
        batch_idx = 0
        while True:
            if batch_idx >= batch_size or self.__cur_id + i >= self.trainNum:
                break

            img_idx = self.trainIdList[self.__cur_id + i]
            i += 1

            # satellite
            img = cv2.imread(self.trainList[img_idx][4])
            # img_semantic = cv2.imread(self.trainList[img_idx][5])
            img_yaw = self.yaw_sat['polor_mat'][:, :, 0].astype(np.float32) / np.pi
            img_radis = (self.yaw_sat['polor_mat'][:, :, 1].astype(np.float32) - 0.5) * 2.0

            if img is None or img.shape[0] != img.shape[1]:
                # print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.trainList[img_idx][4], i))
                continue
            rand_crop = random.randint(1, 1198)
            if rand_crop > self.satSize:
                start = int((1200 - rand_crop) / 2)
                img = img[start : start + rand_crop, start : start + rand_crop, :]
                img_yaw = img_yaw[start: start + rand_crop, start: start + rand_crop]
                img_radis = img_radis[start: start + rand_crop, start: start + rand_crop]

            img = cv2.resize(img, (self.satSize, self.satSize), interpolation=cv2.INTER_AREA)
            img_yaw = cv2.resize(img_yaw, (self.satSize, self.satSize), interpolation=cv2.INTER_AREA)
            img_radis = cv2.resize(img_radis, (self.satSize, self.satSize), interpolation=cv2.INTER_AREA)


            rand_rotate = random.randint(0, 4) * 90
            rot_matrix = cv2.getRotationMatrix2D((self.satSize/2, self.satSize/2), rand_rotate, 1)
            img = cv2.warpAffine(img, rot_matrix, (self.satSize, self.satSize))
            img = img.astype(np.float32)

            img_yaw = cv2.warpAffine(img_yaw, rot_matrix, (self.satSize, self.satSize))
            img_radis = cv2.warpAffine(img_radis, rot_matrix, (self.satSize, self.satSize))

            # normalize it to -1 --- 1
            img = img / 255.0
            img = img * 2.0 - 1.0

            batch_sat[batch_idx, :, :, :] = img
            batch_sat_yaw[batch_idx, :, :, 0] = img_yaw
            batch_sat_yaw[batch_idx, :, :, 1] = img_radis

            # ground
            img = cv2.imread(self.trainList[img_idx][1])

            if img is None:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.trainList[img_idx][1], i))
                continue
            start = int((832 - self.panoCropPixels) / 2)
            img = img[start: start + self.panoCropPixels, :, :]

            # generate a random crop and past step
            shift_pixes = int(np.random.uniform(0,noise_level/360.0*(1664.0-1.0)))

            img_dup = img.copy()
            img_dup[:,0:1664-shift_pixes,:] =  img[:,shift_pixes:1664,:]
            img_dup[:,1664-shift_pixes:1664,:] =  img[:,0:shift_pixes,:]

            img_dup = cv2.resize(img_dup, (self.panoCols, self.panoRows), interpolation=cv2.INTER_AREA)
            img_dup = img_dup.astype(np.float32)
            # normalize it to -1 --- 1
            img_dup = img_dup / 255.0
            img_dup = img_dup * 2.0 - 1.0
            batch_grd[batch_idx, :, :, :] = img_dup
            batch_grd[batch_idx, :, :, :] = img_dup


            # orientation of ground, normilze to [-1 1]
            img = self.yaw_pitch_grd['orient_mat'][:, :, 0].astype(np.float32) / np.pi
            img = img[start: start + self.panoCropPixels, :]
            img_dup = img.copy()
            img_dup[:,0:1664-shift_pixes] =  img[:,shift_pixes:1664]
            img_dup[:,1664-shift_pixes:1664] =  img[:,0:shift_pixes]

            img_dup = cv2.resize(img_dup, (self.panoCols, self.panoRows), interpolation=cv2.INTER_AREA)
            img_dup = img_dup.astype(np.float32)
            batch_grd_yawpitch[batch_idx, :, :, 0] = img_dup

            img = self.yaw_pitch_grd['orient_mat'][:, :, 1].astype(np.float32) / np.pi
            img = img[start: start + self.panoCropPixels, :]
            img_dup = img.copy()
            img_dup[:,0:1664-shift_pixes] =  img[:,shift_pixes:1664]
            img_dup[:,1664-shift_pixes:1664] =  img[:,0:shift_pixes]

            img_dup = cv2.resize(img_dup, (self.panoCols, self.panoRows), interpolation=cv2.INTER_AREA)
            img_dup = img_dup.astype(np.float32)
            batch_grd_yawpitch[batch_idx, :, :, 1] = img_dup

            batch_utm[batch_idx, 0] = self.trainUTM[0, img_idx]
            batch_utm[batch_idx, 1] = self.trainUTM[1, img_idx]

            batch_idx += 1
        # compute the batch gps distance
        for ih in range(batch_size):
            for jh in range(batch_size):
                batch_dis_utm[ih,jh,0] = (batch_utm[ih,0] - batch_utm[jh,0])*(batch_utm[ih,0] - batch_utm[jh,0]) + (batch_utm[ih, 1] - batch_utm[jh, 1]) * (batch_utm[ih, 1] - batch_utm[jh, 1])



        self.__cur_id += i

        return batch_sat, batch_grd, batch_sat_yaw, batch_grd_yawpitch, batch_dis_utm


    def get_dataset_size(self):
        return self.trainNum
    #
    def get_test_dataset_size(self):
        return self.valNum
    #
    def reset_scan(self):
        self.__cur_test_id = 0


if __name__ == '__main__':
    input_data = InputData()
    batch_sat, batch_grd,batch_sat_ori, batch_grd_ori, batch_utm = input_data.next_batch_scan(12)
