import torch
from torch.utils import data
from PIL import Image
from scipy import ndimage
import numpy as np
import os
import pickle
from main_code.data_lib.data_utils import *
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import sys

class COCO_DATASET(data.Dataset):

    def __init__(self,config,mode,data_aug=False):
        super(COCO_DATASET, self).__init__()

        assert mode in ['train','val']
        self.config = config
        self.mode = mode
        self.data_aug = data_aug
        self.data_aug_flip = self.data_aug
        self.data_aug_scales = self.config.data_aug_scales
        # data prepare param
        self.fg_thresh = 0.03


        if self.mode =='train':
            self.coco_data = COCO(self.config.train_anno_file)
            self.image_path = self.config.train_image_path
        elif self.mode=='val':
            self.coco_data = COCO(self.config.val_anno_file)
            self.image_path = self.config.val_image_path


        a = os.path.abspath('.')

        print(a)

        if os.path.exists('data_lib/cache/{}_annos.pkl'.format(self.mode)):
            self.annos = pickle.load(open('data_lib/cache/{}_annos.pkl'.format(self.mode), 'rb'))

        else:
            print('cache did not exist')
            if not os.path.exists('data_lib/cache'):
                os.makedirs('data_lib/cache')
            self.annos = self.prefilter(self.coco_data)

            pickle.dump(self.annos, open('data_lib/cache/{}_annos.pkl'.format(self.mode), 'wb'), pickle.HIGHEST_PROTOCOL)


        self.dilate_structure = get_dilate_structure(5)
        self.size = self.config.im_size
        self.mean_value = self.config.mean_value
        self.scale_value = self.config.scale_value
        self.guide_size = (224, 224)



        self.len = len(self.annos)

        print('data init done')


    def __getitem__(self, index):

        if self.data_aug_scales:
            scale = random.choice(self.data_aug_scales)
            new_size = (int(self.size[0] * scale), int(self.size[1] * scale))
        else:
            new_size = self.size

        anno = self.annos[index]
        image_path = self.image_path.format(anno['image_id'])
        image = Image.open(image_path)
        label_data = self.coco_data.annToMask(anno).astype(np.uint8)
        label = Image.fromarray(label_data)

        guide_image = image.crop(anno['bbox'])
        guide_label = label.crop(anno['bbox'])
        guide_image, guide_label = data_augmentation(guide_image, guide_label,
                                                     self.guide_size, data_aug_flip=self.data_aug_flip,
                                                     keep_aspect_ratio=self.config.vg_keep_aspect_ratio,
                                                     random_crop_ratio=self.config.vg_random_crop_ratio,
                                                     random_rotate_angle=self.config.vg_random_rotate_angle,
                                                     color_aug=self.config.vg_color_aug)

        image, label = data_augmentation(image, label,
                                         new_size, data_aug_flip=self.data_aug_flip,
                                         random_crop_ratio=self.config.random_crop_ratio)
        image_data = np.array(image, dtype=np.float32)
        label_data = np.array(label, dtype=np.float32)
        guide_image_data = np.array(guide_image, dtype=np.float32)
        guide_label_data = np.array(guide_label, dtype=np.uint8)
        if self.config.use_original_mask:
            gb_image = perturb_mask(label_data)
            gb_image = ndimage.morphology.binary_dilation(gb_image,
                                                          structure=self.dilate_structure) * 255
        else:
            gb_image = get_gb_image(label_data, center_perturb=self.config.sg_center_perturb_ratio,
                                    std_perturb=self.config.sg_std_perturb_ratio)
        image_data = to_bgr(image_data)
        guide_image_data = to_bgr(guide_image_data)
        image_data -= self.mean_value
        guide_image_data -= self.mean_value
        # masking
        if not self.config.bbox_sup:
            guide_image_data = mask_image(guide_image_data, guide_label_data)



        image_data = torch.from_numpy(np.transpose(image_data,(2,0,1)).copy()).float()
        label = torch.from_numpy(label_data[np.newaxis,...].copy()).float()
        vg_image = torch.from_numpy(np.transpose(guide_image_data,(2,0,1)).copy()).float()
        gb_image = torch.from_numpy(gb_image[np.newaxis,...].copy()).float()

        return image_data, label, vg_image, gb_image


    def prefilter(self, dataset):
        res_annos = []
        annos = dataset.dataset['annotations']
        for anno in annos:
            # throw away all crowd annotations
            if anno['iscrowd']: continue

            m = dataset.annToMask(anno)
            mask_area = np.count_nonzero(m)
            if mask_area / float(m.shape[0] * m.shape[1]) > self.fg_thresh:
                anno['bbox'] = get_mask_bbox(m)
                res_annos.append(anno)
        return res_annos

    def __len__(self):

        return self.len



if __name__ == '__main__':

    # check dataload module
    from main_code.config import COCO_CONFIG

    wpath = os.getcwd()
    os.chdir(wpath.replace(wpath.split('/')[-1],''))

    config = COCO_CONFIG()

    dataset = COCO_DATASET(mode='val',config=config)

    data_iter = data.DataLoader(dataset,
                                batch_size=1,
                                drop_last=False,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True)
    data_iter = iter(data_iter)

    for image_data, label, vg_image, gb_image in data_iter:

        plt.figure(figsize=(30,30))
        plt.subplot(221)
        plt.imshow(image_data[0,0,:,:])
        plt.imshow(label[0,0,:,:],alpha=0.5)

        plt.subplot(222)
        plt.imshow(image_data[0, 0, :, :])
        plt.imshow(gb_image[0,0,:,:],alpha=0.5)

        plt.subplot(223)
        plt.imshow(vg_image[0,0,:,:])

        plt.show()


