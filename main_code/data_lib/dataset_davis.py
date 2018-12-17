import torch
from torch.utils import data
from PIL import Image
from scipy import ndimage
import numpy as np
import os
from main_code.data_lib.data_utils import *
import matplotlib.pyplot as plt

class DAVIS_DATASET(data.Dataset):

    def __init__(self,mode,config,data_aug=False):
        super(DAVIS_DATASET, self).__init__()

        self.config = config
        self.data_aug = data_aug
        self.data_aug_scales = self.config.data_aug_scales
        self.data_aug_flip = self.data_aug
        # data prepare param
        self.baseDir = self.config.data_path
        self.data_version = self.config.data_version
        self.randomize_guide = self.config.randomize_guide

        train_path = os.path.join(self.baseDir, 'ImageSets/%d/train.txt' % self.data_version)
        val_path = os.path.join(self.baseDir, 'ImageSets/%d/%s.txt' % (self.data_version, self.config.test_split))
        with open(val_path, 'r') as f:
            self.val_seq_names = [line.strip() for line in f]
        with open(train_path, 'r') as f:
            self.train_seq_names = [line.strip() for line in f]

        if self.config.use_full_res:
            self.baseDirImg = os.path.join(self.baseDir, 'JPEGImages', 'Full-Resolution')
        else:
            self.baseDirImg = os.path.join(self.baseDir, 'JPEGImages', '480p')
        label_fd = '480p_split' if self.data_version == 2017 else '480p_all'
        self.baseDirLabel = os.path.join(self.baseDir, 'Annotations', label_fd)
        self.resDirLabel = self.config.result_path


        #

        self.multiclass = hasattr(self.config, 'data_version') and self.config.data_version == 2017 \
                or hasattr(self.config, 'multiclass') and self.config.multiclass
        self.size = self.config.im_size
        self.mean_value = self.config.mean_value #np.array((104, 117, 123))
        self.scale_value = self.config.scale_value # 0.00787 for mobilenet
        self.guide_size = (224, 224)



        # get img_list

        self.img_list = self._get_img_list()

        self.len = len(self.img_list)


    def __getitem__(self, index):

        if self.config.data_aug_scales:
            scale = random.choice(self.config.data_aug_scales)
            new_size = (int(self.size[0] * scale), int(self.size[1] * scale))

        sample = self.img_list[index]
        if len(sample) == 4:
            # guide image is both for appearance and location guidance
            guide_image = Image.open(sample[0])
            guide_label = Image.open(sample[1])
            image = Image.open(sample[2])
            label = Image.open(sample[3])
            ref_label = guide_label
        else:
            # guide image is only for appearance guidance, ref label is only for location guidance
            guide_image = Image.open(sample[0])
            guide_label = Image.open(sample[1])
            # guide_image = Image.open(sample[2])
            ref_label = Image.open(sample[2])
            image = Image.open(sample[3])
            label = Image.open(sample[4])
        if len(sample) > 5:
            label_id = sample[5]
        else:
            label_id = 0
        image = image.resize(new_size, Image.BILINEAR)
        label = label.resize(new_size, Image.NEAREST)
        ref_label = ref_label.resize(new_size, Image.NEAREST)
        guide_label = guide_label.resize(guide_image.size, Image.NEAREST)
        if label_id > 0:
            guide_label = self._get_obj_mask(guide_label, label_id)
            ref_label = self._get_obj_mask(ref_label, label_id)
            label = self._get_obj_mask(label, label_id)
        guide_label_data = np.array(guide_label)
        bbox = get_mask_bbox(guide_label_data)
        guide_image = guide_image.crop(bbox)
        guide_label = guide_label.crop(bbox)
        guide_image, guide_label = data_augmentation(guide_image, guide_label,
                                                     self.guide_size, data_aug_flip=self.data_aug_flip,
                                                     keep_aspect_ratio=self.config.vg_keep_aspect_ratio,
                                                     random_crop_ratio=self.config.vg_random_crop_ratio,
                                                     random_rotate_angle=self.config.vg_random_rotate_angle,
                                                     color_aug=self.config.vg_color_aug)
        if not self.config.use_original_mask:
            gb_image = get_gb_image(np.array(ref_label), center_perturb=self.config.sg_center_perturb_ratio,
                                    std_perturb=self.config.sg_std_perturb_ratio)
        else:
            gb_image = perturb_mask(np.array(ref_label))
            gb_image = ndimage.morphology.binary_dilation(gb_image,
                                                          structure=self.config.dilate_structure) * 255
        image_data = np.array(image, dtype=np.float32)
        label_data = (np.array(label, dtype=np.uint8) > 0)*1.0
        image_data = to_bgr(image_data)
        image_data = (image_data - self.config.mean_value) * self.config.scale_value
        guide_label_data = np.array(guide_label, dtype=np.uint8)
        guide_image_data = np.array(guide_image, dtype=np.float32)
        guide_image_data = to_bgr(guide_image_data)
        guide_image_data = (guide_image_data - self.config.mean_value) * self.config.scale_value
        guide_image_data = mask_image(guide_image_data, guide_label_data)






        image_data = torch.from_numpy(np.transpose(image_data,(2,0,1))).float()
        label = torch.from_numpy(label_data[np.newaxis,...]).float()
        vg_image = torch.from_numpy(np.transpose(guide_image_data,(2,0,1))).float()
        gb_image = torch.from_numpy(gb_image[np.newaxis,...]).float()

        return image_data, label, vg_image, gb_image




    def __len__(self):

        return self.len

    def _get_obj_mask(self,image, idx):
        return Image.fromarray((np.array(image) == idx).astype(np.uint8))

    def _get_img_list(self):
        train_imgs_with_guide = []
        for name in self.train_seq_names:
            train_frames = sorted(os.listdir(os.path.join(self.baseDirImg, name)))
            label_fds = os.listdir(os.path.join(self.baseDirLabel, name)) if self.data_version == 2017 else \
                [os.path.join(self.baseDirLabel, name)]
            for label_id in label_fds:
                # each sample: visual guide image, visual guide mask, spatial guide mask, input image, ground truth mask
                if self.randomize_guide:
                    # filter images to get good quality visual guide images
                    valid_label_idx = []
                    nonblank_label_idx = []
                    for frame in train_frames:
                        label = Image.open(os.path.join(self.baseDirLabel, name, label_id, frame[:-4] + '.png'))
                        label_data = np.array(label) > 0
                        bbox = get_mask_bbox(label_data, border_pixels=0)
                        if np.sum(label_data) > 0:
                            nonblank_label_idx.append(frame)
                        if np.sum(label_data) > label_data.size * self.config.label_valid_ratio and \
                                np.sum(label_data) > (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) * self.config.bbox_valid_ratio:
                            valid_label_idx.append(frame[:-4])
                    train_frames = nonblank_label_idx
                    if len(valid_label_idx) > 0:
                        # randomly select guide image for each frame
                        random_guide_idx = np.random.randint(0, len(valid_label_idx), (len(train_frames)))
                    else:
                        # default to use the first frame
                        valid_label_idx = [train_frames[0][:-4]]
                        random_guide_idx = np.zeros((len(train_frames)), dtype=np.int32)
                    # use random frame as visual guide and ground truth of previous frame as spatial guide
                    train_imgs_with_guide += [(os.path.join(self.baseDirImg, name, valid_label_idx[guide_id] + '.jpg'),
                                               os.path.join(self.baseDirLabel, name, label_id,
                                                            valid_label_idx[guide_id] + '.png'),
                                               os.path.join(self.baseDirLabel, name, label_id, prev_frame[:-4] + '.png'),
                                               os.path.join(self.baseDirImg, name, frame),
                                               os.path.join(self.baseDirLabel, name, label_id, frame[:-4] + '.png'))
                                              for prev_frame, frame, guide_id in
                                              zip(train_frames[:-1], train_frames[1:], random_guide_idx[1:])]

                else:
                    # use the first fram as visual guide and ground truth of previous frame as spatial guide
                    train_imgs_with_guide += [(os.path.join(self.baseDirImg, name, '00000.jpg'),
                                               os.path.join(self.baseDirLabel, name, label_id, '00000.png'),
                                               os.path.join(self.baseDirLabel, name, label_id, prev_frame[:-4] + '.png'),
                                               os.path.join(self.baseDirImg, name, frame),
                                               os.path.join(self.baseDirLabel, name, label_id, frame[:-4] + '.png'))
                                              for prev_frame, frame in zip(train_frames[:-1], train_frames[1:])]


        return train_imgs_with_guide


if __name__ == '__main__':

    from main_code.config import DAVIS_CONFIG

    config = DAVIS_CONFIG()

    dataset = DAVIS_DATASET(config=config,mode='train')

    data_iter = data.DataLoader(dataset,
                                batch_size=1,
                                drop_last=False,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True)
    data_iter = iter(data_iter)

    for image_data, label, vg_image, gb_image in data_iter:

        label = label.numpy()
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


