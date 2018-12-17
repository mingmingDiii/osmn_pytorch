

class DAVIS_CONFIG(object):

    data_path = '/mnt/sda1/don/documents/project_paper/video_seg/data/DAVIS-2017-trainval-480p'
    #Path to save predicted results
    result_path = '/mnt/sda1/don/documents/video_seg/code_ref/video_seg/results'



    #Whether to modulate conv1 and conv2 in VGG model
    mod_early_conv = False

    #
    no_spatial_modulator = False

    no_visual_modulator = False

    fix_bn = False

    aligned_size = None

    fix_seg = True

    bbox_sup = False

    use_original_mask = False

    crf_postprocessing = False

    random_crop_ratio = 0.0

    vg_keep_aspect_ratio = True

    vg_color_aug =False

    vg_random_crop_ratio = 0.1

    vg_pad_ratio = 0.03

    vg_random_rotate_angle = 10

    sg_center_perturb_ratio = 0.2

    sg_std_perturb_ratio = 0.4

    mean_value = [0,0,0]#[104,117,123]

    scale_value = 1.0

    batch_size = 1

    num_loader = 4

    save_score = False

    gpu_id = 0

    training_iters = 200000

    save_iters = 1000

    learning_rate = 1e-5

    epochs  = 10



    data_version = 2017

    use_full_res = False

    randomize_guide = False

    label_valid_ratio = 0.003

    bbox_valid_ratio = 0.2

    test_split = 'val'

    im_size = [854, 480]

    data_aug_scales = [1]


class COCO_CONFIG(object):

    train_image_path = '/mnt/sda1/don/datac/coco2017/images/train2017/{:012d}.jpg'
    val_image_path = '/mnt/sda1/don/datac/coco2017/images/val2017/{:012d}.jpg'

    train_anno_file = '/mnt/sda1/don/datac/coco2017/annotations_trainval2017/annotations/instances_train2017.json'

    val_anno_file = '/mnt/sda1/don/datac/coco2017/annotations_trainval2017/annotations/instances_val2017.json'


    # Path to save predicted results
    result_path = '/mnt/sda1/don/documents/public/osmn_pytorch/main_code/results/'



    # Whether to modulate conv1 and conv2 in VGG model
    mod_early_conv = False

    #
    no_spatial_modulator = False

    no_visual_modulator = False

    fix_bn = False

    aligned_size = None

    fix_seg = True

    bbox_sup = False

    use_original_mask = False

    crf_postprocessing = False

    random_crop_ratio = 0.0

    vg_keep_aspect_ratio = True

    vg_color_aug = False

    vg_random_crop_ratio = 0.1

    vg_pad_ratio = 0.03

    vg_random_rotate_angle = 10

    sg_center_perturb_ratio = 0.2

    sg_std_perturb_ratio = 0.4

    mean_value = [0, 0, 0]  # [104,117,123]

    scale_value = 1.0

    batch_size = 1

    num_loader = 4

    save_score = False

    gpu_id = 0

    training_iters = 200000

    save_iters = 1000

    learning_rate = 1e-5

    epochs = 10




    use_full_res = False

    randomize_guide = False

    label_valid_ratio = 0.003

    bbox_valid_ratio = 0.2



    im_size = [400, 400]

    data_aug_scales = [0.8, 1, 1.2]