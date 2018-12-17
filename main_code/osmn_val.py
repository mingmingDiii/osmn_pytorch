import torch
from torch.utils import data
from torch import nn
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from main_code.models import osmn_net,osmn_net0
from main_code.config import DAVIS_CONFIG,COCO_CONFIG
from main_code.data_lib import dataset_davis
from main_code.data_lib import dataset_coco
from main_code.losses import class_balanced_cross_entropy_loss
import os
from tqdm import tqdm
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
# def build_network(snapshot, backend):
#     epoch = 0
#     backend = backend.lower()
#     net = models[backend]()
#     net = nn.DataParallel(net)
#     if snapshot is not None:
#         epoch = os.path.basename(snapshot).split('_')[-1]
#         epoch = int(epoch)+1
#         net.load_state_dict(torch.load(snapshot))
#
#     net = net.cuda()
#     return net, epoch







os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
def m_eval(dataset_name='davis'):


    assert dataset_name in ['coco','davis']

    #
    if dataset_name=='coco':
        config = COCO_CONFIG()
        dataset = dataset_coco.COCO_DATASET(mode='val', config=config,data_aug=True)
    elif dataset_name=='davis':
        config = DAVIS_CONFIG()
        dataset = dataset_davis.DAVIS_DATASET(mode='val',config=config,data_aug=True)
    else:
        config = None
        dataset = None

    result_path = config.result_path


    # get data loader
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True)


    # get net
    net = osmn_net0.OSMN_NET()
    net = nn.DataParallel(net)

    snapshot = '/mnt/sda1/don/documents/public/osmn_pytorch/main_code/logs/osmn/weights/osmn_9'
    net.load_state_dict(torch.load(snapshot))
    net = net.cuda()

    with torch.no_grad():
        val_iterator = tqdm(data_loader, total=len(data_loader))
        net.eval()

        for image, vg_image, gb_image,ref_name in val_iterator:
            image = Variable(image).cuda()
            vg_image = Variable(vg_image).cuda()
            gb_image = Variable(gb_image).cuda()



            logit = net(image, vg_image, gb_image)
            pred = torch.sigmoid(logit)
            pred = pred.data.cpu().numpy()[0, 0, ...]

            pred = np.where(pred>0.5,1.0,0.0)

            # plt.imshow(pred)
            # plt.show()

            save_path = os.path.join(result_path, *(ref_name[0].split('/')[:-1]))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            misc.imsave(os.path.join(result_path, ref_name[0]), pred)







if __name__ == '__main__':

    m_eval()











