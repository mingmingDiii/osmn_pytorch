import torch
from torch.utils import data
from torch import nn
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from main_code.models import osmn_net
from main_code.config import DAVIS_CONFIG,COCO_CONFIG
from main_code.data_lib import dataset_davis
from main_code.data_lib import dataset_coco
from main_code.losses import class_balanced_cross_entropy_loss
import os
from tqdm import tqdm
import numpy as np
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
def m_train(dataset_name='davis',ex_='osmn'):


    assert dataset_name in ['coco','davis']

    #
    if dataset_name=='coco':
        config = COCO_CONFIG()
        dataset = dataset_coco.COCO_DATASET(mode='train', config=config,data_aug=True)
    elif dataset_name=='davis':
        config = DAVIS_CONFIG()
        dataset = dataset_davis.DAVIS_DATASET(mode='train',config=config,data_aug=True)
    else:
        config = None
        dataset = None



    ex_name = '{}_{}'.format(ex_,dataset_name)
    exp_dir = 'logs/{}/'.format(ex_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    ### weights path
    weights_path = exp_dir+'weights/'
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    ### board path
    board_path = exp_dir+'board/'
    if not os.path.exists(board_path):
        os.mkdir(board_path)
    writer = SummaryWriter(board_path)
    ### config file
    config_path = exp_dir+'config/'
    if not os.path.exists(config_path):
        os.mkdir(config_path)


    # get data loader
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)


    # get net
    net = osmn_net.OSMN_NET()
    net = nn.DataParallel(net)
    net = net.cuda()


    epochs = 20
    lr = 1e-5#config.learning_rate


    criterion = class_balanced_cross_entropy_loss()
    #criterion = nn.BCEWithLogitsLoss()
    val_loss = 0.0


    for epoch in range(epochs):
        # if epoch>=15:
        #     lr=1e-5
        # if epoch>=30:
        #     lr=1e-6

        optimizer = optim.Adam(net.parameters(), lr=lr)
        epoch_train_loss = []

        train_iterator = tqdm(data_loader, total=len(data_loader))
        net.train()
        steps = 0
        for image, label, vg_image, gb_image in train_iterator:
            image = Variable(image).cuda()
            label = Variable(label).cuda()
            vg_image = Variable(vg_image).cuda()
            gb_image = Variable(gb_image).cuda()

            optimizer.zero_grad()

            logit = net(image, vg_image, gb_image)

            train_loss = criterion(logit, label)




            epoch_train_loss.append(train_loss.data)


            status = "[{}][{:03d}]" \
                     "la = {:0.7f}," \
                     "LR = {:0.7f} " \
                     "vall = {:0.5f},".format(
                ex_name, epoch,
                np.mean(epoch_train_loss),
                lr,
                val_loss)

            train_iterator.set_description(status)

            train_loss.backward()
            optimizer.step()
            steps+=1

            if steps%1000==0:
                torch.save(net.state_dict(), weights_path + '{}_{}_{}'.format(ex_name, epoch,steps))



        torch.save(net.state_dict(), weights_path + '{}_{}'.format(ex_name,epoch))

        # with torch.no_grad():
        #     # make val
        #
        #     net.eval()
        #     val_losses = []
        #     val_losses1 = []
        #     val_losses2 = []
        #
        #     for qur_imgv, qur_maskv, sup_imgv, sup_maskv in valloader:
        #         qur_imgv = Variable(qur_imgv).cuda()
        #         qur_maskv = Variable(qur_maskv).cuda()
        #         sup_imgv = Variable(sup_imgv).cuda()
        #         sup_maskv = Variable(sup_maskv).cuda()
        #
        #         qur_pred_v, match_logit_v = net(qur_imgv, sup_imgv, sup_maskv)
        #
        #         val_loss_all1 = criterion(qur_pred_v, qur_maskv)
        #         val_loss_all2 = criterion(match_logit_v, qur_maskv)
        #
        #         val_loss_all = val_loss_all1 + val_loss_all2
        #
        #         val_losses.append(val_loss_all.data)
        #         val_losses1.append(val_loss_all1.data)
        #         val_losses2.append(val_loss_all2.data)
        #
        #     val_loss = np.mean(val_losses)
        #     val_loss1 = np.mean(val_losses1)
        #     val_loss2 = np.mean(val_losses2)


    pass




if __name__ == '__main__':

    m_train(dataset_name='coco')











