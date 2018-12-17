import torch
from torch import nn
from torch.nn import functional as F

from main_code.models.backbone import vgg

class OSMN_NET(nn.Module):

    def __init__(self,mod_early_conv=False):
        super(OSMN_NET, self).__init__()

        self.mod_early_conv = mod_early_conv

        self.vgg_vg = vgg.vgg16_bn(pretrained=True)
        self.vgg_seg = vgg.vgg16(pretrained=True)


        # vg_image part
        self.n_modulator_param = 512*6+256*3+mod_early_conv*384
        self.vg_features = nn.Sequential(
            self.vgg_vg.features,
            nn.Conv2d(512,4096,kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(4096,4096,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(4096,self.n_modulator_param,kernel_size=1)
        )

        # gb_image part
        self.num_mod_layers = [2,2,3,3,3]
        self.conv1_att = nn.Sequential(
            nn.Conv2d(1,64*self.num_mod_layers[0],kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv2_att = nn.Sequential(
            nn.Conv2d(1,128*self.num_mod_layers[1],kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv3_att = nn.Sequential(
            nn.Conv2d(1,256*self.num_mod_layers[2],kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv4_att = nn.Sequential(
            nn.Conv2d(1,512*self.num_mod_layers[3],kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv5_att = nn.Sequential(
            nn.Conv2d(1,512*self.num_mod_layers[4],kernel_size=1),
            nn.ReLU(inplace=True)
        )


        # segmentation part
        self.seg_block1 = modulated_conv_block(blocks=[self.vgg_seg.features[0],
                                                       self.vgg_seg.features[2]],
                                               vm = self.mod_early_conv,
                                               sm = self.mod_early_conv)
        self.seg_block2 = modulated_conv_block(blocks=[self.vgg_seg.features[5],
                                                       self.vgg_seg.features[7]],
                                               vm = self.mod_early_conv,
                                               sm = self.mod_early_conv)
        self.seg_block3 = modulated_conv_block(blocks=[self.vgg_seg.features[10],
                                                       self.vgg_seg.features[12],
                                                       self.vgg_seg.features[14]],
                                               vm = True,
                                               sm = True)
        self.seg_block4 = modulated_conv_block(blocks=[self.vgg_seg.features[17],
                                                       self.vgg_seg.features[19],
                                                       self.vgg_seg.features[21]],
                                               vm = True,
                                               sm = True)
        self.seg_block5 = modulated_conv_block(blocks=[self.vgg_seg.features[24],
                                                       self.vgg_seg.features[26],
                                                       self.vgg_seg.features[28]],
                                               vm = True,
                                               sm = True)



        # output part


        self.out_conv2 = nn.Conv2d(128,16,kernel_size=3,padding=1)
        self.out_up2 = nn.ConvTranspose2d(16,16,kernel_size=4,stride=2,bias=None)

        self.out_conv3 = nn.Conv2d(256,16,kernel_size=3,padding=1)
        self.out_up3 = nn.ConvTranspose2d(16,16,kernel_size=8,stride=4,bias=None)

        self.out_conv4 = nn.Conv2d(512,16,kernel_size=3,padding=1)
        self.out_up4 = nn.ConvTranspose2d(16,16,kernel_size=16,stride=8,bias=None)

        self.out_conv5 = nn.Conv2d(512,16,kernel_size=3,padding=1)
        self.out_up5 = nn.ConvTranspose2d(16,16,kernel_size=32,stride=16,bias=None)

        self.final_conv = nn.Conv2d(16*4,1,kernel_size=1)



    def forward(self, input_image,vg_image,gb_image):

        bs,ch,h,w = input_image.size()
        modulator_params = self.vg_features(vg_image) #512,7,7
        modulator_params = modulator_params.view(modulator_params.size(0),
                                                 modulator_params.size(1))#3840

        ds_mask = F.avg_pool2d(gb_image,2,2)
        if self.mod_early_conv:
            conv1_att = self.conv1_att(gb_image)
            conv2_att = self.conv2_att(ds_mask)
        else:
            conv1_att = None
            conv2_att = None

        ds_mask = F.avg_pool2d(ds_mask,2,2)
        conv3_att = self.conv3_att(ds_mask)
        ds_mask = F.avg_pool2d(ds_mask,2,2)
        conv4_att = self.conv4_att(ds_mask)
        ds_mask = F.avg_pool2d(ds_mask,2,2)
        conv5_att = self.conv5_att(ds_mask)

        if self.mod_early_conv:
            md_p1 = modulator_params[:,:64*2]
            md_p2 = modulator_params[:,64*2:64*2+128*2]
            md_p3 = modulator_params[:,64*2+128*2:64*2+128*2+256*3]
            md_p4 = modulator_params[:,64*2+128*2+256*3:64*2+128*2+256*3+512*3]
            md_p5 = modulator_params[:,64*2+128*2+256*3+512*3:64*2+128*2+256*3+512*6]
        else:
            md_p1 = None
            md_p2 = None
            md_p3 = modulator_params[:,:256*3]
            md_p4 = modulator_params[:,256*3:256*3+512*3]
            md_p5 = modulator_params[:,256*3+512*3:256*3+512*6]


        seg_1 = self.seg_block1(input_image,
                                md_p1,
                                conv1_att)

        seg_2 = F.max_pool2d(seg_1,2,2)
        seg_2 = self.seg_block2(seg_2,
                                md_p2,
                                conv2_att)

        seg_3 = F.max_pool2d(seg_2, 2, 2)
        seg_3 = self.seg_block3(seg_3,
                                md_p3,
                                conv3_att)

        seg_4 = F.max_pool2d(seg_3, 2, 2)
        seg_4 = self.seg_block4(seg_4,
                                md_p4,
                                conv4_att)

        seg_5 = F.max_pool2d(seg_4, 2, 2)
        seg_5 = self.seg_block5(seg_5,
                                md_p5,
                                conv5_att)

        side_2 = self.out_conv2(seg_2)
        side_2 = self.out_up2(side_2)
        side_2 = self.crop_feature(side_2,h,w)

        side_3 = self.out_conv3(seg_3)
        side_3 = self.out_up3(side_3)
        side_3 = self.crop_feature(side_3, h, w)

        side_4 = self.out_conv4(seg_4)
        side_4 = self.out_up4(side_4)
        side_4 = self.crop_feature(side_4, h, w)

        side_5 = self.out_conv5(seg_5)
        side_5 = self.out_up5(side_5)
        side_5 = self.crop_feature(side_5, h, w)

        concat_side = torch.cat([side_2,side_3,side_4,side_5],dim=1)

        logit = self.final_conv(concat_side)


        return logit



    def crop_feature(self,feature,h,w):

        _,_,oh,ow = feature.size()

        sh = (oh-h)//2
        sw = (ow-w)//2

        out_feature = feature[:,:,sh:sh+h,sw:sw+w]

        return out_feature

class modulated_conv_block(nn.Module):
    def __init__(self,blocks,vm=False,sm=False):
        super(modulated_conv_block, self).__init__()


        self.block_num = len(blocks)
        self.vm = vm
        self.sm = sm

        self.conv1 = nn.Sequential(
            blocks[0],
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            blocks[1],
            nn.ReLU(inplace=True)
        )
        if self.block_num>2:
            self.conv3 = nn.Sequential(
                blocks[2],
                nn.ReLU(inplace=True)
            )

    def forward(self, img_feature,vm_params,sm_params):

        img_feature = self.conv1(img_feature)
        if self.vm:

            img_feature = img_feature*vm_params[:,:img_feature.size(1)].view(img_feature.size(0),img_feature.size(1),
                                                       1,1)
        if self.sm:
            img_feature = img_feature+sm_params[:,:img_feature.size(1),:,:]

        img_feature = self.conv2(img_feature)
        if self.vm:
            img_feature = img_feature*vm_params[:,img_feature.size(1):2*img_feature.size(1)].view(img_feature.size(0),img_feature.size(1),
                                                       1,1)
        if self.sm:
            img_feature = img_feature+sm_params[:,img_feature.size(1):2*img_feature.size(1)]

        if self.block_num>2:
            img_feature = self.conv3(img_feature)
            if self.vm:
                img_feature = img_feature * vm_params[:, 2*img_feature.size(1):3*img_feature.size(1)].view(img_feature.size(0),img_feature.size(1),
                                                       1,1)
            if self.sm:
                img_feature = img_feature + sm_params[:, 2*img_feature.size(1):3*img_feature.size(1)]

        return img_feature














if __name__ == '__main__':

    model = OSMN_NET(mod_early_conv=False)




    vg_image = torch.ones((1,3,224,224))
    input_image = torch.ones((1,3,480,480))
    gb_image = torch.ones((1,1,480,480))




    logit = model(input_image,vg_image,gb_image)

    print(logit.size())