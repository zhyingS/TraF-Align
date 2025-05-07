
import torch
from torch import nn
import torch.nn.functional as F
import copy
import numba

from models.ops.iou3d_nms import iou3d_nms_cuda


class CenterHead(nn.Module):

    def __init__(
            self,
            cfg,
            init_bias=-2.19,
            share_conv_channel=64,
            num_hm_conv=2,
            rectifier=[[0.], [0.], [0.]]):

        super(CenterHead, self).__init__()
        self.cfg = cfg
        kwargs = cfg['model']['head']
        in_channels = kwargs['in_channels']
        tasks = cfg['dataset']['cls_group']
        
        common_heads = kwargs['common_heads']
        strides = kwargs['strides']
        
        rectifier = kwargs['rectifier']
        self.rectifier = rectifier
        init_bias=kwargs['init_bias']
        num_classes = [len(t) for t in tasks]
        self.class_names = tasks

        self.in_channels = in_channels
        self.num_classes = num_classes

        if common_heads is None:
            common_heads={}

        try:
            upstride = cfg['model']['head']['upstride']
        except:
            upstride = 1
            
        # a shared convolution
        if upstride >= 1:
            self.shared_conv = nn.Sequential(
                nn.Conv2d(in_channels, share_conv_channel,
                        kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(share_conv_channel),
                nn.ReLU(inplace=True)
            )
        else:
            self.shared_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, share_conv_channel,
                        kernel_size=2, stride=int(1/upstride), bias=False),
                nn.BatchNorm2d(share_conv_channel, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        
        self.tasks = nn.ModuleList()

        for (num_cls, stride) in zip(num_classes, strides):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SepHead(share_conv_channel, heads, stride=stride,
                        bn=True, init_bias=init_bias, final_kernel=3)
            )       

        self.init_weight()
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                self.init_weight_(m)
            if isinstance(m,(nn.ModuleList,nn.Sequential)):
                m.apply(self.init_weight_)

    def init_weight_(self,m):
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
                
    def forward(self, x):
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))
       
        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def norm_exp(self,map):
        map = F.relu(map)
        mu,sigma=0,2
        map=torch.exp(-(map-mu)**2/2/sigma**2)
        map = torch.clamp(map, min=1e-4, max=1-1e-4)

        return map

def predict(preds_dicts_, cfg):
    """decode, nms, then return the detection result. Additionaly support double flip testing
    """
    # get loss info
    rets = []
    metas = []
    # test_cfg = cfg['post_processing']
    try:
        post_center_range = cfg['dataset']['eval_range'] 
    except:
        post_center_range = cfg['voxelization']['lidar_range'] 

    if len(post_center_range) > 0:
        post_center_range = torch.tensor(
            post_center_range,
            dtype=preds_dicts_[0]['hm'].dtype,
            device=preds_dicts_[0]['hm'].device,
        )
        
    for task_id, preds_dict_ in enumerate(preds_dicts_):
        # convert N C H W to N H W C
        preds_dict = {}
        for key, val in preds_dict_.items():
            preds_dict.update({key:val.permute(0, 2, 3, 1).contiguous()})
            # preds_dict[key] = val.permute(0, 2, 3, 1)

        batch_size = preds_dict['hm'].shape[0]

        meta_list = [None] * batch_size
    
        batch_hm = torch.sigmoid(preds_dict['hm'])
        
        # batch_hm = preds_dict['hm']
        # plt.imshow(batch_hm[0][:,:,0].cpu())
        # plt.show(block=True)
        batch_dim = torch.exp(preds_dict['dim']).clone()

        batch_rots = preds_dict['rot'][..., 0:1]
        batch_rotc = preds_dict['rot'][..., 1:2]
        batch_reg = preds_dict['reg']
        batch_hei = preds_dict['height']
        if 'iou' in preds_dict.keys():
            batch_iou = (preds_dict['iou'].squeeze(dim=-1) + 1) * 0.5
            batch_iou = batch_iou.type_as(batch_dim)
        else:
            batch_iou = torch.ones((batch_hm.shape[0], batch_hm.shape[1], batch_hm.shape[2]),
                                    dtype=batch_dim.dtype).to(batch_hm.device)

        batch_rot = torch.atan2(batch_rots, batch_rotc)
        batch, H, W, num_cls = batch_hm.size()
        
        batch_reg = batch_reg.reshape(batch, H*W, 2)
        batch_hei = batch_hei.reshape(batch, H*W, 1)

        batch_rot = batch_rot.reshape(batch, H*W, 1)
        batch_dim = batch_dim.reshape(batch, H*W, 3)
        batch_hm = batch_hm.reshape(batch, H*W, num_cls)
        # batch_cls = torch.zeros_like(batch_hm)
        # batch_cls[:,:,:]=task_id
        
        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        
        xs_, ys_ = [], []
        for _ in range(batch):
            xs_.append(xs.view(1,H,W))
            ys_.append(ys.view(1,H,W))
        xs = torch.cat(xs_,dim=0).to(batch_hm.device).float()
        ys = torch.cat(ys_,dim=0).to(batch_hm.device).float()

        xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
        ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

        out_size_factor = cfg['model']['backbone']['out_size_factor']
        vs = cfg['voxelization']['voxel_size']
        lr = cfg['voxelization']['lidar_range']
        xs = xs * out_size_factor[task_id] * vs[0] + lr[0]
        ys = ys * out_size_factor[task_id] * vs[1] + lr[1]

        batch_box_preds = torch.cat(
            [xs, ys, batch_hei, batch_dim, batch_rot], dim=2)
        metas.append(meta_list)

        rets.append(post_processing(task_id, batch_box_preds,
                    batch_hm, cfg, post_center_range, batch_iou))
    
    # Merge branches results
    ret_list = []
    num_samples = len(rets[0])

    tasks = cfg['dataset']['cls_group']
    num_classes = [len(t) for t in tasks]
    
    for i in range(num_samples):
        ret = {}
        for k in rets[0][i].keys():
            if k in ["box3d_lidar", "scores"]:
                ret[k] = torch.cat([ret[i][k] for ret in rets])
            elif k in ["label_preds"]:
                flag = 0
                for j, num_class in enumerate(num_classes):
                    rets[j][i][k] += flag
                    flag += num_class
                ret[k] = torch.cat([ret[i][k] for ret in rets])

        ret['token'] = metas[0][i]
        ret_list.append(ret)

    return ret_list

def post_processing(task_id, batch_box_preds, batch_hm, cfg, post_center_range, batch_iou):
    batch_size = len(batch_hm)
    prediction_dicts = []
    test_cfg = cfg['post_processing']
    rectifier_ = cfg['model']['head']['rectifier']
    
    for i in range(batch_size):
        box_preds = batch_box_preds[i]
        hm_preds = batch_hm[i]
        iou_preds = batch_iou[i].view(-1)
        scores, labels = torch.max(hm_preds, dim=-1)
        distance_mask = (box_preds[..., :2] >= post_center_range[:2]).all(1) \
            & (box_preds[..., :2] <= post_center_range[3:5]).all(1)
        score_mask = scores > test_cfg['score_threshold']
        
        mask = distance_mask & score_mask

        box_preds = box_preds[mask]
        scores = scores[mask]
        labels = labels[mask]
        iou_preds = torch.clamp(iou_preds[mask], min=0., max=1.)
        rectifier = torch.tensor(rectifier_[task_id]).to(hm_preds)
        scores = torch.pow(
            scores, 1-rectifier[labels]) * torch.pow(iou_preds, rectifier[labels])
        selected_boxes = torch.zeros((0, box_preds.shape[1])).to(box_preds)
        selected_labels = torch.zeros((0,), dtype=torch.int64).to(labels)
        selected_scores = torch.zeros((0,)).to(scores)

        for class_id in range(hm_preds.shape[-1]):
            # if isinstance(test_cfg['score_threshold'][task_id],int):
            #     score_thre = test_cfg['score_threshold'][task_id]
            # else:
            #     score_thre = test_cfg['score_threshold'][task_id][class_id]

            # score_mask = scores > score_thre

            scores_class = scores[labels == class_id]
            labels_class = labels[labels == class_id]
            box_preds_class = box_preds[labels == class_id]
            boxes_for_nms_class = box_preds_class[:, [
                0, 1, 2, 5, 4, 3, -1]] # hwl -> lwh

            selected = rotate_nms_pcdet(boxes_for_nms_class, scores_class,
                    thresh=test_cfg['nms']['nms_iou_threshold'][task_id][class_id],
                    pre_maxsize=test_cfg['nms']['nms_pre_max_size'],
                    post_max_size=test_cfg['nms']['nms_post_max_size'])

            selected_boxes = torch.cat(
                (selected_boxes, box_preds_class[selected]), dim=0)
            selected_scores = torch.cat(
                (selected_scores, scores_class[selected]), dim=0)
            selected_labels = torch.cat(
                (selected_labels, labels_class[selected]), dim=0)

        prediction_dict = {
            'box3d_lidar': selected_boxes,
            'scores': selected_scores,
            'label_preds': selected_labels
        }
        prediction_dicts.append(prediction_dict)

    return prediction_dicts

class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        stride=1,
        head_conv=64,
        final_kernel=1,
        bn=True,
        init_bias=-0,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)
        if stride > 1:
            self.deblock = ConvBlock(in_channels, head_conv, kernel_size=int(stride), 
                            stride=int(stride), padding=0, conv_layer=nn.ConvTranspose2d)
        else:
            self.deblock = nn.Identity()
        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = nn.Sequential()
            for i in range(num_conv-1):
                layer=(nn.Conv2d(in_channels, head_conv,
                                    kernel_size=final_kernel, stride=1,
                                    padding=final_kernel // 2, bias=True))
                fc.add_module('conv_{}'.format(i),layer)
                if bn:
                    fc.add_module('norm_{}'.format(i),nn.BatchNorm2d(head_conv))
                fc.add_module('relu_{}'.format(i),nn.ReLU())

            fc.add_module('conv_{}'.format(i+1),nn.Conv2d(head_conv, classes,
                                kernel_size=final_kernel,  stride=1,
                                padding=final_kernel // 2, bias=True))

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            self.__setattr__(head, fc)
                        
        self.init_weight()
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                self.init_weight_(m)
            if isinstance(m,(nn.ModuleList,nn.Sequential)):
                m.apply(self.init_weight_)

    def init_weight_(self,m):
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            
    def forward(self, x):

        x=self.deblock(x)
        import matplotlib.pyplot as plt

        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


def rotate_nms_pcdet(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """
    :param boxes: (N, 7) [x, y, z, size_x, size_y, size_z, theta]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # transform back to pcdet's coordinate
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))

    if len(boxes) == 0:
        num_out = 0
    else:
        num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)

    selected = order[keep[:num_out].cuda()].contiguous()

    if post_max_size is not None:
        selected = selected[:post_max_size]

    return selected

class Conv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, 
                 conv_layer=nn.Conv2d, bias=False, **kwargs):
        super(Conv, self).__init__()
        padding = kwargs.get('padding', kernel_size // 2)  # dafault same size

        self.conv = conv_layer(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)
                        
    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1,
                 conv_layer=nn.Conv2d,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU, **kwargs):
        super(ConvBlock, self).__init__()
        padding = kwargs.get('padding', kernel_size // 2)  # dafault same size

        self.conv = Conv(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False, conv_layer=conv_layer)

        self.norm = norm_layer(planes)
        self.act = act_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.block1 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.block2 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = out + identity
        out = self.act(out)

        return out
    