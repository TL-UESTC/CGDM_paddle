from x2paddle import torch2paddle
import paddle
import paddle.vision
from paddle.vision import transforms
from x2paddle.torch2paddle import DataLoader
import paddle.nn.functional as F
from paddle.vision import datasets
import paddle.nn as nn
from scipy.spatial.distance import cdist
import numpy as np
from data_loader import mnist
from data_loader import svhn
from data_loader import usps
from data_loader import office31
from paddle import grad
from itertools import chain
import random
import paddle
from paddle.vision.transforms import RandomCrop
from paddle.vision.transforms import RandomRotation


def digit_load(args):
    train_bs = args.batch_size
    if args.trans == 's2m':
        train_source = svhn.SVHN(args.dataset_root + '/svhn/', split=\
            'train', download=True, transform=transforms.Compose([
            transforms.Resize(32), torch2paddle.ToTensor(), torch2paddle.
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_source = svhn.SVHN(args.dataset_root + '/svhn/', split='test',
            download=True, transform=transforms.Compose([transforms.Resize(
            32), torch2paddle.ToTensor(), torch2paddle.Normalize((0.5, 0.5,
            0.5), (0.5, 0.5, 0.5))]))
        train_target = mnist.MNIST_idx(args.dataset_root + '/mnist/', train
            =True, download=True, transform=transforms.Compose([transforms.
            Resize(32), torch2paddle.Lambda(lambda x: x.convert('RGB')),
            torch2paddle.ToTensor(), torch2paddle.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))]))
        test_target = mnist.MNIST(args.dataset_root + '/mnist/', train=\
            False, download=True, transform=transforms.Compose([transforms.
            Resize(32), torch2paddle.Lambda(lambda x: x.convert('RGB')),
            torch2paddle.ToTensor(), torch2paddle.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))]))
    elif args.trans == 'u2m':
        train_source = usps.USPS(args.dataset_root + '/usps/', train=True,
            download=True, transform=transforms.Compose([RandomCrop(28,
            padding=4), RandomRotation(10), torch2paddle.ToTensor(),
            torch2paddle.Normalize((0.5,), (0.5,))]))
        test_source = usps.USPS(args.dataset_root + '/usps/', train=False,
            download=True, transform=transforms.Compose([RandomCrop(28,
            padding=4), RandomRotation(10), torch2paddle.ToTensor(),
            torch2paddle.Normalize((0.5,), (0.5,))]))
        train_target = mnist.MNIST_idx(args.dataset_root + '/mnist/', train
            =True, download=True, transform=transforms.Compose([
            torch2paddle.ToTensor(), torch2paddle.Normalize((0.5,), (0.5,))]))
        test_target = mnist.MNIST(args.dataset_root + '/mnist/', train=\
            False, download=True, transform=transforms.Compose([
            torch2paddle.ToTensor(), torch2paddle.Normalize((0.5,), (0.5,))]))
    elif args.trans == 'm2u':
        train_source = mnist.MNIST(args.dataset_root + '/mnist/', train=\
            True, download=True, transform=transforms.Compose([torch2paddle
            .ToTensor(), torch2paddle.Normalize((0.5,), (0.5,))]))
        test_source = mnist.MNIST(args.dataset_root + '/mnist/', train=\
            False, download=True, transform=transforms.Compose([
            torch2paddle.ToTensor(), torch2paddle.Normalize((0.5,), (0.5,))]))
        train_target = usps.USPS_idx(args.dataset_root + '/usps/', train=\
            True, download=True, transform=transforms.Compose([torch2paddle
            .ToTensor(), torch2paddle.Normalize((0.5,), (0.5,))]))
        test_target = usps.USPS(args.dataset_root + '/usps/', train=False,
            download=True, transform=transforms.Compose([torch2paddle.
            ToTensor(), torch2paddle.Normalize((0.5,), (0.5,))]))
    dset_loaders = {}
    dset_loaders['source_train'] = DataLoader(train_source, batch_size=\
        train_bs, shuffle=True, num_workers=args.num_workers, drop_last=False)
    dset_loaders['source_test'] = DataLoader(test_source, batch_size=\
        train_bs * 2, shuffle=True, num_workers=args.num_workers, drop_last
        =False)
    dset_loaders['target_train'] = DataLoader(train_target, batch_size=\
        train_bs, shuffle=True, num_workers=args.num_workers, drop_last=False)
    dset_loaders['target_train_no_shuff'] = DataLoader(train_target,
        batch_size=train_bs, shuffle=False, num_workers=args.num_workers,
        drop_last=False)
    dset_loaders['target_test'] = DataLoader(test_target, batch_size=\
        train_bs * 2, shuffle=False, num_workers=args.num_workers,
        drop_last=False)
    return dset_loaders


def office31_load(args):
    train_bs = args.batch_size
    source = args.trans.split('2')[0]
    target = args.trans.split('2')[1]
    dset_loaders = {}
    dset_loaders['source_train'] = office31.get_office_dataloader(source,
        train_bs, True)
    dset_loaders['source_test'] = office31.get_office_dataloader(source,
        train_bs, True)
    dset_loaders['target_train'] = office31.get_office_dataloader(target,
        train_bs, True)
    dset_loaders['target_train_no_shuff'] = office31.get_office_dataloader(
        target, train_bs, False)
    dset_loaders['target_test'] = office31.get_office_dataloader(target,
        train_bs, True)
    return dset_loaders


def init_weights_orthogonal(m):
    if type(m) == nn.Conv2d:
        torch2paddle.normal_init_(m.weight)
    if type(m) == nn.Linear:
        torch2paddle.normal_init_(m.weight)


def init_weights_xavier_normal(m):
    xavier = paddle.nn.initializer.XavierNormal()
    if type(m) == nn.Conv2d:
        xavier(m.weight)
    if type(m) == nn.Linear:
        xavier(m.weight)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def discrepancy(out1, out2):
    return torch2paddle.mean(paddle.abs(F.softmax(out1) - F.softmax(out2)))


def discrepancy_matrix(out1, out2):
    out1 = F.softmax(out1, axis=1)
    out2 = F.softmax(out2, axis=1)
    mul = out1.transpose(0, 1).mm(out2)
    loss_dis = torch2paddle.sum(mul) - paddle.trace(mul)
    return loss_dis


class CrossEntropyLabelSmooth(nn.Layer):

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average
        =True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = paddle.nn.LogSoftmax(axis=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = paddle.zeros(log_probs.size()).requires_grad_(False
            ).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon
            ) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = (-targets * log_probs).sum(1)
        return loss


def obtain_label(loader, netE, netC1, netC2, args, c=None):
    start_test = True
    netE.eval()
    netC1.eval()
    netC2.eval()
    with paddle.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            indexs = data[2]
            inputs = inputs.cuda()
            feas = netE(inputs)
            outputs1 = netC1(feas)
            outputs2 = netC2(feas)
            outputs = outputs1 + outputs2
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch2paddle.concat((all_fea, feas.float().cpu()), 0)
                all_output = torch2paddle.concat((all_output, outputs.float
                    ().cpu()), 0)
                all_label = torch2paddle.concat((all_label, labels.float()), 0)
    all_output = nn.Softmax(axis=1)(all_output)
    predict = paddle.argmax(all_output, 1)
    accuracy = torch2paddle.sum(paddle.squeeze(predict).float() == all_label
        ).item() / float(all_label.size()[0])
    all_fea = torch2paddle.concat((all_fea, paddle.ones([all_fea.size(0), 1
        ]).requires_grad_(False)), 1)
    all_fea = (all_fea.t() / paddle.norm(all_fea, p=2, axis=1)).t()
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-08 + aff.sum(axis=0)[:, None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-08 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = (
        'Only source accuracy = {:.2f}% -> After the clustering = {:.2f}%'.
        format(accuracy * 100, acc * 100))
    print(log_str + '\n')
    return pred_label.astype('int')


def gradient_discrepancy_loss(args, preds_s1, preds_s2, src_y, preds_t1,
    preds_t2, tgt_y, netE, netC1, netC2):
    loss_w = Weighted_CrossEntropy
    loss = nn.CrossEntropyLoss()
    total_loss = 0
    c_candidate = list(range(args.class_num))
    random.shuffle(c_candidate)
    for c in c_candidate[0:args.gmn_N]:
        gm_loss = 0
        src_ind = (src_y == c).nonzero().squeeze()
        tgt_ind = (tgt_y == c).nonzero().squeeze()
        if src_ind.shape == [0] or tgt_ind.shape == [0] or src_ind.shape == [0
            ] or tgt_ind.shape == [0]:
            continue
        p_s1 = preds_s1[src_ind]
        p_s2 = preds_s2[src_ind]
        p_t1 = preds_t1[tgt_ind]
        p_t2 = preds_t2[tgt_ind]
        s_y = src_y[src_ind]
        t_y = tgt_y[tgt_ind]
        src_loss1 = loss(p_s1, s_y)
        tgt_loss1 = loss_w(p_t1, t_y)
        src_loss2 = loss(p_s2, s_y)
        tgt_loss2 = loss_w(p_t2, t_y)
        grad_cossim11 = []
        grad_cossim22 = []
        for n, p in netC1.named_parameters():
            real_grad = grad([src_loss1], [p], create_graph=True,
                only_inputs=True, allow_unused=False)[0]
            fake_grad = grad([tgt_loss1], [p], create_graph=True,
                only_inputs=True, allow_unused=False)[0]
            if len(p.shape) > 1:
                _cossim = paddle.nn.CosineSimilarity(fake_grad, real_grad,
                    dim=1).mean()
            else:
                _cossim = paddle.nn.CosineSimilarity(fake_grad, real_grad,
                    dim=0)
            grad_cossim11.append(_cossim)
        grad_cossim1 = paddle.stack(grad_cossim11)
        gm_loss1 = (1.0 - grad_cossim1).sum()
        for n, p in netC2.named_parameters():
            real_grad = grad([src_loss2], [p], create_graph=True,
                only_inputs=True)[0]
            fake_grad = grad([tgt_loss2], [p], create_graph=True,
                only_inputs=True)[0]
            if len(p.shape) > 1:
                cosineSim = paddle.nn.CosineSimilarity(axis=1)
                _cossim = cosineSim(fake_grad, real_grad).mean()
            else:
                cosineSim = paddle.nn.CosineSimilarity(axis=0)
                _cossim = cosineSim(fake_grad, real_grad)
            grad_cossim22.append(_cossim)
        grad_cossim2 = paddle.stack(grad_cossim22)
        gm_loss2 = (1.0 - grad_cossim2).sum()
        gm_loss = (gm_loss1 + gm_loss2) / 2.0
        total_loss += gm_loss
    return total_loss / args.gmn_N


def gradient_discrepancy_loss_margin(args, p_s1, p_s2, s_y, p_t1, p_t2, t_y,
    netE, netC1, netC2):
    loss_w = Weighted_CrossEntropy
    loss = nn.CrossEntropyLoss(reduction='sum')
    gm_loss = 0
    src_loss1 = loss(p_s1, s_y)
    tgt_loss1 = loss_w(p_t1, t_y)
    src_loss2 = loss(p_s2, s_y)
    tgt_loss2 = loss_w(p_t2, t_y)
    grad_cossim11 = []
    grad_cossim22 = []
    for n, p in netC1.named_parameters():
        try:
            print(src_loss1.stop_gradient, p.stop_gradient)
        except:
            continue
        real_grad = grad([src_loss1], [p], create_graph=True, only_inputs=True)[0]
        fake_grad = grad([tgt_loss1], [p], create_graph=True, only_inputs=True)[0]
        #real_grad = grad([src_loss1], [p], create_graph=False, only_inputs=True)[0]
        #fake_grad = grad([tgt_loss1], [p], create_graph=False, only_inputs=True)[0]
        if len(p.shape) > 1:
            cosineSim = paddle.nn.CosineSimilarity(axis=1)
            _cossim = cosineSim(fake_grad, real_grad).mean()
        else:
            cosineSim = paddle.nn.CosineSimilarity(axis=0)
            _cossim = cosineSim(fake_grad, real_grad)
        grad_cossim11.append(_cossim)
    grad_cossim1 = paddle.stack(grad_cossim11)
    gm_loss1 = (1.0 - grad_cossim1).mean()
    for n, p in netC2.named_parameters():
        real_grad = grad([src_loss2], [p], create_graph=True, only_inputs=True)[0]
        fake_grad = grad([tgt_loss2], [p], create_graph=True, only_inputs=True)[0]
        #real_grad = grad([src_loss2], [p], create_graph=False, only_inputs=True)[0]
        #fake_grad = grad([tgt_loss2], [p], create_graph=False, only_inputs=True)[0]
        if len(p.shape) > 1:
            cosineSim = paddle.nn.CosineSimilarity(axis=1)
            _cossim = cosineSim(fake_grad, real_grad).mean()
        else:
            cosineSim = paddle.nn.CosineSimilarity(axis=0)
            _cossim = cosineSim(fake_grad, real_grad).mean()
        grad_cossim22.append(_cossim)
    grad_cossim2 = paddle.stack(grad_cossim22)
    gm_loss2 = (1.0 - grad_cossim2).mean()
    gm_loss = (gm_loss1 + gm_loss2) / 2.0
    return gm_loss


def Entropy_div(input_):
    epsilon = 1e-05
    input_ = torch2paddle.mean(input_, 0) + epsilon
    entropy = input_ * paddle.log(input_)
    entropy = torch2paddle.sum(entropy)
    return entropy


def Entropy_condition(input_):
    bs = input_.size(0)
    entropy = -input_ * paddle.log(input_ + 1e-05)
    entropy = torch2paddle.sum(entropy, dim=1).mean()
    return entropy


def Entropy(input_):
    return Entropy_condition(input_) + Entropy_div(input_)


def Weighted_CrossEntropy(input_, labels):
    input_s = F.softmax(input_)
    entropy = -input_s * paddle.log(input_s + 1e-05)
    entropy = torch2paddle.sum(entropy, dim=1)
    weight = 1.0 + paddle.exp(-entropy)
    weight = weight / torch2paddle.sum(weight).detach().item()
    return torch2paddle.mean(weight * nn.CrossEntropyLoss(reduction='none')
        (input_, labels))
