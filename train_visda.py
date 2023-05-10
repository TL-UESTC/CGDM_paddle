from __future__ import print_function
from paddle.fluid.dygraph.base import to_variable
from x2paddle import torch2paddle
import argparse
import paddle.optimizer as optim
from utils import *
from taskcv_loader import CVDataLoader
from models.basenet import *
from paddle.vision import transforms
from paddle.vision import datasets
import paddle.nn.functional as F
import os
import time
import numpy as np
import warnings
from data_loader.folder import ImageFolder_ind
from utils import discrepancy
import paddle
warnings.filterwarnings('ignore')
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0003,
                    metavar='LR', help='learning rate (default: 0.0003)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--optimizer', type=str, default='momentum',
                    metavar='OP', help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='dibles CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--num_layer', type=int, default=2,
                    metavar='K', help='how many layers for classifier')
parser.add_argument('--train_path', type=str, default='/media/server1/xxxxxsmall/VisDA/train',
                    metavar='B', help='directory of source datasets')
parser.add_argument('--val_path', type=str, default='/media/server1/xxxxxsmall/VisDA/validation',
                    metavar='B', help='directory of target datasets')
parser.add_argument('--gmn_N', type=int, default='12', metavar='B',
                    help='The number of classes to calulate gradient similarity')
parser.add_argument('--class_num', type=int, default='12', metavar='B',
                    help='The number of classes')
parser.add_argument('--pseudo_interval', type=int,
                    default=2000, metavar='B', help='')
parser.add_argument('--resnet', type=str, default='101',
                    metavar='B', help='which resnet 18,50,101,152,200')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
args.cuda = not args.no_cuda and paddle.is_compiled_with_cuda()
torch2paddle.set_cuda_device(args.gpu)
paddle.seed(args.seed)
"""if args.cuda:
    torch.cuda.manual_seed(args.seed)"""
train_path = args.train_path
val_path = args.val_path
num_k = args.num_k
num_layer = args.num_layer
batch_size = args.batch_size
lr = args.lr
data_transforms = {
    train_path: transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        torch2paddle.ToTensor(),
        torch2paddle.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_path: transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        torch2paddle.ToTensor(),
        torch2paddle.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
dsets = {x: ImageFolder_ind(os.path.join(x), data_transforms[x]) for x in [
    train_path, val_path]}
dsets_tgt_no_shuffle = ImageFolder_ind(os.path.join(val_path),
                                       data_transforms[val_path])
data_loader_T_no_shuffle = torch2paddle.DataLoader(dsets_tgt_no_shuffle, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
#data_loader_T_no_shuffle = paddle.io.DataLoader(dsets_tgt_no_shuffle,batch_size=32, shuffle=False, drop_last=False, num_workers=4)
dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]}
dset_classes = dsets[train_path].classes
dset_classes = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
                'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
classes_acc = {}
for i in dset_classes:
    classes_acc[i] = []
    classes_acc[i].append(0)
    classes_acc[i].append(0)
train_loader = CVDataLoader()
train_loader.initialize(dsets[train_path], dsets[val_path], batch_size,
                        shuffle=True, drop_last=True)
dataset = train_loader.load_data()
test_loader = CVDataLoader()
test_loader.initialize(dsets[train_path], dsets[val_path], 256,
                       shuffle=True, drop_last=False)
dataset_test = test_loader.load_data()
option = 'resnet' + args.resnet
G = ResBottle(option)
#test
#G2 = ResBottle(option)
F1 = ResClassifier(num_classes=12, num_layer=num_layer, num_unit=G.
                   output_num(), middle=1000)
F2 = ResClassifier(num_classes=12, num_layer=num_layer, num_unit=G.
                   output_num(), middle=1000)
F1.apply(weights_init)
F2.apply(weights_init)
if args.cuda:
    G.cuda()
    F1.cuda()
    F2.cuda()
if args.optimizer == 'momentum':
    optimizer_g = torch2paddle.Momentum(
        list(G.features.parameters()), lr=args.lr, weight_decay=0.0005)
    optimizer_f = torch2paddle.Momentum(list(F1.parameters()) + list(F2.
                                                                     parameters()), momentum=0.9, lr=args.lr, weight_decay=0.0005)
elif args.optimizer == 'adam':
    optimizer_g = torch2paddle.Adam(G.features.parameters(), lr=args.lr,
                                    weight_decay=0.0005)
    optimizer_f = torch2paddle.Adam(list(F1.parameters()) + list(F2.
                                                                 parameters()), lr=args.lr, weight_decay=0.0005)
else:
    optimizer_g = paddle.optimizer.Adadelta(parameters=G.features.
                                            parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = paddle.optimizer.Adadelta(parameters=list(F1.parameters()
                                                            ) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
start = -1


def train(num_epoch):
    criterion = nn.CrossEntropyLoss()
    criterion_w = Weighted_CrossEntropy
    for ep in range(num_epoch):
        #test(1)
        since = time.time()
        for batch_idx, data in enumerate(dataset):
            if ep > start and batch_idx % args.pseudo_interval == 0:
                print('Obtaining target label...')
                G.eval()
                F1.eval()
                F2.eval()
                mem_label = obtain_label(data_loader_T_no_shuffle, G, F1, F2, args)
                mem_label = paddle.to_tensor(mem_label).cuda()
            G.train()
            F1.train()
            F2.train()
            data_s = data['S']
            label_s = data['S_label']
            data_t = data['T']
            label_t = data['T_label']
            index_t = data['T_index']
            if ep > start:
                pseudo_label_t = mem_label[index_t]
            if dataset.stop_S:
                break
            if args.cuda:
                data_s, label_s = data_s.cuda(), label_s.cuda()
                data_t, label_t = data_t.cuda(), label_t.cuda()
                if ep > start:
                    pseudo_label_t = pseudo_label_t.cuda()
            data_all = to_tensor(torch2paddle.concat((data_s, data_t), 0))
            #print(G)
            label_s = to_tensor(label_s)
            #print(data_all.stop_gradient)
            bs = len(label_s)
            """source domain discriminative"""
            optimizer_g.clear_grad()
            optimizer_f.clear_grad()
            output = G(data_all)
            output1 = F1(output)
            output2 = F2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1_s = F.softmax(output_t1)
            output_t2_s = F.softmax(output_t2)
            entropy_loss = Entropy(output_t1_s)
            entropy_loss += Entropy(output_t2_s)
            if ep > start:
                supervision_loss = criterion_w(output_t1, pseudo_label_t) + criterion_w(output_t2, pseudo_label_t)
            else:
                supervision_loss = 0
            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)
            all_loss = (loss1 + loss2 + 0.01 * entropy_loss + 0.01 *
                        supervision_loss)
            #print("loss1:{}".format(all_loss))
            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()
            """target domain diversity"""
            optimizer_g.clear_grad()
            optimizer_f.clear_grad()
            #print(data_all[0][0])
            output = G(data_all)
            output1 = F1(output)
            ################### test #####################
            # print(data_all)
            #output.backward()
            ################### test #####################
            output2 = F2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1_s = F.softmax(output_t1)
            output_t2_s = F.softmax(output_t2)
            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)
            entropy_loss = Entropy(output_t1_s)
            entropy_loss += Entropy(output_t2_s)
            loss_dis = discrepancy(output_t1, output_t2)
            all_loss = loss1 + loss2 - 1.0 * loss_dis + 0.01 * entropy_loss
            all_loss.backward()  # 报错
            optimizer_f.step()
            """target domain discriminability"""
            for i in range(num_k):
                optimizer_g.clear_grad()
                optimizer_f.clear_grad()
                output = G(data_all)
                output1 = F1(output)
                output2 = F2(output)
                output_s1 = output1[:bs, :]
                output_s2 = output2[:bs, :]
                output_t1 = output1[bs:, :]
                output_t2 = output2[bs:, :]
                output_t1_s = F.softmax(output_t1)
                output_t2_s = F.softmax(output_t2)
                entropy_loss = Entropy(output_t1_s)
                entropy_loss += Entropy(output_t2_s)
                loss_dis = discrepancy(output_t1, output_t2)
                if ep > start and False:
                    gmn_loss = gradient_discrepancy_loss_margin(args, output_s1,
                                                            output_s2, label_s, output_t1, output_t2,
                                                            pseudo_label_t, G, F1, F2)
                    
                else:
                    gmn_loss = 0
                all_loss = (1.0 * loss_dis + 0.01 * entropy_loss + 0.01 *
                            gmn_loss)
                all_loss.backward()
                optimizer_g.step()
            if batch_idx % args.log_interval == 0:
                print( 
                    'Train Ep: {} [{}/{} ({:.6f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t CDD: {:.6f} Entropy: {:.6f} '
                    .format(ep, batch_idx, len(dataset.data_loader_S),
                            100.0 * batch_idx / len(dataset.data_loader_S), loss1.
                            item(), loss2.item(), loss_dis.item(), entropy_loss.item())
                )
        test(ep + 1)
        print('time:', time.time() - since)
        paddle.save(G.state_dict(), os.path.join('models_trained', 'visda', 'extractor.pdparams'))
        paddle.save(F1.state_dict(), os.path.join('models_trained', 'visda', 'classifier1.pdparams'))
        paddle.save(F2.state_dict(), os.path.join('models_trained', 'visda', 'classifier2.pdparams'))
        print('-' * 100)


def test(epoch):
    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct_add = 0
    size = 0
    print('-' * 100, '\nTesting')
    with paddle.no_grad():
        for batch_idx, data in enumerate(dataset_test):
            if dataset_test.stop_T:
                break
            if args.cuda:
                img = data['T']
                label = data['T_label']
                img, label = img.cuda(), label.cuda()
            img, label = to_tensor(img), to_tensor(label)
            output = G(img)
            output1 = F1(output)
            output2 = F2(output)
            paddle_nll_loss = paddle.nn.loss.NLLLoss()
            test_loss += paddle_nll_loss(output1, label)
            output_add = output1 + output2
            pred = output_add.max(1)[1]
            #print(pred.cpu())
            correct_add += pred.equal(label).astype('float64').cpu().sum(None)
            #correct_add += pred.sum()
            #print(correct_add)
            size += label.data.size()[0]
            for i in range(len(label)):
                key_label = dset_classes[label.long()[i].item()]
                key_pred = dset_classes[pred.long()[i].item()]
                classes_acc[key_label][1] += 1
                if key_pred == key_label:
                    classes_acc[key_pred][0] += 1

    test_loss /= len(test_loader)
    #print(epoch, test_loss, float(correct_add), size, float(correct_add)/size)
    print('Epoch: {:d} Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'
        .format(epoch, float(test_loss), correct_add, size, 100.0 * (float(correct_add)/size)))
    avg = []
    for i in dset_classes:
        print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0],
                                               classes_acc[i][1], 100.0 * classes_acc[i][0] / classes_acc[i][1]))
        avg.append(100.0 * float(classes_acc[i][0]) / classes_acc[i][1])
    print('\taverage:', np.average(avg))
    for i in dset_classes:
        classes_acc[i][0] = 0
        classes_acc[i][1] = 0


if __name__ == '__main__':
    train(args.epochs + 1)
