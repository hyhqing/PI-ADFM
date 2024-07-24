import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import torch.optim as optim
import ast
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from model_config import get_config
from model import PI_ADFM
from loss import Loss_HyNet,Loss_SOSNet
from train_tool import poly_adjust_learning_rate,ErrorRateAt95Recall,dis_meandistance
from Dataset import test_Dataset,train_Dataset
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--train_root',type=str,default='./mulit_dataset')
parser.add_argument('--test_root',type=str,default='./muilt_test')
parser.add_argument('--train_batchsize',type=int,default=24)
parser.add_argument('--num_pt_per_batch', type=int, default=24)
parser.add_argument('--test_batchsize',type=int,default=24)
parser.add_argument('--dim_desc', type=int, default=256)
parser.add_argument('--is_sosr', type=ast.literal_eval, default=False)
parser.add_argument('--knn_sos', type=int, default=8)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--output_dir', type=str, default='./trainweight_file')
parser.add_argument('--train_para_file', type=str, default='train_step_epoch')
parser.add_argument('--train_val_para_file', type=str, default='train_test_epoch')
parser.add_argument('--train_pth', type=str, default='./train_pth')
args = parser.parse_args()
#是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.CenterCrop(224),
                                 transforms.Resize(224),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 常用标准化
                                 ]),
    "test": transforms.Compose([transforms.ToTensor(),

                                transforms.CenterCrop(224),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])}


train_loader = torch.utils.data.DataLoader(train_Dataset(root_patch=args.train_root,transfrom=data_transform['train'],batchsize=args.train_batchsize),
                                           batch_size=args.train_batchsize, shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(test_Dataset(test_patch=args.test_root,test_transfrom=data_transform['test']),
                                          batch_size=args.test_batchsize, shuffle=True, num_workers=0)

config = get_config("small")
net =PI_ADFM(config).to(device)
loss_desc = Loss_HyNet(device, args.num_pt_per_batch, args.dim_desc, args.margin, args.alpha, args.is_sosr, args.knn_sos)
f_loss = Loss_SOSNet(device, args.num_pt_per_batch, args.dim_desc, args.margin, args.knn_sos)
model_weight_path = "./Pretrained weights/mobilenet_v2.pth"
mode_vit = './Pretrained weights/mobilevit2.pt'
assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
assert os.path.exists(mode_vit), "file {} dose not exist.".format(model_weight_path)
pre_weights = torch.load(model_weight_path,map_location=device)
pre_weights1 = torch.load(mode_vit, map_location=device)
net.load_state_dict(pre_weights,strict=False)
net.load_state_dict(pre_weights1,strict=False)
 # freeze features weights
for param in net.features.parameters():
    param.requires_grad = False
for name, param in net.conv_1.named_parameters():
    param.requires_grad = False
for name, param in net.layer_1.named_parameters():
    param.requires_grad = False
for name, param in net.layer_2.named_parameters():
    param.requires_grad =False


genxin_params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(genxin_params,lr=args.base_lr)
for epoch in range(args.max_epoch):
    net.train()
    running_loss = 0.0
    running_dist_pos = 0.0
    running_dist_neg = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    count = 0
    max_iteration = args.max_epoch*len(train_loader)
    for step, data in enumerate(train_bar):
        count +=1
        img_a, img_p = data
        optimizer.zero_grad()
        l2out_a,out_a,mobile_a,mvit_a = net(img_a.to(device),model='train')
        l2out_p,out_p,mobile_p,mvit_p= net(img_p.to(device),model='train')
        loss, dist_pos, dist_neg = loss_desc.compute(l2out_a, l2out_p, out_a, out_p)
        mobile_loss, mobile_dist_pos, mobile_dist_neg = f_loss.compute(mobile_a,mobile_p)
        mvit_loss,mvit_dist_pos,mvit_dist_neg = f_loss.compute(mvit_a,mvit_p)
        z_loss = 0.5*loss+0.25*mobile_loss+0.25*mvit_loss
        z_loss.backward()
        optimizer.step()
        lr = poly_adjust_learning_rate(optimizer,args.base_lr,max_iteration,step+1+epoch*len(train_loader), 0)
        running_loss += z_loss.item()
        running_dist_pos += dist_pos.item()
        running_dist_neg += dist_neg.item()
        # a=loss.detach().cpu().numpy()
        if count % 20 == 0:
            print('epoch {}: {}/{}:  dist_pos: {:.4f}, dist_neg: {:.4f}, mean_loss: {:.4f},lr: {:.6f},\n train_loss:{:.6f},mobile_loss:{:.6f},mvit_loss{:.6f}'.format(
                epoch + 1,
                step + 1,
                len(train_loader),
                running_dist_pos / (step + 1),
                running_dist_neg / (step + 1),
                running_loss / (step + 1), lr,loss.detach().cpu().numpy()[0],mobile_loss.detach().cpu().numpy()[0],
                mvit_loss.detach().cpu().numpy()[0]))
        if not torch.isfinite(loss):  # 判断损失是否有界
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        table_iteration = {"iteration": step + 1,
                           "mean_loss": running_loss / (step + 1),
                           "run_dis_pos": running_dist_pos / (step + 1),
                           "run_dis_neg": running_dist_neg / (step + 1),
                           'train_loss':loss.detach().cpu().numpy(),
                           'mobile_loss':mobile_loss.detach().cpu().numpy(),
                           'mvit_loss':mvit_loss.detach().cpu().numpy(),
                           "lr": lr}
        data_iteration = pd.DataFrame(table_iteration, index=[[0]])
        if not os.path.exists(args.output_dir + '/' + args.train_para_file + '_step' + '.csv'):
            data_iteration.to_csv(args.output_dir + '/' + args.train_para_file + '_step' + '.csv', mode='a',
                                  header=True, index=False)
        else:
            data_iteration.to_csv(args.output_dir + '/' + args.train_para_file + '_step' + '.csv', mode='a',
                                  header=False, index=False)
    print("\n\033[1;33;44m 第 %s 个epoch的训练损失为 %f \033[0m]" % (epoch+1, running_loss / len(train_loader)))
    torch.save(net.state_dict(), "./models.pth".format(256, epoch+1))
    #测试数据
    net.eval()
    labels, distances = [], []
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for batch_idx, data in enumerate(test_bar):
            data_a, data_p, label = data
            out_a, out_p, = net(data_a.to(device), model= 'eval'), net(data_p.to(device),model='eval')
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance计算欧式距离
            distances.append(dists.data.cpu().numpy().reshape(-1, 1))
            ll = label.data.cpu().numpy().reshape(-1, 1)
            labels.append(ll)
        num_tests = len(test_loader.dataset.test_imagelist)
        labels = np.vstack(labels).reshape(num_tests)
        distances = np.vstack(distances).reshape(num_tests)
        mean_distance = dis_meandistance(distances, labels)
        fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
        table = {"epoch": epoch,
                 "train_loss": running_loss / len(train_loader),
                 "fpr95": fpr95,
                 "mean distance": mean_distance}
        data = pd.DataFrame(table, index=[[0]])
        if not os.path.exists(args.output_dir + '/' + args.train_val_para_file + '.csv'):
            data.to_csv(args.output_dir + '/' + args.train_val_para_file + '.csv', mode='a',
                        header=True, index=False)
        else:
            data.to_csv(args.output_dir + '/' + args.train_val_para_file + '.csv', mode='a',
                        header=False, index=False)
        # writer.add_scalar(tags[2], fpr95, epoch)
        print('\33[91mTest set: Accuracy(FPR95): {:.8f}, mean distance: {:.5f}\n\33[0m'.format(fpr95, mean_distance))









