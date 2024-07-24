import numpy as np
import matplotlib.pyplot as plt
import torch
from match_ulit import match_xy,descriptors,mosaic_map,pcscong
from plot_match import plot_matches
import copy
from model import PI_ADFM
from model_config import get_config
import cv2
from enum import Enum
from cv2.xfeatures2d import matchGMS
class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5
image1_root = './image/opt_filter.png'
image2_root = './image/SAR_filter.png'

image1,int_imgxy1,_,kp1,_ = pcscong(image1_root)
image2,int_imgxy2,_,kp2,_ = pcscong(image2_root)

h,w = image1.shape[:2]
padding = 84
image1_pad = cv2.copyMakeBorder(image1,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=0)
image2_pad = cv2.copyMakeBorder(image2,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
config = get_config("small")
net = PI_ADFM(config).to(device)
discriptors1 = descriptors(int_imgxy1,image1_pad,net,device=device)
discriptors2 = descriptors(int_imgxy2,image2_pad,net,device=device)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
fl_many_matchs = flann.knnMatch(discriptors1,discriptors2,k=2)
goodMatch=[]
disdif_avg =0
for m, n in fl_many_matchs:
    disdif_avg += n.distance - m.distance
disdif_avg = disdif_avg / len(fl_many_matchs)
for m, n in fl_many_matchs:
    if n.distance > m.distance + disdif_avg:
        goodMatch.append(m)
print('Adaptive threshold matching yields {} matches'.format(len(goodMatch)))

matches_gms = matchGMS(image1.shape[:2], image1.shape[:2], kp1, kp2, goodMatch, withScale=False, withRotation=False,
                       thresholdFactor=6)
gms_z = []
gms_y = []
gms_lin =[]
for i,math in enumerate(matches_gms):
    gms_z.append([kp1[math.queryIdx].pt[0], kp1[math.queryIdx].pt[1]])
    gms_y.append([kp2[math.trainIdx].pt[0], kp2[math.trainIdx].pt[1]])
    gms_lin.append([i,i])
src_pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in matches_gms]).reshape(-1, 1, 2)
dst_pts1 = np.float32(
        [kp2[m.trainIdx].pt for m in matches_gms]).reshape(-1, 1, 2)

M,mask =cv2.findHomography(src_pts1, dst_pts1, cv2.USAC_MAGSAC,3.0,
                               confidence=0.999,
                               maxIters=1000)
mask_matches = mask.ravel().tolist()
block = 64
a,b = int(h/block),int(w/block)
_,_,_,matchxy3,matchxy4,_,_ = match_xy(matches_gms,discriptors1,discriptors2,kp1,kp2)
inlie_idx = np.nonzero(mask_matches)[0]
image1_warp1 = cv2.warpPerspective(image1,M,(w,h),flags=cv2.INTER_LINEAR)
image2_warp1 = copy.copy(image2)
image_44= cv2.add(image1_warp1,image2_warp1)
image_fusion2 = mosaic_map(image1_warp1,image2_warp1,block_size=block)
Registration_image = cv2.cvtColor(image_fusion2,cv2.COLOR_RGB2BGR)
plt.figure(1)
plot_matches(
    image1,
    image2,
    matchxy3,
    matchxy4,
    np.column_stack((inlie_idx, inlie_idx)),
    liners='',
    plot_matche_points = False,
    matchline = True,
    marketsize= 2,
    market='o',
    matchlinewidth = 0.3)
plt.show()

