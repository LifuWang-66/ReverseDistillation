import torch
import numpy as np
import random
import os
import pandas as pd
from argparse import ArgumentParser
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_test import evaluation_multi_proj
from utils.utils_train import MultiProjectionLayer
from dataset.dataset import MVTecDataset_test, get_data_transforms
from utils.utils_test import cal_anomaly_map
from scipy.ndimage import gaussian_filter
import torchvision.transforms as transforms
from PIL import Image
import cv2
import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_folder', default = './your_checkpoint_folder', type=str)
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--classes', nargs="+", default=["carpet", "leather"])
    pars = parser.parse_args()
    return pars

def inference(_class_, pars):
    if not os.path.exists(pars.checkpoint_folder):
        os.makedirs(pars.checkpoint_folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)
    
    test_path = 'C:/Users/lifuw/Desktop/Projects/anomaly_detection/Revisiting-Reverse-Distillation/content/' + _class_

    checkpoint_class  = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # Use pretrained wide_resnet50 for encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)

    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    proj_layer =  MultiProjectionLayer(base=64).to(device)  
    # Load trained weights for projection layer, bn (OCBE), decoder (student)    
    checkpoint_class  = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    ckp = torch.load(checkpoint_class, map_location='cpu')
    proj_layer.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])
  
    # Lifu: I added this part to test the anomaly map
    folder_path = 'C:/Users/lifuw/Desktop/Projects/anomaly_detection/Revisiting-Reverse-Distillation/content/' + _class_ + '/test/'
    test_list = os.listdir(folder_path)
    for test_type in test_list:
        sub_folder_path = os.path.join(folder_path, test_type)
        file_list = os.listdir(sub_folder_path)
        counter = 0
        time_counter = 0
        for img_name in file_list:
            img = cv2.imread(os.path.join(sub_folder_path, img_name))
            cv2.imshow('img', img)
            cv2.waitKey(0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img= cv2.resize(img/255., (256, 256))
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.to(device)

            encoder.eval()
            proj_layer.eval()
            bn.eval()
            decoder.eval()

            inputs = encoder(img)
            features = proj_layer(inputs)
            outputs = decoder(bn(features))
            start = time.time()
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            end = time.time()
            time_counter += end - start

            cv2.imshow('anomaly_map', anomaly_map)
            cv2.waitKey(0)

            # Metrics
            print(np.mean(anomaly_map), np.std(anomaly_map))
            good_product_mean = 0.15
            median = np.median(anomaly_map)
            absolute_deviations = np.abs(anomaly_map - median)
            median_ad = np.median(absolute_deviations)
            mean_ad = np.mean(absolute_deviations)
            if (np.mean(anomaly_map) > 0.2 or np.std(anomaly_map) > 0.065 
                or anomaly_map[anomaly_map > good_product_mean * 3].shape[0] > 110
                or median_ad > 0.035 or mean_ad > 0.05):
                # if test_type == 'good':
                #     print(np.mean(anomaly_map), np.std(anomaly_map), anomaly_map[anomaly_map > good_product_mean * 3].shape[0]
                #       ,median_ad, mean_ad)
                # print("Anomaly detected!")
                if test_type != 'good':
                    counter += 1
            else:
                # print("No anomaly detected!")
                if test_type == 'good':
                    counter += 1
            # print(anomaly_map[anomaly_map > 0.45].shape)
            # print(np.max(anomaly_map))
            # print(anomaly_map[anomaly_map > 0.5].shape)
            # print("Time: ", end - start)
        print('Test type: ', test_type, ',\tAccuracy: ', counter/len(file_list), ',\tAverage inference time: ', time_counter/len(file_list))
    # auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device)        



if __name__ == '__main__':
    pars = get_args()

    item_list = [ 'carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']
    setup_seed(111)
    metrics = {'class': [], 'AUROC_sample':[], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    
    for c in pars.classes:
        inference(c, pars)
        # auroc_sp, auroc_px, aupro_px = inference(c, pars)
        # metrics['class'].append(c)
        # metrics['AUROC_sample'].append(auroc_sp)
        # metrics['AUROC_pixel'].append(auroc_px)
        # metrics['AUPRO_pixel'].append(aupro_px)
        # metrics_df = pd.DataFrame(metrics)
        # metrics_df.to_csv(f'{pars.checkpoint_folder}/metrics_checkpoints.csv', index=False)