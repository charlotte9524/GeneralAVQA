from argparse import ArgumentParser
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import os
import scipy.io
import cv2
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, frame_length, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.format = video_format
        self.width = width
        self.height = height
        self.frame_length = frame_length
    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        assert self.format == 'YUV420' or self.format == 'RGB'
        video_data = []
        cap = cv2.VideoCapture(os.path.join(self.videos_dir, video_name))
        frameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frameInter = math.floor(frameNum / self.frame_length)
        index = -1
        while (cap.isOpened()):
            ret, frame = cap.read()
            index = index + 1
            if ret:
                if index % frameInter == 0:
                    video_data.append(frame)
                else:
                    continue
            else:
                break
        video_data = np.array(video_data)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        print('video_width: {} video_height: {}'.format(video_width, video_height))
        transformed_video = torch.zeros([video_length, video_channel,  video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video}

        return sample

class CNNModel(torch.nn.Module):
    """Modified CNN models for feature extraction"""
    def __init__(self, model='ResNet-50'):
        super(CNNModel, self).__init__()
        if model == 'SpatialExtractor':
            print("use SpatialExtractor")
            from SpatialExtractor.get_spatialextractor_model import make_spatial_model
            model = make_spatial_model()
            self.features = nn.Sequential(*list(model.backbone.children())[:-2])
        else:
            print("use default ResNet-50")
            self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])

    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x) 
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
        return features_mean 

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)

def get_features(video_data, frame_batch_size=64, model='ResNet-50', device='cuda'):
    """feature extraction"""
    extractor = CNNModel(model=model).to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        while frame_end < video_length:
            batch = video_data[frame_start:frame_end].to(device)
            features_mean = extractor(batch)
            output = torch.cat((output, features_mean), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:video_length].to(device)
        features_mean = extractor(last_batch)
        output = torch.cat((output, features_mean), 0)
    
    output = output.squeeze()
    return output

if __name__ == "__main__":
    parser = ArgumentParser(description='Extracting Video Spatial Features using model-based transfer learning')
    parser.add_argument("--seed", type=int, default=19901116)
    parser.add_argument('--model', default='SpatialExtractor', type=str,
                        help='which pre-trained model used (default: ResNet-50)')
    parser.add_argument('--frame_batch_size', type=int, default=10,
                        help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--frame_length', type=int, default=30,
                        help='frame num to extract feature')
    parser.add_argument('--features_dir', type=str, default='./features_spatial/',
                        help='frame num to extract feature')
    parser.add_argument('--videos_dir', type=str, default='/mnt/sdb/cyq_data/Data/UGCAVQA/SJTU-UAV',
                        help='frame num to extract feature')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument("--ith", type=int, default=0, help='start video id')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    features_dir = args.features_dir
    videos_dir = args.videos_dir
    Info = scipy.io.loadmat(args.videos_dir + '/MOS.mat')
    video_names = []
    width = []
    height = []
    for i in range(len(Info['videoName'])):
        vName = ''.join(Info['videoName'][i, 0])
        video_names.append(vName)
        width.append(int(Info['resolution'][i][0]))
        height.append(int(Info['resolution'][i][1]))

    video_format = 'RGB'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    dataset = VideoDataset(videos_dir, video_names, args.frame_length, video_format, width, height)


    for i in range(args.ith, len(dataset)):
        current_data = dataset[i]
        print('Video {} : length {}'.format(i, current_data['video'].shape[0]))
        features = get_features(current_data['video'], args.frame_batch_size, args.model, device)
        print(features.shape)
        np.save(features_dir + video_names[i], features.to('cpu').numpy())

