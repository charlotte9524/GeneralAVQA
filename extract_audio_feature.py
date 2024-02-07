import numpy as np
import librosa
import librosa.display
from spafe.features.gfcc import gfcc
from argparse import ArgumentParser
import scipy.io
import os
import warnings
import sys
from moviepy.editor import *
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def creat_features(raw_sig, sr, standardize=False):
    # generate 4 channels 128*128 time-frequency feature: chromagram, chroma_cq, mfcc, gfcc
    chromagram = librosa.feature.chroma_stft(y=raw_sig, sr=sr, n_fft=1024, hop_length=512, n_chroma=128).T
    chroma_cq = librosa.feature.chroma_cqt(y=raw_sig, sr=sr, hop_length=512, n_chroma=128, bins_per_octave=None).T  #
    mfcc_ = librosa.feature.mfcc(y=raw_sig, sr=sr, n_fft=1024, hop_length=512, n_mfcc=128).T
    gfcc_ = gfcc(sig=raw_sig, fs=sr, nfft=1024, win_len=1024 / sr, win_hop=508 / sr, nfilts=256, num_ceps=128)
    if gfcc_.shape[0] > mfcc_.shape[0]:
        gfcc_ = gfcc_[: mfcc_.shape[0], :]
    elif gfcc_.shape[0] < mfcc_.shape[0]:
        chromagram = chromagram[: gfcc_.shape[0], :]
        chroma_cq = chroma_cq[: gfcc_.shape[0], :]
        mfcc_ = mfcc_[: gfcc_.shape[0], :]
    
    assert (chromagram.shape == chroma_cq.shape == mfcc_.shape == gfcc_.shape)
    return np.dstack((chromagram, chroma_cq, mfcc_, gfcc_))

def generate_multi_channel(file_name, length=8, frame_len=0.75, frame_overlap=0.5, augmentation=False): 
    features = []
    y, sr = librosa.load(file_name, sr=None) 
    frame_len =int(np.floor(len(y)/length*frame_len)) 
    frame_overlap = int(np.ceil(len(y)/length*frame_overlap)) 
    if augmentation is True:
        y_noise = y + 0.009 * np.random.normal(0, 1, len(y))
        y_roll = np.roll(y, int(sr / 10))
        y_time_stch = librosa.effects.time_stretch(y, 0.8)
        y_pitch_sf = librosa.effects.pitch_shift(y, sr, n_steps=-5)
        for y_ in [y, y_noise, y_roll, y_time_stch, y_pitch_sf]:
            if len(y_) < frame_len:
                y_pad = np.hstack((y_, y_[-(frame_len - len(y_)):]))
                fea = creat_features(y_pad, sr)
                features.append(fea)
            else:
                for i in range(0, len(y_) - frame_len, frame_len - frame_overlap):
                    y_tmp = y_[i: i + frame_len]
                    fea = creat_features(y_tmp, sr)
                    features.append(fea)
    else:
        if len(y) < frame_len:
            y_pad = np.hstack((y, y[-(frame_len - len(y)):]))
            fea = creat_features(y_pad, sr)
            features.append(fea)
        else:
            for i in range(0, len(y) - frame_len+1, frame_len - frame_overlap):
                y_tmp = y[i: i + frame_len]
                fea = creat_features(y_tmp, sr)
                features.append(fea)
    return features 


if __name__ == "__main__":
    parser = ArgumentParser(description='Extracting Audio Features')
    parser.add_argument('--features_dir', type=str, default='./features_audio/',
                        help='path to save audio features')
    parser.add_argument('--videos_dir', type=str, default='/mnt/sdb/cyq_data/Data/UGCAVQA/SJTU-UAV',
                        help='path where save database')
    args = parser.parse_args()

    features_dir = args.features_dir
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)


    Info = scipy.io.loadmat(os.path.join(args.videos_dir, 'MOS.mat'))
    for i in range(len(Info['videoName'])):
        vName = ''.join(Info['videoName'][i, 0])
        afeatures = generate_multi_channel(os.path.join(args.videos_dir, 'wav', vName[:-3]+'wav'))

        afeatures = np.array(afeatures)
        afeatures = np.transpose(afeatures, (0, 3, 2, 1))
        # print(afeatures.shape)
        np.save(os.path.join(features_dir, vName+'.npy'), afeatures)
    
