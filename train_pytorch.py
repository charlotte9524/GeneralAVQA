from argparse import ArgumentParser
import os
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import random
from tensorboardX import SummaryWriter
import datetime
import librosa
import librosa.display
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

import scipy.io


class AF(nn.Module):
    def __init__(self):
        super(AF, self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 2)) #torch.Size([480, 32, 128, 32])
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64), #torch.Size([480, 64, 128, 32])
            nn.MaxPool2d(kernel_size=(4, 1)) #torch.Size([480, 64, 32, 32])
        )

        self.p5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(2, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=(5, 3), padding=(2, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(4, 2))
        )

        self.p7 = nn.Sequential(
            nn.Linear(16384, 2048),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
        )

    def forward(self, input):
        input = self.p1(input) 
        input = self.p2(input)
        input = self.p5(input)
        input = torch.flatten(input, start_dim=-3)
        input = self.p7(input)
        return input

class VQADataset(Dataset):
    def __init__(self, videofeatures_dir, audiofeatures_dir, index=None, score=None, scale=1):
        super(VQADataset, self).__init__()
        self.features = np.zeros((len(index), 30, 2048))
        self.afeatures = np.zeros((len(index), 30, 4, 128, 64))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
            self.features[i, :, :] = np.squeeze(np.load(os.path.join(videofeatures_dir, str(index[i]) + '.npy')))[:30, :2048]
            self.afeatures[i, :, :, :, :] = np.load(os.path.join(audiofeatures_dir, str(index[i]) + '.npy'))[:30, :4, :, :64]
            self.mos[i] = score[i]  
        self.scale = scale  
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.afeatures[idx], self.label[idx]
        return sample

class Bi_RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=2, rnn_type='LSTM'):
        super(Bi_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the initial linear hidden layer
        self.init_linear = nn.Linear(self.input_dim, self.input_dim)

        # Define the LSTM layer
        self.lstm = eval('nn.' + rnn_type)(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True,
                                           bidirectional=True)

    def forward(self, input):
        # Forward pass through initial hidden layer
        linear_input = self.init_linear(input)

        lstm_out, self.hidden = self.lstm(linear_input)

        return lstm_out


class ATimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(ATimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        x_reshaped = x.contiguous().view(-1, x.size(-3), x.size(-2), x.size(-1))
        y = self.module(x_reshaped)

        # We have to reshape Y
        y = y.contiguous().view(-1, x.size(1), y.size(-1))
        return y

class VSFA(nn.Module):
    def __init__(self):
        super(VSFA, self).__init__()
        self.TP_BiL1 = Bi_RNN(input_dim=2048, hidden_dim=128)
        self.TP_BiL2 = Bi_RNN(input_dim=256, hidden_dim=64)

        self.ofc1 = nn.Linear(3840, 1024)
        self.odro1 = nn.Dropout(p=0.5)

        self.af = AF()
        self.ATD = ATimeDistributed(self.af)

        self.ATP_BiL1 = Bi_RNN(input_dim=2048, hidden_dim=128)
        self.ATP_BiL2 = Bi_RNN(input_dim=256, hidden_dim=64)


        self.oAfc1 = nn.Linear(3840, 1024)
        self.oAdro1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(2048, 1024)
        self.dro2 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(1024, 1)


    def forward(self, videoinput, audioinput):
        audioinput = self.ATD(audioinput)

        videoinput = self.TP_BiL1(videoinput) 
        videoinput = self.TP_BiL2(videoinput)  

        audioinput = self.ATP_BiL1(audioinput)
        audioinput = self.ATP_BiL2(audioinput)

        videoinput = torch.flatten(videoinput, start_dim=-2)
        videoinput = self.ofc1(videoinput)
        videoinput = self.odro1(videoinput)

        audioinput = torch.flatten(audioinput, start_dim=-2)
        audioinput = self.oAfc1(audioinput)
        audioinput = self.oAdro1(audioinput)


        all = torch.cat((videoinput, audioinput), 1) 
        all = self.fc2(all)
        all = self.dro2(all)
        all = self.relu(all)
        all = self.fc3(all)
        return all


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def compute_metrics(y_pred, y):
    '''
    compute metrics btw predictions & labels
    '''
    # compute SRCC & KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    # logistic regression btw y_pred & y
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)

    # compute  PLCC RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return [SRCC, KRCC, PLCC, RMSE]


if __name__ == "__main__":
    parser = ArgumentParser(description='"GeneralAVQA: Quality Assessment of UGC Audio-Videos')
    parser.add_argument("--seed", type=int, default=19920517) 
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--model', default='GeneralAVQA-', type=str,
                        help='model name (default: GeneralAVQA)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test ratio (default: 0.2)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='val ratio (default: 0.2)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    parser.add_argument('--videofeatures_dir', type=str, default='/home/cyq/Work/UGCAVquality/GeneralAVQA/features_spatial/',
                        help='path save video features')
    parser.add_argument('--audiofeatures_dir', type=str, default='/home/cyq/Work/UGCAVquality/GeneralAVQA/features_audio',
                        help='path save audio features')
    parser.add_argument('--videos_dir', type=str, default='/mnt/sdb/cyq_data/Data/UGCAVQA/SJTU-UAV',
                        help='path save SJTU-UAV database')
    parser.add_argument('--trained_model_path', type=str, default='./save_weight',
                        help='path to save model checkpoint')
    
    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()


    trained_model_path = args.trained_model_path
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)


    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    videofeatures_dir = args.videofeatures_dir
    audiofeatures_dir = args.audiofeatures_dir
    Info = scipy.io.loadmat(os.path.join(args.videos_dir,'MOS.mat'))
    video_names = []
    Mos = []
    for i in range(len(Info['videoName'])):
        vName = ''.join(Info['videoName'][i, 0])
        video_names.append(vName)
        Mos.append(float(Info['MOSz'][i, 0]))

    print('EXP ID: {}'.format(args.exp_id))
    print(args.model)

    time_str = datetime.datetime.now().strftime("%I%M%B%d")

    if not args.disable_visualization:  # Tensorboard Visualization
        writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}'
                               .format(args.log_dir, args.exp_id, args.model,
                                       args.lr, args.batch_size, args.epochs,time_str))

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    all_SRCC = 0
    all_PLCC = 0
    for exepoch in range(10):
        trained_model_file = trained_model_path + '/' + str(exepoch)
        index = [i for i in range(len(video_names))]
        random.shuffle(index)
        testindex = index[int(np.ceil((1 - args.test_ratio - args.val_ratio) * len(index))):int(np.ceil((1 - args.test_ratio) * len(index)))]
        train_index, val_index, test_index = [], [], []
        train_cost, val_cost, test_cost = [], [], []
        for i in range(len(video_names)):
            if i in testindex:
                test_index.append(video_names[i])
                test_cost.append(Mos[i])
            else:
                train_index.append(video_names[i])
                train_cost.append(Mos[i])
                val_index.append(video_names[i])
                val_cost.append(Mos[i])
        
        scale = max(train_cost) # label normalization factor
        # print(scale)
        train_dataset = VQADataset(videofeatures_dir, audiofeatures_dir, train_index, train_cost, scale=scale)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_dataset = VQADataset(videofeatures_dir, audiofeatures_dir, val_index, val_cost, scale=scale)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, pin_memory=False, num_workers=0)
        if args.test_ratio > 0:
            test_dataset = VQADataset(videofeatures_dir, audiofeatures_dir, test_index, test_cost, scale=scale)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, pin_memory=False, num_workers=0)

        model = VSFA().to(device)  

        criterion = nn.MSELoss() 
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val_criterion = 100 
        for epoch in range(args.epochs):
            # Train
            model.train()
            L = 0
            for i, (features, afeatures, label) in enumerate(train_loader):
                # print("\r train_index:{}".format(i), end='')
                features = features.to(device).float()
                afeatures = afeatures.to(device).float()
                label = label.to(device).float()
                optimizer.zero_grad()  
                outputs = model(features, afeatures)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                L = L + loss.item()
            train_loss = L / (i + 1)
            print('{}-{}:train loss {}'.format(exepoch, epoch, train_loss))

            model.eval()
            # Val
            y_pred = np.zeros(len(val_index))
            y_val = np.zeros(len(val_index))
            L = 0
            with torch.no_grad():
                for i, (features, afeatures, label) in enumerate(val_loader):
                    # print("\r valid_index:{}".format(i), end='')
                    y_val[i] = scale * label.item()  #
                    features = features.to(device).float()
                    afeatures = afeatures.to(device).float()
                    label = label.to(device).float()
                    outputs = model(features, afeatures)
                    y_pred[i] = scale * outputs.item()
                    loss = criterion(outputs, label)
                    L = L + loss.item()
            val_loss = L / (i + 1)
            [val_SROCC, val_KROCC, val_PLCC, val_RMSE] = compute_metrics(y_pred, y_val)


            # Test
            if  args.test_ratio > 0:
                y_pred = np.zeros(len(test_index))
                y_test = np.zeros(len(test_index))
                L = 0
                with torch.no_grad():
                    for i, (features, afeatures, label) in enumerate(test_loader):
                        # print("\r test_index:{}".format(i), end='')
                        y_test[i] = scale * label.item()  #
                        features = features.to(device).float()
                        afeatures = afeatures.to(device).float()
                        label = label.to(device).float()
                        outputs = model(features, afeatures)
                        y_pred[i] = scale * outputs.item()
                        loss = criterion(outputs, label)
                        L = L + loss.item()
                test_loss = L / (i + 1)
                [SROCC, KROCC, PLCC, RMSE] = compute_metrics(y_pred, y_test)



            if not args.disable_visualization:  # record training curves
                writer.add_scalar("loss/train-{}".format(exepoch), train_loss, epoch)  
                writer.add_scalar("loss/val-{}".format(exepoch), val_loss, epoch)  
                writer.add_scalar("SROCC/val-{}".format(exepoch), val_SROCC, epoch)  
                writer.add_scalar("KROCC/val-{}".format(exepoch), val_KROCC, epoch)  
                writer.add_scalar("PLCC/val-{}".format(exepoch), val_PLCC, epoch)  
                writer.add_scalar("RMSE/val-{}".format(exepoch), val_RMSE, epoch)  
                writer.add_scalar("loss/test-{}".format(exepoch), test_loss, epoch)  
                writer.add_scalar("SROCC/test-{}".format(exepoch), SROCC, epoch)  
                writer.add_scalar("KROCC/test-{}".format(exepoch), KROCC, epoch)  
                writer.add_scalar("PLCC/test-{}".format(exepoch), PLCC, epoch)  
                writer.add_scalar("RMSE/test-{}".format(exepoch), RMSE, epoch)  

            print("Val results: {}-{}, val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(exepoch, epoch, val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            
            # Update the model with the best val_RMSE
            if val_RMSE < best_val_criterion:
                # torch.save(model.state_dict(), trained_model_file)
                print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(args.exp_id, epoch))
                best_val_criterion = val_RMSE  
                best_SRCC = SROCC
                best_PLCC = PLCC
                print("Tes results: {}-{}, test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                    .format(exepoch, epoch, test_loss, SROCC, KROCC, PLCC, RMSE))

        all_SRCC += best_SRCC
        all_PLCC += best_PLCC
    
    print(all_SRCC/10)
    print(all_PLCC/10)
