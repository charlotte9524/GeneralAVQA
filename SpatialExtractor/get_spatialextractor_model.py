import torch
from SpatialExtractor.BaseCNN_4FeatureGetting import BaseCNN
from SpatialExtractor.Main_4FeatureGetting import parse_config
from collections import OrderedDict

def loadweight(model_path):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

def make_spatial_model():

    config = parse_config()
    model = BaseCNN(config).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    ckpt = './SpatialExtractor/weights/DataParallel-00008.pt'
    state_dict = loadweight(ckpt)
    model.load_state_dict(state_dict)
    # checkpoint = torch.load(ckpt)
    # model.load_state_dict(checkpoint['state_dict'])

    return model

if __name__ == "__main__":
    make_spatial_model()