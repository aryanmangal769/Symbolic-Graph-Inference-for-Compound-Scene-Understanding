import torch
import torch.nn as nn
from model.gsnn.gsnn import GSNN

def build_gsnn(configs):
    num_gpu = configs['num_gpu']
    use_gpu = (len(num_gpu) > 0)
    pretrained_model_path = configs['gsnn_path']
    model = GSNN(configs)

    if use_gpu:
        model = model.to(torch.device('cuda:{}'.format(num_gpu[0])))
        model = torch.nn.DataParallel(model, device_ids=num_gpu)
        print("Finish cuda loading")

    if pretrained_model_path != "":
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cuda:{}'.format(num_gpu[0])))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        if use_gpu:
            model.load_state_dict(model_dict)
        else:
            print("loading on cpu")
            model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
            print("load model from {}".format(pretrained_model_path))

    return model