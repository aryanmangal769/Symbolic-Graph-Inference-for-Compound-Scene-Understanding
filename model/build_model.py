import torch
import torch.nn as nn
from model.gsnn.gsnn import GSNN
from model.mgsnn.gsnn import MGSNN
from model.vit.vit import ViT

def build_gsnn(configs, KG_vocab, KG_nodes):
    num_gpu = configs['num_gpu']
    use_gpu = (len(num_gpu) > 0)
    pretrained_model_path = configs['gsnn_path']
    model = GSNN(configs, KG_vocab, KG_nodes)

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

def build_mgsnn(configs, KG_vocab, KG_nodes):
    num_gpu = configs['num_gpu']
    use_gpu = (len(num_gpu) > 0)
    pretrained_model_path = configs['gsnn_path']
    model = MGSNN(configs, KG_vocab, KG_nodes)

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

def build_vit(configs):
    num_gpu = configs['num_gpu']
    use_gpu = (len(num_gpu) > 0)
    pretrained_model_path = configs['vit_path']
    num_classes = configs['num_classes']
    model = ViT(num_classes=num_classes)

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