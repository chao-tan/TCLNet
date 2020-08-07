from modules import network
from modules.components import TCLNET




def create_backbone(net_name, init_type, init_gain, gpu_ids):
    net = None

    if net_name is 'TCLNET':
        net = TCLNET.net()

    else:
        raise NotImplementedError("model not found")

    return network.init_net(net, init_type, init_gain, gpu_ids)


# model = create_backbone(net_name='TCLNET',init_type='normal',init_gain=0.01,gpu_ids=[0])
# print('# generator parameters:', sum(param.numel() for param in model.parameters())/1000./1000.)
