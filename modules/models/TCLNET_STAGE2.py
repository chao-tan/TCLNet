from modules.comdel import cmodel
import torch
from modules import backbone
from utils import tools_plot
from torch.nn import functional as F
import numpy as np



class network(cmodel):

    def __init__(self, config):
        cmodel.__init__(self,config)

        self.config = config
        self.loss_names = ['REGRESSION','TMP']
        self.visual_names = ['INPUT_SHOW','LABEL_HEATMAP_512','SIM_HEATMAP_512','OUTPUT_DIST']
        self.model_names = ["G"]

        self.netG = backbone.create_backbone(net_name='TCLNET',
                                             init_type=config['init_type'],
                                             init_gain=float(config['init_gain']),
                                             gpu_ids=config['gpu_ids'])

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=float(config['learning_rate']),betas=(0.5, 0.999),weight_decay=1e-4)
        self.optimizers.append(['G',self.optimizer_G])
        self.cretition = torch.nn.MSELoss()

        self.INPUT = None
        self.INPUT_SHOW = None
        self.SIM_HEATMAP_512 = None
        self.OUTPUT_HEATMAPS = None
        self.OUTPUT_DIST = None
        self.LABEL = None
        self.LABEL_HEATMAP_512 = None
        self.LABEL_HEATMAP_128 = None

        self.LOSS_REGRESSION = None
        self.LOSS_TMP = 0.


    def set_input(self, inputs):
        self.INPUT = inputs['IMAGE'].to(self.device)
        self.INPUT_SHOW = inputs['IMAGE_SHOW'].to(self.device)
        self.LABEL = inputs['LABEL']
        self.LABEL_HEATMAP_512 = inputs['HEATMAP'].to(self.device)
        self.LABEL_HEATMAP_128 = F.max_pool2d(F.max_pool2d(inputs['HEATMAP'].to(self.device),2),2)


    def forward(self):
        self.OUTPUT_HEATMAPS = self.netG.forward(self.INPUT)
        self.SIM_HEATMAP_512 = F.upsample_nearest(self.OUTPUT_HEATMAPS, (512, 512))

        self.SIM_HEATMAP_512 = torch.clamp(self.SIM_HEATMAP_512,min=0,max=1.)

        HEATMAP_POINT = tools_plot.GetCenterfromHeatMap(self.SIM_HEATMAP_512)
        HEATMAP_POINT = torch.from_numpy(HEATMAP_POINT).to(self.device)
        self.OUTPUT_DIST = tools_plot.add_prediction_to_image_batch(self.INPUT_SHOW,HEATMAP_POINT,self.LABEL)



    def test_forward(self):
        self.test_result = []
        with torch.no_grad():
            PREDICTION_HEATMAP = F.upsample_nearest(self.netG.forward(self.INPUT),(512,512))
            HEATMAP_POINT = tools_plot.GetCenterfromHeatMap(PREDICTION_HEATMAP)
            HEATMAP_POINT = torch.from_numpy(HEATMAP_POINT).to(self.device)
            self.test_result.append(["PREDICTION", HEATMAP_POINT])
            self.test_result.append(['PREDICTION_DIST',tools_plot.add_prediction_to_image_base(self.INPUT_SHOW,HEATMAP_POINT,self.LABEL)])
            self.test_result.append(['PREDICTION_HEATMAP',PREDICTION_HEATMAP])



    def backward_G(self):
        LOSS_OLD = self.cretition(self.OUTPUT_HEATMAPS,self.LABEL_HEATMAP_128)
        LOSS_TMP = torch.pow(np.e,(-20000*LOSS_OLD))
        self.LOSS_REGRESSION = torch.min(LOSS_OLD,LOSS_TMP)
        self.LOSS_REGRESSION = self.cretition(self.OUTPUT_HEATMAPS,self.LABEL_HEATMAP_128)
        self.LOSS_REGRESSION.backward()


    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


