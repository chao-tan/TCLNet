from datasets import create_dataset
from modules import create_model
from utils import startup
import os
import utils.tools as util
import numpy as np
import evaluation
import cv2



def test(config):
    config['num_threads'] = 1                     # only <num_threads = 1> supported when testing_usr
    config['flip'] = False                        # not allowed to flip image
    config['status'] = 'test'
    config['crop_scale'] = 1.0

    dataset = create_dataset(config)
    model = create_model(config)
    model.setup(config)

    result_root_path = os.path.join(config['checkpoints_dir'], config['name'], 'evaluation')
    util.mkdir(result_root_path)
    util.mkdir(os.path.join(result_root_path,'prediction_distance'))
    util.mkdir(os.path.join(result_root_path,'prediction_heatmap'))
    print(" create evaluate folder: " + result_root_path)

    # set module to testing_usr mode
    model.eval()

    save_npy = np.ndarray(shape=(dataset.__len__()+1,2),dtype=np.float)
    save_npy[0][0],save_npy[0][1] = -1,-1

    for i, data in enumerate(dataset):
        model.set_input(data)  # push test datasets to module
        model.test()  # forward module

        datapoints = (model.test_result[0][1]).cpu().data.numpy()
        index = data["PATH"].cpu().data.numpy()[0]
        save_npy[index][0],save_npy[index][1] = datapoints[0][0], datapoints[0][1]

        dist_img = model.test_result[1][1]
        util.save_image(util.tensor2im(dist_img), os.path.join(result_root_path,'prediction_distance', str(index) + ".png"))

        heatmap_img = model.test_result[2][1]
        util.save_image(util.tensor2im(heatmap_img),os.path.join(result_root_path, 'prediction_heatmap', str(index) + ".png"))

        print("Evaluate forward-- complete:" + str(i + 1) + "  total:" + str(dataset.__len__()))

    np.save(os.path.join(result_root_path,'regression.npy'),save_npy)
    l2_dist, easy_dist, hard_dist = evaluation.evaluate_detailed(save_npy)
    print("Testing npy result have been saved! Evaluation distance: " + str(round(l2_dist)) + "(" + str(round(easy_dist)) + "," + str(round(hard_dist)) + ")")


if __name__ == '__main__':
    configs = startup.SetupConfigs(config_path='configs/TCLNET_STAGE2.yaml')
    configs = configs.setup()
    test(configs)


