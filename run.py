import time
from datasets import create_dataset
from modules import create_model
from utils.visdom.visualizer import Visualizer
from utils import startup
import os
import utils.tools as util
import numpy as np
import evaluation

def train(config):
    dataset = create_dataset(config)
    model = create_model(config)
    model.setup(config)
    dataset_size = len(dataset)  # get the size of dataset
    print('The number of training images = %d' % dataset_size)

    test_config = config.copy()
    test_config['status'] = 'test'
    test_config['num_threads'] = 1
    test_dataset = create_dataset(test_config)
    test_dataset_size = len(test_dataset)
    print('The number of testing images = %d' % test_dataset_size)

    visualizer = Visualizer(config)  # create visualizer to show/save iamge


    total_iters = 0  # total iteration for datasets points
    t_data = 0

    if int(config['resume_epoch']) > 0:
        print("\n resume traing from rpoch " + str(int(config['resume_epoch']))+" ...")
        model.resume_scheduler(int(config['resume_epoch']))
        model.load_networks(config['resume_epoch'])
        model.load_optimizers(config['resume_epoch'])

    # outter iteration for differtent epoch; we save module via <epoch_count> and <epoch_count>+<save_latest_freq> options
    for epoch in range(int(config['resume_epoch'])+1, int(config['epoch']) +1):
        epoch_start_time = time.time()  # note the starting time for current epoch
        iter_data_time = time.time()  # note the starting time for datasets iteration
        epoch_iter = 0  # iteration times for current epoch, reset to 0 for each epoch

        #innear iteration for single epoch
        for i, data in enumerate(dataset):
            iter_start_time = time.time()  # note the stating time for current iteration
            if total_iters % int(config['print_freq']) == 0:  # note during time each <print_freq> times iteration
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters = total_iters + int(config['train_batch_size'])
            epoch_iter = epoch_iter + int(config['train_batch_size'])
            model.set_input(data)  # push loading image to the module
            model.optimize_parameters()  # calculate loss, gradient and refresh module parameters

            if total_iters % int(config['display_freq']) == 0:  # show runing result in visdom each <display_freq> iterations
                save_result = total_iters % int(config['update_html_freq']) == 0  # save runing result to html each <update_html_freq> iteartions
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % int(config['print_freq']) == 0:  # print/save training loss to console each <print_freq> iterations
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / int(config['train_batch_size'])
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if int(config['display_id']) > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        if epoch % int(config['save_epoch_freq']) == 0:  # save module each <save_epoch_freq> epoch iterations
            print('saving the module at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
            model.save_optimizers(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, int(config['epoch']), time.time() - epoch_start_time))

        val(config=test_config, epoch=epoch, dataset=test_dataset,model=model)

        # update learning rate after each epoch
        model.update_learning_rate()


def val(config,epoch,dataset,model):

    result_root_path = os.path.join(config['checkpoints_dir'], config['name'], config['results_dir'],"epoch"+str(epoch))
    util.mkdir(result_root_path)

    model.eval()
    save_npy = np.ndarray(shape=(dataset.__len__() + 1, 2), dtype=np.float)
    save_npy[0][0], save_npy[0][1] = -1, -1
    print("Start evaluating epoch "+str(epoch)+"...")

    for i, data in enumerate(dataset):
        model.set_input(data)  # push test datasets to module
        model.test()  # forward module

        datapoints = (model.test_result[0][1]).cpu().data.numpy()
        index = data["PATH"].cpu().data.numpy()[0]
        save_npy[index][0], save_npy[index][1] = datapoints[0][0], datapoints[0][1]

        dist_img = model.test_result[1][1]
        util.save_image(util.tensor2im(dist_img),os.path.join(result_root_path,str(index)+".png"))

    model.train()

    np.save(os.path.join(result_root_path, 'regression.npy'), save_npy)
    l2_dist,easy_dist,hard_dist = evaluation.evaluate_detailed(save_npy)
    text = open(os.path.join(config['checkpoints_dir'], config['name'], config['results_dir'],"evaluation.txt"),"a+")
    text.writelines("EPOCH "+str(epoch)+": "+str(round(l2_dist,4))+"   "+str(round(easy_dist,4)) + '   '+str(round(hard_dist,4))+"\n")
    text.close()
    print("Testing npy result have been saved! Evaluation distance: "+str(round(l2_dist))+"("+str(round(easy_dist))+","+str(round(hard_dist))+")")



if __name__ == '__main__':
    configs_stage1 = startup.SetupConfigs(config_path='configs/TCLNET_STAGE1.yaml')
    configs_stage2 = startup.SetupConfigs(config_path='configs/TCLNET_STAGE2.yaml')
    configs_stage1 = configs_stage1.setup()
    configs_stage2 = configs_stage2.setup()

    if (configs_stage1['status'] == "train") and (configs_stage2['status'] == 'train'):
        train(configs_stage1)
        train(configs_stage2)


