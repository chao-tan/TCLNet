# TCLNet: Learning to Locate Typhoon Center using Deep Neural Network


This repository contains the source code, dataset and pretrained model for TCLNet, provided by [Chao Tan](https://chao-tn.gitee.io).

The paper is avaliable for download [here](https://arxiv.org/abs/2010.01282). 
Click [here](https://chao-tan.gitee.io/projects/tcl-net/project-page.html) for more details.


***

## Dataset & Pretrained Model
TCLD (Typhoon Center Location Dataset) is a brand new typhoon center location dataset for deep learning research.
It contains 1809 grayscale images for training and another 319 images for testing.
Our TCLD dataset is available for download at [TianYiCloud(600MB)](https://cloud.189.cn/t/vYfqyu2iMbyu) or [BaiduCloud(600MB)](https://pan.baidu.com/s/1ej_35e1hJgynOrDHjbmeYQ)(extraction code: ixmb).      
You can get the TCLD dataset at any time but only for scientific research. 
At the same time, please cite our work when you use the TCLD dataset.

The pretrained model of our TCLNet on TCLD dataset can be download at [TianYiCloud](https://cloud.189.cn/t/Ar2m2qfm2eaa) or [BaiduCloud](https://pan.baidu.com/s/1bTUAi0KEYbwoWUgQ9JysFg) (extraction code: 8l1j).


        
## Prerequisites
* Python 3.7
* PyTorch >= 1.4.0
* opencv 0.4
* PyQt 4
* numpy
* visdom


## Training
1. Please download and unzip TCLD dataset and place it in ```datasets/data``` folder.
2. Run ```python -m visdom.server"``` to activate visdom server.
3. Run ```python run.py``` to start training from scratch.
4. You can easily monitor training process at any time by visiting ```http://localhost:8097``` in your browser.


## Testing
1. For TCLD dataset, please download and unzip pretrained model and place it in ```checkpoints``` folder.
2. Replace the test data in the ```daatsets/data/TCLD/TEST_INPUT"``` folder with your own data (optional).
3. Run ```python test.py``` to start testing.
4. The results of the testing will be saved in the ```checkpoint/TCLNET/evaluation"``` directory.

## Citation

@inproceedings{      
&nbsp;&nbsp;&nbsp;&nbsp;  title={{TCLNet}: Learning to Locate Typhoon Center Using Deep Neural Network},         
&nbsp;&nbsp;&nbsp;&nbsp;  author={Tan, Chao},         
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle={IGARSS},        
&nbsp;&nbsp;&nbsp;&nbsp;  year={2021},        
&nbsp;&nbsp;&nbsp;&nbsp;  note={to appear},       
}
