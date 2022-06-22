# Bi-Calibration Networks for Weakly-Supervised Video Representation Learning

## Models
This repository includes the source files of the proposed bi-calibration Caffe loss function layer and the network structure for weakly-supervised video representation learning of our Bi-Calibration Networks (BCN).

You can easily integrate the loss function layer into your own Caffe repository. 
You could also check the design of the calibration selection and the process of the optimization in these codes.

## YOVO-3M and YOVO-10M
The newly-created large-scale web video datasets YOVO-3M and YOVO-10M are attached in the `./datasets`. Please check the `DATASETS.md` in the folder for more details.

The download link of related files is https://pan.baidu.com/s/17wR4uYNqMfjLUo0KhXJBuQ (code: dzel) 

## Prerequisites

- Caffe 
- CUDA 9.0
- cuDNN 7.0


## Caffe Loss Function Files
- caffe_layers/include/softmax_bi_calibrate_loss_layer.hpp
- caffe_layers/src/softmax_bi_calibrate_loss_layer.cpp
- caffe_layers/caffe.proto

These files include the loss function for query/text classification in the bi-calibration design.


## Network Structure Files
- networks_proto/deploy_bcn_resnet50_yovo3m.prototxt
- networks_proto/deploy_bcn_resnet50_yovo3m_solver.prototxt

These files include the network structure of BCN (`deploy_bcn_resnet50_yovo3m.prototxt`) and the hyper-parameter settings of the second training stage (`deploy_bcn_resnet50_yovo3m_solver.prototxt`).

## Citation

If you use these models and datasets in your research, please cite:

    @article{Long:BCN22,
      title={Bi-Calibration Networks for Weakly-Supervised Video Representation Learning},
      author={Fuchen Long and Ting Yao and Zhaofan Qiu and Xinmei Tian and Jiebo Luo and Tao Mei},
      journal={arXiv preprint arXiv:2206.10491},
      year={2022}
    }