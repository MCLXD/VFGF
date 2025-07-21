# VFGF The source code will be released after the paper is officially accepted！
Virtual Frame-Augmented Guided Prediction Framework for Long-Term Egocentric Activity Forecasting
# Introduction
This is a Pytorch implementation of the model described in our paper:
    Xiangdong Long, Shuqin Wang and Yong Chen. VFGF:Virtual Frame-Augmented Guided Prediction Framework for Long-Term Egocentric Activity Forecasting.
# Dependencies
    PyTorch ≥ 1.10.2
    CUDA 9.0.176
    CuDNN 7.4.2
    Python 3.6.13

# Data
### EPIC-Kitchens dataset

For the raw data of the EPIC-Kitchens dataset, please refer to https://github.com/epic-kitchens/download-scripts to download.

For the three modality features (rgb, flow, obj), please refer to https://github.com/fpv-iplab/rulstm to download. After downloading, put them in the folder './data'.


# Training
### EPIC-Kitchens
- Pre_Train of FGM on rgb feature: python main.py --gpu_id 0 --batch_size 128 --mode train --modality rgb --hidden 1024 --feat_in 1024 --lr 0.05 --wd 1e-5 --epoch 100 --pre_train
- Train of model on rgb feature: python main.py --gpu_id 0 --batch_size 128 --mode train --modality rgb --hidden 1024 --feat_in 1024 --lr 0.05 --wd 1e-5 --epoch 200 
- Silimar commonds can be used for flow or obj features.
- For three modality features: python main.py --gpu_id 0 --batch_size 16 --mode train --modality fusion --lr 0.1 --wd 1e-5 --epoch 200
# Validation
###  Validation for Epic-Kitchen dataset
Please download the pre-trained model weigths from [Baidu]([https://pan.baidu.com/s/1n-5uCvD2VomkPiTgGlmkOQ] passward: '343x', and put them in the folder './results/EPIC/base_srl/pre_trained/'.

 - For rgb feature: python main.py --gpu_ids 0 --batch_size 128 --mode validate --modality rgb --hidden 1024 --feat_in 1024 --best_model_name model_name --resume_timestamp pre_trained
 - For three modality features, python main.py --gpu_ids 0 --batch_size 128 --mode validate --modality fusion --best_model_name model_name

# Citation
None
