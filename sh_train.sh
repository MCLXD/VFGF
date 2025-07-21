#RGB train
python main.py --gpu_id 0 --batch_size 128 --mode train --modality rgb --hidden 1024 --feat_in 1024 --lr 0.05 --wd 1e-5 --epoch 100 --pre_train
python main.py --gpu_id 0 --batch_size 128 --mode train --modality rgb --hidden 1024 --feat_in 1024 --lr 0.05 --wd 1e-5 --epoch 200 
#RGB validate
#python main.py --gpu_ids 0 --batch_size 128 --mode validate --modality rgb --hidden 1024 --feat_in 1024 --best_model_name RV62714 --resume_timestamp pre_trained

#FLOW train
#python main.py --gpu_id 0 --batch_size 128 --mode train --modality flow --hidden 1024 --feat_in 1024 --lr 0.05 --wd 1e-5 --epoch 300 --pre_train
#python main.py --gpu_ids 0 --batch_size 128 --mode validate --modality flow --hidden 1024 --feat_in 1024 --best_model_name F612.1 --resume_timestamp pre_trained

#OBJ train
#python main.py --gpu_id 0 --batch_size 128 --mode train --modality rgb --hidden 352 --feat_in 352 --lr 0.05 --wd 1e-5 --epoch 100 --pre_train
#OBJ validate
#python main.py --gpu_ids 0 --batch_size 128 --mode validate --modality obj --hidden 352 --feat_in 352 --best_model_name O612.1 --resume_timestamp pre_trained

#FUSION train
#python main.py --gpu_id 0 --batch_size 16 --mode train --modality fusion --lr 0.1 --wd 1e-5 --epoch 200
