# train ConRFL
python continual_part_train.py --config_file configs/no_replay_list1/coseg_model.yaml
# evaluate ConRFL
CUDA_VISIBLE_DEVICES=0 python continual_part_train.py --config_file configs/no_replay_list1/coseg_model.yaml --evaluate;
# evaluate ConRFL on merged Test dataset
CUDA_VISIBLE_DEVICES=0 python continual_part_train_test_all.py --config_file configs/no_replay_list1/coseg_model.yaml;

# train ConRFL with CKD
python continual_part_TMTKD.py --config_file configs/no_replay_list1/TMTLearnForgetKD_kd_P_no_assignmodel.yaml
# evaluate ConRFL with CKD
CUDA_VISIBLE_DEVICES=0 python continual_part_TMTKD.py --config_file configs/no_replay_list1/TMTLearnForgetKD_kd_P_no_assignmodel.yaml --evaluate;
# evaluate ConRFL on merged Test dataset
CUDA_VISIBLE_DEVICES=0 python continual_part_TMTKD_test_all.py --config_file configs/no_replay_list1/TMTLearnForgetKD_kd_P_no_assignmodel.yaml;

