# Consistent Region Features Learning for Lifelong Person Re-Identification(ConRFL)
The official implementation for the [Consistent Region Features Learning for Lifelong Person Re-Identification].

# Install 
## base installation
follow [PTKP](https://github.com/g3956/PTKP)
## install Crank
```
cd ./reid/evaluation_metrics/rank_cylib; make all
python setup.py develop

```
### Problem
- ImportError: No module named '_swigfaiss
following this: [link](https://github.com/facebookresearch/faiss/issues/821)
- install Crank failed
execute the commands step by step in Makefile
- ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found
following this: [link](https://github.com/AllenDowney/ThinkStats2/issues/92)
```angular2html
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anaconda/anaconda3/lib/
```
# Train and test

To train with ConRFL, simply run
```bash
# train ConRFL
python continual_part_train.py --config_file configs/replay/coseg_model.yaml
# evaluate ConRFL
CUDA_VISIBLE_DEVICES=0 python continual_part_train.py --config_file configs/replay/coseg_model.yaml --evaluate;
# evaluate ConRFL on merged Test dataset
CUDA_VISIBLE_DEVICES=0 python continual_part_train_test_all.py --config_file configs/replay/coseg_model.yaml;

# train ConRFL with CKD
python continual_part_TMTKD.py --config_file configs/replay/TMTLearnForgetKD_kd_P_no_assignmodel.yaml
# evaluate ConRFL with CKD
CUDA_VISIBLE_DEVICES=0 python continual_part_TMTKD.py --config_file configs/replay/TMTLearnForgetKD_kd_P_no_assignmodel.yaml --evaluate;
# evaluate ConRFL on merged Test dataset
CUDA_VISIBLE_DEVICES=0 python continual_part_TMTKD_test_all.py --config_file configs/replay/TMTLearnForgetKD_kd_P_no_assignmodel.yaml;
```
