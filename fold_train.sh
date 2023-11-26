 #!/bin/bash

python run/train.py exp_name=exp_fold_train_0 features=["anglez","enmo","hour_sin","hour_cos"] split=fold_0
python run/train.py exp_name=exp_fold_train_1 features=["anglez","enmo","hour_sin","hour_cos"] split=fold_1
python run/train.py exp_name=exp_fold_train_2 features=["anglez","enmo","hour_sin","hour_cos"] split=fold_2
python run/train.py exp_name=exp_fold_train_3 features=["anglez","enmo","hour_sin","hour_cos"] split=fold_3
python run/train.py exp_name=exp_fold_train_4 features=["anglez","enmo","hour_sin","hour_cos"] split=fold_4