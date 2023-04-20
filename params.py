import os
from datetime import datetime

BATCH_SIZE = 32
SCALE = 256
ITER = 200  # Original: 500
LENGTH = BATCH_SIZE * ITER
Learning_Rate = 0.0001  # lr=0.001 -> bad performance


""" PARAMS which are often modified """
perturb_prob = 50
# 1、baseline    2、Mut5    3、Mix2_50    ...
exp_name = "Mut5"
training_classes = 20  # 1, 2, 4, 20
ae_path = "/data1/Huaming-Wang/Print_Aug/wD_ae_epoch13.pt"
# ae_path = "/data1/Huaming-Wang/Print_Aug/GANprintAug1116/GANprintR/ae_trained/ae_epoch43.pt"
Model = "ResNet50"

Sampling = "random"  # random & full
if Sampling == "full":
    EPOCH = 20
elif Sampling == "random":
    EPOCH = 50

now = datetime.now()
dt_string = now.strftime("%d%H%M%S")

if exp_name == "baseline":
    runs_path = f"RUNS_baseline/{Model}-Class{training_classes}-{dt_string}-{exp_name}"
else:
    runs_path = f"RUNS_woD({exp_name[:3]})/runs_wD({exp_name})/{Model}-Class{training_classes}-{dt_string}-{exp_name}-Prob{perturb_prob}"
if not os.path.exists(runs_path):
    os.makedirs(runs_path)

if not os.path.exists(os.path.join(runs_path, "models")):
    os.makedirs(os.path.join(runs_path, "models"))
