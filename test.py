import torch
import torch.nn as nn
import torch.optim as optim
from models.AutoEncoder import AutoEncoder
from models.ResNet import resnet50, resnet18
from models.XceptionNet import xception
from utils import FullDataset1, FullDataset2, RandomDataset, Val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def test_model(model, model_path, runs_path):
    model.to(device)
    model.load_state_dict(torch.load(model_path)["model"])

    print("[Waiting for loading validation datasets...]")
    # root_dir = "/data1/whm/00-Dataset/CNNDetection/Test"
    root_dir = "/data1/Jianwei-Fei/00-Dataset/01-Images/CNNDetection/Test"
    ValSet_biggan = FullDataset2(root=root_dir + "/biggan")
    ValSet_cyclegan = FullDataset1(root=root_dir + "/cyclegan")
    ValSet_deepfake = FullDataset2(root=root_dir + "/deepfake")
    ValSet_gaugan = FullDataset2(root=root_dir + "/gaugan")
    ValSet_progan = FullDataset1(root=root_dir + "/progan")
    ValSet_stargan = FullDataset2(root=root_dir + "/stargan")
    ValSet_stylegan = FullDataset1(root=root_dir + "/stylegan")
    ValSet_stylegan2 = FullDataset1(root=root_dir + "/stylegan2")
    ValSet_list = [ValSet_progan, ValSet_stylegan, ValSet_stylegan2, ValSet_biggan,
                   ValSet_cyclegan, ValSet_stargan, ValSet_gaugan, ValSet_deepfake]

    print("[Start testing...]")
    Acc_list, AP_list = [], []
    for i in range(len(ValSet_list)):
        acc, AP = Val(model, ValSet_list[i])
        Acc_list.append(acc)
        AP_list.append(AP)

    avg_acc6 = (sum(Acc_list)-Acc_list[0]-Acc_list[7]) / 6
    mAP6 = (sum(AP_list)-AP_list[0]-AP_list[7]) / 6
    average6 = (avg_acc6 + mAP6) / 2

    tag6 = "-Acc-AP  stylegan: -%.4f-%.4f " \
           "stylegan2: -%.4f-%.4f  biggan: -%.4f-%.4f  cyclegan: -%.4f-%.4f  " \
           "stargan: -%.4f-%.4f  gaugan: -%.4f-%.4f \nAvg: -%.4f-%.4f" % (
            Acc_list[1], AP_list[1], Acc_list[2], AP_list[2],
            Acc_list[3], AP_list[3], Acc_list[4], AP_list[4], Acc_list[5], AP_list[5], Acc_list[6],
            AP_list[6], avg_acc6, mAP6)
    print(tag6)
    print("-" * 50)

    with open(os.path.join(runs_path, "test-logs-6.txt"), "a", encoding="utf-8") as f:
        f.write(tag6)
        f.write("\n")

    best_log6 = "mean Acc: %.4f; mean AP: %.4f; Average: %.4f" % (avg_acc6, mAP6, average6)
    with open(os.path.join(runs_path, "Avg-logs6.txt"), "w", encoding="utf-8") as f:
        f.write(best_log6)
        f.write("\n")


if __name__ == "__main__":
    from params import *

    # Create Model.
    if Model == "XceptionNet":
        model = xception(num_classes=1000, pretrained=True)
    elif Model == "ResNet18":
        model = resnet18(num_classes=1000, pretrained=True)
    elif Model == "ResNet50":
        model = resnet50(num_classes=1000, pretrained=True)
    model.num_classes = 1
    dim_feats = model.fc.in_features
    model.fc = nn.Linear(dim_feats, 1)

    model_path = "/data1/Huaming-Wang/Print_Aug/Cross-Model2/runs_mut(Mix2_50)/ResNet50-Class4-10162714-mix-Prob50/models/model-22.pth.tar"

    runs_path = "/data1/Huaming-Wang/Print_Aug/Cross-Model2/runs_mut(Mix2_50)/ResNet50-Class4-10162714-mix-Prob50"

    print("[Start training the model...]")
    test_model(model, model_path, runs_path)
