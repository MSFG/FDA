import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from models.AutoEncoder import AutoEncoder
from models.ResNet import resnet50, resnet18
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.XceptionNet import xception
from utils import FullDataset1, FullDataset2, RandomDataset, Val
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

save_img_num = 0
save_id = 0
save_id2 = 0


def train_model(model, AE, exp_name, dataset, criterion, optimizer, num_epochs, runs_path, Sampling):
    model.to(device)
    AE.to(device)
    AE.eval()

    print("[Waiting for loading validation datasets...]")
    # root_dir = "/data1/whm/00-Dataset/CNNDetection/Test"
    root_dir = "/data1/Jianwei-Fei/00-Dataset/01-Images/CNNDetection/Test"
    ValSet_stylegan = FullDataset1(root=root_dir + "/stylegan")
    ValSet_stylegan2 = FullDataset1(root=root_dir + "/stylegan2")
    ValSet_biggan = FullDataset2(root=root_dir + "/biggan")
    ValSet_cyclegan = FullDataset1(root=root_dir + "/cyclegan")
    ValSet_stargan = FullDataset2(root=root_dir + "/stargan")
    ValSet_gaugan = FullDataset2(root=root_dir + "/gaugan")
    ValSet_list = [ValSet_stylegan, ValSet_stylegan2, ValSet_biggan, ValSet_cyclegan, ValSet_stargan, ValSet_gaugan]

    if Sampling == "full":
        dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # Variables used to record the best performance.
    best_mAcc = 0
    best_mAP = 0
    best_average = 0  # average = (mAcc + mAP) / 2
    best_mAcc_epoch = 0
    best_mAP_epoch = 0
    best_average_epoch = 0
    best_average_tag = ""

    num_iter = len(dataset) // BATCH_SIZE
    with tqdm(total=num_epochs * num_iter) as pbar:
        for epoch in range(1, num_epochs + 1):
            print("Epoch {}/{}".format(epoch, num_epochs))
            print("-" * 10)
            start = time.time()
            model.train()

            if Sampling == "random":
                dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True,
                                        worker_init_fn=lambda id: np.random.seed(id + epoch * 8))

            for i, data in enumerate(dataloader, 1):
                image, label = data
                images = image.to(device)
                labels = label.to(device)

                """ Choose which experiment to conduct according to parameter "exp_name".  """
                if exp_name == "baseline":  # baseline, no perturbations.
                    images_perturbed = images
                elif "Mut" in exp_name:  # e.g. Mut5, Mut10, Mut100
                    factor = int(exp_name.split("Mut")[1])
                    images_perturbed = Perturb_fingerprints_mut(AE, images, labels, factor)
                elif "Mix" in exp_name:  # e.g. Mix2_50, Mix2_0
                    mix_num = int(exp_name[3])
                    original_prop = int(exp_name.split("_")[1])
                    images_perturbed = Perturb_fingerprints_mix(AE, images, labels, mix_num, original_prop)
                else:
                    print("The experiment name was set incorrectly!")

                inputs = images_perturbed
                inputs = inputs.to(device)

                optimizer.zero_grad()
                AE.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                acc = torch.sum(torch.eq(torch.ge(outputs.detach().cpu(), torch.full_like(label, 0.)), label))
                train_log = "[%03d/%03d, %03d/%03d] Loss: %.04f Acc: %.04f" % (epoch, num_epochs, i, num_iter, loss.item(), acc.cpu().numpy() / BATCH_SIZE)

                with open(os.path.join(runs_path, "training-logs.txt"), "a", encoding="utf-8") as f:
                    f.write(train_log)
                    f.write("\n")
                tqdm.write(train_log)

                pbar.update(1)

            # Validating after one epoch.
            print("[Start validating...]")
            Acc_list, AP_list = [], []
            for i in range(len(ValSet_list)):
                acc, AP = Val(model, ValSet_list[i])
                acc *= 100  # -> percentage
                AP *= 100
                Acc_list.append(acc)
                AP_list.append(AP)

            mAcc = (sum(Acc_list)) / 6
            mAP = (sum(AP_list)) / 6
            average = (mAcc + mAP) / 2

            if mAcc > best_mAcc:
                best_mAcc = mAcc
                best_mAcc_epoch = epoch
            if mAP > best_mAP:
                best_mAP = mAP
                best_mAP_epoch = epoch
            if average > best_average:
                best_average = average
                best_average_epoch = epoch

            tag = "Epoch-%03d-Trainingloss-%.03f  -Acc-AP  stylegan: -%.1f-%.1f  stylegan2: -%.1f-%.1f  " \
                  "biggan: -%.1f-%.1f  cyclegan: -%.1f-%.1f  stargan: -%.1f-%.1f  gaugan: -%.1f-%.1f \n" \
                  "Avg: -%.1f-%.1f" % (epoch, loss.item(), Acc_list[0], AP_list[0], Acc_list[1], AP_list[1], Acc_list[2],
                  AP_list[2], Acc_list[3], AP_list[3], Acc_list[4], AP_list[4], Acc_list[5], AP_list[5], mAcc, mAP)
            print("-" * 50)
            with open(os.path.join(runs_path, "val-logs.txt"), "a", encoding="utf-8") as f:
                f.write(tag)
                f.write("\n\n")

            if epoch == best_average_epoch:
                best_average_tag = tag

            torch.save({"model": model.state_dict()}, os.path.join(runs_path, "models", f"model-{epoch}.pth.tar"))
            time_consumed = time.time() - start
            time_log = "Epoch {} consumed {:.0f}h {:.0f}m {:.0f}s".format(epoch, time_consumed // 3600,
                                                                          time_consumed // 60, time_consumed % 60)
            print(time_log)

            best_log = "best_mAcc: %.1f, epoch=%d; best_mAP: %.1f, epoch=%d;\n\nbest_Average: %.1f, epoch=%d.\n" % \
                       (best_mAcc, best_mAcc_epoch, best_mAP, best_mAP_epoch, best_average, best_average_epoch)
            with open(os.path.join(runs_path, "best-logs.txt"), "w", encoding="utf-8") as f:
                f.write(best_log)
                f.write(best_average_tag)

    # show the best performance
    os.rename(runs_path, runs_path + "（%.1f）" % best_average)


def Perturb_fingerprints_mut(ae, images, labels, factor):
    images_perturbed_list = []
    factor = 100 / factor  # e.g. factor=5 -> factor=100/5=20
    for image, label in zip(images, labels):
        image = image.unsqueeze(0)
        if label == 1 and (random.randint(0, 100) < perturb_prob):
            image_ae = ae(image)
            fingerprint = image - image_ae

            # Perturbations.
            multy = random.randint(-100, 100) / factor  # original: 20
            fingerprint = fingerprint * multy

            # Add the perturbed fingerprints back to the images processed by ae32.
            image_perturbed = image_ae + fingerprint

            image_perturbed = torch.clamp(image_perturbed, 0, 1)  # clamp会返回一个新变量，而不是直接对原变量处理！！！

            # Save 100 perturbed images for visualization.
            save_perturbed_images(image_perturbed)

        else:
            image_perturbed = image

        images_perturbed_list.append(image_perturbed)
    images_perturbed = torch.cat(images_perturbed_list, dim=0)
    images_perturbed = images_perturbed.to(device)

    return images_perturbed


def Perturb_fingerprints_mix(ae, images, labels, mix_num, original_prop):
    images_perturbed_list = []
    seg_content = []
    fingers = []

    for image, label in zip(images, labels):
        image = image.unsqueeze(0)
        image_ae = ae(image)
        fingerprint = image - image_ae
        seg_content.append(image_ae)
        fingers.append(fingerprint)

    for cont, fing, label in zip(seg_content, fingers, labels):
        if label == 1 and (random.randint(0, 100) < perturb_prob):
            select_fingers_num = mix_num
            rate = [random.randint(original_prop, 100)]  # rate[0] -> original fingerprint proportion
            sum = 0
            for i in range(select_fingers_num - 2):  # rate[1:-1]
                sum += rate[i]
                rate.append(random.randint(0, 100 - sum))
            rate.append(100 - sum)  # rate[-1] -> the last fingerprint proportion
            select = []
            for i in range(select_fingers_num - 1):  # Randomly choose fingers to mix up.
                select.append(random.randint(0, len(fingers) - 1))
            mixed_fingerprint = fing * rate[0]  # Mix up fingers.
            for i in range(select_fingers_num - 1):
                mixed_fingerprint += fingers[select[i]] * rate[i + 1]

            mixed_fingerprint = mixed_fingerprint / 100

            image_perturbed = cont + mixed_fingerprint

            # Save 100 perturbed images for visualization.
            save_perturbed_images(image_perturbed)

        else:
            image_perturbed = cont + fing

        image_perturbed = torch.clamp(image_perturbed, 0, 1)

        images_perturbed_list.append(image_perturbed)
    images_perturbed = torch.cat(images_perturbed_list, dim=0)
    images_perturbed = images_perturbed.to(device)

    return images_perturbed


def save_perturbed_images(image_perturbed):
    # Save 100 perturbed images for visualization.
    global save_img_num
    if not os.path.exists(f"{runs_path}/perturbed_imgs"):
        os.makedirs(f"{runs_path}/perturbed_imgs")
    toImg = transforms.ToPILImage()
    if save_img_num < 100:
        img = image_perturbed.squeeze(0)
        img = toImg(img)
        img.save(f"{runs_path}/perturbed_imgs/{save_img_num}.jpg")
        save_img_num += 1


def save_perturbations_related(image, finger, image_perturbed, finger_perturbed):
    # Save 100 perturbed images for visualization.
    global save_img_num
    if not os.path.exists(f"{runs_path}/perturbed_imgs/image"):
        os.makedirs(f"{runs_path}/perturbed_imgs/image")
        os.makedirs(f"{runs_path}/perturbed_imgs/finger")
        os.makedirs(f"{runs_path}/perturbed_imgs/image_perturbed")
        os.makedirs(f"{runs_path}/perturbed_imgs/finger_perturbed")
    toImg = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale()])
    if save_img_num < 100:
        img = image.squeeze(0)
        img = toImg(img)
        img.save(f"{runs_path}/perturbed_imgs/image/image{save_img_num}.jpg", quality=95)
        img = finger.squeeze(0)
        img = toImg(img)
        img.save(f"{runs_path}/perturbed_imgs/finger/finger{save_img_num}.jpg", quality=95)
        img = image_perturbed.squeeze(0)
        img = toImg(img)
        img.save(f"{runs_path}/perturbed_imgs/image_perturbed/image_perturbed{save_img_num}.jpg", quality=95)
        img = finger_perturbed.squeeze(0)
        img = toImg(img)
        img.save(f"{runs_path}/perturbed_imgs/finger_perturbed/finger_perturbed{save_img_num}.jpg", quality=95)
        save_img_num += 1


if __name__ == "__main__":
    from params import *

    if Sampling == "full":
        train_set = FullDataset1(root="/data1/Jianwei-Fei/00-Dataset/01-Images/CNNDetection/Train")
    elif Sampling == "random":
        train_set = RandomDataset(LENGTH=LENGTH, class_num=training_classes, root="/data1/Jianwei-Fei/00-Dataset/01-Images/CNNDetection/Train")

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

    ae32 = AutoEncoder()
    ae32.load_state_dict(torch.load(ae_path))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Learning_Rate)

    print("[Start training the model...]")
    train_model(model, ae32, exp_name, train_set, criterion, optimizer, EPOCH, runs_path, Sampling)
