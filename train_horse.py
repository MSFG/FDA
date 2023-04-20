import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from models.AutoEncoder import AutoEncoder
from models.ResNet import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    root_dir = "/data1/whm/00-Dataset/CNNDetection/Test"
    ValSet_progan = FullDataset1(root=root_dir + "/progan", exp="cross-category")

    if Sampling == "full":
        dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    best_Acc = 0
    best_AP = 0
    best_average = 0
    best_epoch_1 = 0
    best_epoch_2 = 0
    best_epoch_3 = 0

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
                elif exp_name == "noise":
                    images_perturbed = Perturb_fingerprints_noise(AE, images, labels)
                elif exp_name == "mut":
                    images_perturbed = Perturb_fingerprints_mut(AE, images, labels)
                elif exp_name == "mix":
                    images_perturbed = Perturb_fingerprints_mix(AE, images, labels)

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
            acc, AP = Val(model, ValSet_progan)

            average = (acc + AP) / 2

            if acc > best_Acc:
                best_Acc = acc
                best_epoch_1 = epoch
            if AP > best_AP:
                best_AP = AP
                best_epoch_2 = epoch
            if average > best_average:
                best_average = average
                best_epoch_3 = epoch

            tag = "Epoch-%03d-Trainingloss-%.04f  -Acc-AP  progan(except horse): -%.4f-%.4f  Avg: -%.4f" % (
                       epoch, loss.item(), acc, AP, average)

            print(tag)
            print("-" * 50)
            with open(os.path.join(runs_path, "val-logs.txt"), "a", encoding="utf-8") as f:
                f.write(tag)
                f.write("\n")
            torch.save({"model": model.state_dict()}, os.path.join(runs_path, "models", f"model-{epoch}.pth.tar"))
            time_consumed = time.time() - start
            time_log = "Epoch {} consumed {:.0f}h {:.0f}m {:.0f}s".format(epoch, time_consumed // 3600,
                                                                            time_consumed // 60, time_consumed % 60)
            print(time_log)

            best_log = "best_Acc: %.4f, epoch=%d; best_AP: %.4f, epoch=%d; best_Average: %.4f, epoch=%d." % \
                       (best_Acc, best_epoch_1, best_AP, best_epoch_2, best_average, best_epoch_3)
            with open(os.path.join(runs_path, "best-logs.txt"), "w", encoding="utf-8") as f:
                f.write(best_log)
                f.write("\n")


def Perturb_fingerprints_noise(ae, images, labels):
    images_perturbed_list = []
    for image, label in zip(images, labels):
        image = image.unsqueeze(0)
        if label == 1 and (random.randint(0, 100) < perturb_prob):
            image_ae = ae(image)
            fingerprint = image - image_ae

            strength = random.randint(1, 100) / 10000  # original: 10000
            perturbation = strength * torch.normal(0, 1, size=(1, 3, fingerprint.shape[-1], fingerprint.shape[-1]))
            perturbation = perturbation.to(device)
            fingerprint += perturbation

            multy = random.randint(-100, 100) / 50
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


def Perturb_fingerprints_mut(ae, images, labels):
    images_perturbed_list = []
    for image, label in zip(images, labels):
        image = image.unsqueeze(0)
        if label == 1 and (random.randint(0, 100) < perturb_prob):
            image_ae = ae(image)
            fingerprint = image - image_ae

            multy = random.randint(-100, 100) / 20  # original: 20
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


def Perturb_fingerprints_mix(ae, images, labels):
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
            select_fingers_num = 2
            rate = [random.randint(0, 10000)]  # rate[0] -> original fingerprint proportion
            sum = 0
            for i in range(select_fingers_num - 2):  # rate[1:-1]
                sum += rate[i]
                rate.append(random.randint(0, 10000 - sum))
            rate.append(10000 - sum)  # rate[-1] -> the last fingerprint proportion
            select = []
            for i in range(select_fingers_num - 1):  # Randomly choose fingers to mix up.
                select.append(random.randint(0, len(fingers) - 1))
            mixed_fingerprint = fing * rate[0]  # Mix up fingers.
            for i in range(select_fingers_num - 1):
                mixed_fingerprint += fingers[select[i]] * rate[i + 1]

            mixed_fingerprint = mixed_fingerprint / 10000

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


def Perturb_images_noise(images, labels):
    images_perturbed_list = []
    for image, label in zip(images, labels):
        image = image.unsqueeze(0)
        if label == 1 and (random.randint(0, 100) < perturb_prob):
            strength = random.randint(1, 100) / 10000  # original: 10000
            perturbation = strength * torch.normal(0, 1, size=(1, 3, image.shape[-1], image.shape[-1]))
            perturbation = perturbation.to(device)
            image_perturbed = image + perturbation
            image_perturbed = torch.clamp(image_perturbed, 0, 1)
            # Save 100 perturbed images for visualization.
            save_perturbed_images(image_perturbed)
        else:
            image_perturbed = image
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


if __name__ == "__main__":
    from params import *

    if Sampling == "full":
        train_set = FullDataset1(root="/data1/whm/00-Dataset/CNNDetection/Train")
    elif Sampling == "random":
        train_set = RandomDataset(LENGTH=LENGTH, root="/data1/whm/00-Dataset/CNNDetection/Train")

    # Create Model.
    # if Model == "XceptionNet":
    #     model = xception(num_classes=1000, pretrained=True)
    # elif Model == "ResNet18":
    #     model = resnet18(num_classes=1000, pretrained=True)
    if Model == "ResNet50":
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
