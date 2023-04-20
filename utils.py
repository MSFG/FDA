import time
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from params import *
from sklearn.metrics import average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# sample_id = 0
#
# # Save 100 samples for visualization.
# if not os.path.exists(f"{runs_path}/samples"):
#     os.makedirs(f"{runs_path}/samples")

preprocess = transforms.Compose([
    transforms.Resize((SCALE, SCALE)),
    transforms.ToTensor(),
])


def loader(path):
    global sample_id
    img_pil = Image.open(path).convert("RGB")
    img_tensor = preprocess(img_pil)
    # toImg = transforms.ToPILImage()
    # if sample_id < 100:
    #     img = toImg(img_tensor)
    #     img.save(f"{runs_path}/samples/{sample_id}.jpg")
    #     sample_id += 1
    return img_tensor


# for cyclegan, progan, stylegan, stylegan2.
# File Structure:
# cyclegan:
#   apple:
#       0_real:
#           xxx.png
#             ...
#           xxx.png
#       1_fake:
#           xxx.png
#             ...
#           xxx.png
#   horse:
#       0_real:
#             ...
#       1_fake:
#             ...
#   ...
class FullDataset1(Dataset):
    def __init__(self, root="/data1/whm/00-Dataset/CNNDetection", exp="cross-model"):
        self.loader = loader
        self.root = root

        self.real_imgs = []
        self.fake_imgs = []
        train_class = os.listdir(self.root)
        for i in train_class:
            if i == "horse" and exp == "cross-category":
                continue
            class_path = os.path.join(self.root, i)
            real_img = os.listdir(os.path.join(class_path, "0_real"))
            self.real_imgs += [os.path.join(class_path, "0_real", j) for j in real_img]
            fake_img = os.listdir(os.path.join(class_path, "1_fake"))
            self.fake_imgs += [os.path.join(class_path, "1_fake", j) for j in fake_img]

        self.num_real = len(self.real_imgs)
        self.num_fake = len(self.fake_imgs)
        print("real images num:", self.num_real)
        print("fake images num:", self.num_fake)

        self.img_list = []
        self.label_list = []

        for idx in range(self.num_real):
            self.img_list.append(self.real_imgs[idx])
            self.label_list.append(torch.tensor([0]))
        for idx in range(self.num_fake):
            self.img_list.append(self.fake_imgs[idx])
            self.label_list.append(torch.tensor([1]))

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = self.loader(img_path)
        label = self.label_list[index]
        return img, label

    def __len__(self):
        return self.num_real + self.num_fake


# for others, such as: deepfake, biggan...
# File Structure:
# deepfake:
#   0_real:
#       xxx.png
#         ...
#       xxx.png
#   1_fake:
#       xxx.png
#         ...
#       xxx.png
class FullDataset2(Dataset):
    def __init__(self, root="/data1/whm/00-Dataset/CNNDetection/Test/biggan"):
        self.loader = loader
        self.root = root

        self.real_imgs = []
        self.fake_imgs = []
        real_img = os.listdir(os.path.join(self.root, "0_real"))
        self.real_imgs += [os.path.join(self.root, "0_real", j) for j in real_img]
        fake_img = os.listdir(os.path.join(self.root, "1_fake"))
        self.fake_imgs += [os.path.join(self.root, "1_fake", j) for j in fake_img]

        self.num_real = len(self.real_imgs)
        self.num_fake = len(self.fake_imgs)
        print("real images num:", self.num_real)
        print("fake images num:", self.num_fake)

        self.img_list = []
        self.label_list = []

        for idx in range(self.num_real):
            self.img_list.append(self.real_imgs[idx])
            self.label_list.append(torch.tensor([0]))
        for idx in range(self.num_fake):
            self.img_list.append(self.fake_imgs[idx])
            self.label_list.append(torch.tensor([1]))

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = self.loader(img_path)
        label = self.label_list[index]
        return img, label

    def __len__(self):
        return self.num_real + self.num_fake


class RandomDataset(Dataset):
    def __init__(self, LENGTH, class_num=training_classes, root="/data1/whm/00-Dataset/CNNDetection/Train"):
        self.len = LENGTH
        self.loader = loader
        self.root = root

        self.real_imgs = []
        self.fake_imgs = []

        # train_class = ["horse", "car", "cat", "airplane"]
        if class_num == 20:
            train_class = os.listdir(self.root)
        elif class_num == 4:
            train_class = ["horse", "car", "cat", "airplane"]
        elif class_num == 2:
            train_class = ["horse", "car"]
        elif class_num == 1:
            train_class = ["horse"]
        for i in train_class:
            class_path = os.path.join(self.root, i)
            real_img = os.listdir(os.path.join(class_path, "0_real"))
            self.real_imgs.append([os.path.join(class_path, "0_real", j) for j in real_img])
            fake_img = os.listdir(os.path.join(class_path, "1_fake"))
            self.fake_imgs.append([os.path.join(class_path, "1_fake", j) for j in fake_img])  # fake_imgs is a 2-D List.

        self.N_class = len(train_class)
        # print("N_class =", self.N_class)

    def __getitem__(self, index):
        # Randomly sample images from real and fake.
        if np.random.randint(0, 2):  # fake images
            class_index = np.random.randint(0, self.N_class)
            img_index = np.random.randint(0, len(self.fake_imgs[class_index]))
            # tag = f"Fake: class_index={class_index},img_index={img_index}"
            # print(tag)
            img_path = self.fake_imgs[class_index][img_index]  # 尽可能多地取不同类别下的images
            img = self.loader(img_path)
            label = torch.tensor([1])
        else:  # real images
            class_index = np.random.randint(0, self.N_class)
            img_index = np.random.randint(0, len(self.real_imgs[class_index]))
            # tag = f"Real: class_index={class_index},img_index={img_index}"
            # print(tag)
            img_path = self.real_imgs[class_index][img_index]
            img = self.loader(img_path)
            label = torch.tensor([0])
        return img, label

    def __len__(self):
        return self.len


def Val(model, ValDataset):
    start = time.time()
    val_loader = DataLoader(dataset=ValDataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    model.eval()
    acc_total = 0
    batch_num = 0
    _labels = []
    rets = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            inputs = images
            inputs = inputs.to(device)
            prd = model(inputs)

            acc = torch.sum(torch.eq(torch.ge(prd, torch.full_like(labels, 0.)), labels))
            acc_total += acc.cpu().numpy() / len(images)
            # print("length=", len(images))
            _labels += list(labels.cpu().numpy())
            rets += list(prd.detach().cpu().numpy())

            batch_num += 1

        AP = average_precision_score(_labels, rets)
    model.train()
    time_consumed = time.time() - start
    print("Validation consumed {:.0f}m {:.0f}s".format(time_consumed // 60, time_consumed % 60))

    return acc_total / batch_num, AP
