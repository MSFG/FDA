import os.path
import os
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


# PARAMS
batch_size = 32
num_epochs = 100

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def loader(path):
    img_pil = Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor


# def to_one_hot(y, n_class):  # y: label list, [0, 19, 17, ...]; n_class: 20
#     label = np.eye(n_class)[y]
#     label = torch.from_numpy(label)
#     return label
# x = to_one_hot([0, 19, 17, ...], 20)


class DealDataset(Dataset):
    def __init__(self, root="/data1/Jianwei-Fei/00-Dataset/01-Images/CNNDetection/Train"):
        self.loader = loader
        self.root = root

        self.real_imgs = []
        self.fake_imgs = []
        self.real_img_class_labels = []
        self.fake_img_class_labels = []
        train_class = os.listdir(self.root)
        class_label = 0
        for i in train_class:
            class_path = os.path.join(self.root, i)

            real_img = os.listdir(os.path.join(class_path, "0_real"))
            self.real_imgs += [os.path.join(class_path, "0_real", j) for j in real_img]
            self.real_img_class_labels += [class_label for _ in range(len(real_img))]

            fake_img = os.listdir(os.path.join(class_path, "1_fake"))
            self.fake_imgs += [os.path.join(class_path, "1_fake", j) for j in fake_img]
            self.fake_img_class_labels += [class_label for _ in range(len(fake_img))]
            class_label += 1

        self.num_real = len(self.real_imgs)
        self.num_fake = len(self.fake_imgs)
        print("real images num:", self.num_real)
        print("fake images num:", self.num_fake)

        self.img_list = []
        self.label_list = []
        self.class_label_list = []

        for idx in range(self.num_real):
            self.img_list.append(self.real_imgs[idx])
            self.label_list.append(torch.tensor([0]))
            self.class_label_list.append(self.real_img_class_labels[idx])
        for idx in range(self.num_fake):
            self.img_list.append(self.fake_imgs[idx])
            self.label_list.append(torch.tensor([1]))
            self.class_label_list.append(self.fake_img_class_labels[idx])

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = self.loader(img_path)
        label = self.label_list[index]  # real, fake
        class_label = self.class_label_list[index]  # airplane, train, person, horse...
        return img, label, class_label

    def __len__(self):
        return self.num_real + self.num_fake


root_dir = "/data1/Jianwei-Fei/00-Dataset/01-Images/CNNDetection/Train"
dataset_train = DealDataset(root=root_dir)
# root_dir = "/data1/Jianwei-Fei/00-Dataset/01-Images/CNNDetection/Val"
# dataset_val = DealDataset(root=root_dir)
# root_dir = "/data1/Jianwei-Fei/00-Dataset/01-Images/CNNDetection/Test/progan"
# dataset_test = DealDataset(root=root_dir)
# my_dataset = dataset_train + dataset_val + dataset_test
dataset_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 32, 3, padding=1)

        self.trans1 = torch.nn.ConvTranspose2d(32, 128, 3, padding=1)
        self.trans2 = torch.nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.trans3 = torch.nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.trans4 = torch.nn.ConvTranspose2d(32, 3, 3, padding=1)
        self.mp = torch.nn.MaxPool2d(2, return_indices=True)
        self.up = torch.nn.MaxUnpool2d(2)
        self.relu = torch.nn.ReLU()

    def encoder(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        s1 = x.size()
        x, ind1 = self.mp(x)  # [?, 32, 128, 128]
        x = self.conv2(x)
        x = self.relu(x)
        s2 = x.size()
        x, ind2 = self.mp(x)  # [?, 64, 64, 64]
        x = self.conv3(x)
        x = self.relu(x)
        s3 = x.size()
        x, ind3 = self.mp(x)  # [?, 128, 32, 32]
        x = self.conv4(x)
        x = self.relu(x)  # latent code: [?, 32, 32, 32]

        return x, ind1, s1, ind2, s2, ind3, s3

    def decoder(self, x, ind1, s1, ind2, s2, ind3, s3):
        x = self.trans1(x)
        x = self.relu(x)  # [?, 128, 32, 32]
        x = self.up(x, ind3, output_size=s3)  # [?, 128, 64, 64]
        x = self.trans2(x)
        x = self.relu(x)
        x = self.up(x, ind2, output_size=s2)  # [?, 64, 128, 128]
        x = self.trans3(x)
        x = self.relu(x)
        x = self.up(x, ind1, output_size=s1)  # [?, 32, 256, 256]
        x = self.trans4(x)
        x = self.relu(x)  # [?, 3, 256, 256]
        return x

    def forward(self, x):
        x, ind1, s1, ind2, s2, ind3, s3 = self.encoder(x)
        output = self.decoder(x, ind1, s1, ind2, s2, ind3, s3)
        return output


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dis = nn.Sequential(
            # input size = 3*256*256
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 20)
        )

        self.grl = GRL()

    def forward(self, x):
        output = self.dis(self.grl.apply(x))
        return output


class GRL(Function):
    @staticmethod
    def forward(self, input):
        return input.view_as(input)

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.neg() * 0.0001  # alpha = 0.01, 0.001, 0.0001
        return grad_input


if __name__ == "__main__":
    ae = AutoEncoder().to(device)
    class_dis = Discriminator().to(device)
    # grl = GRL()
    # dis = Discriminator().to(device)
    # dis.dis[-1] = nn.Linear(128, 1)
    loss_rec = nn.MSELoss()
    loss_dis1 = nn.CrossEntropyLoss()  # 自带one-hot转换！
    # loss_dis2 = nn.BCELoss
    optimizer_ae = torch.optim.Adam(ae.parameters())
    optimizer_Dc = torch.optim.Adam(class_dis.parameters())

    # Begin training.
    print("Num of images in the dataset: {}".format(len(dataset_loader.dataset)))
    per_epoch_num = (len(dataset_loader.dataset)) // batch_size
    with tqdm(total=num_epochs * per_epoch_num) as pbar:  # 进度条
        for epoch in range(num_epochs):
            i = 0
            epoch_loss_r, epoch_loss_d1 = 0, 0
            epoch_num_real, epoch_num_fake = 0, 0
            for images, labels, class_labels in dataset_loader:
                i = i + 1
                batch_num_real, batch_num_fake = 0, 0
                # class_labels = to_one_hot(class_labels, 20)
                images = images.to(device)
                labels = labels.to(device)
                class_labels = class_labels.to(device)

                real_img_list = []
                fake_img_list = []
                fake_class_label_list = []
                for image, label, class_label in zip(images, labels, class_labels):
                    image = image.unsqueeze(0)  # image size: [1, 3, 256, 256]
                    class_label = class_label.unsqueeze(0)  # class label size: [1, 1]
                    if label == 0:
                        real_img_list.append(image)  # only use real images to compute reconstruction loss.
                        batch_num_real += 1
                    elif label == 1:
                        fake_img_list.append(image)
                        fake_class_label_list.append(class_label)
                        batch_num_fake += 1

                epoch_num_fake += batch_num_fake
                epoch_num_real += batch_num_real

                real_imgs = torch.cat(real_img_list, dim=0)
                fake_imgs = torch.cat(fake_img_list, dim=0)
                fake_class_labels = torch.cat(fake_class_label_list, dim=0)

                # AE forward.
                real_outputs = ae(real_imgs)
                fake_outputs = ae(fake_imgs)

                # GRL+D forward.
                fingers = fake_imgs - fake_outputs
                fingers = fingers.to(device)
                fingers_class_outputs = class_dis(fingers)

                # Compute reconstruction loss and class discriminator loss.
                loss_r = loss_rec(real_outputs, real_imgs)
                loss_d1 = loss_dis1(fingers_class_outputs, fake_class_labels.long())

                optimizer_ae.zero_grad()
                optimizer_Dc.zero_grad()
                loss_d1.backward()
                loss_r.backward()
                optimizer_Dc.step()
                optimizer_ae.step()

                epoch_loss_r += loss_r * batch_num_real
                epoch_loss_d1 += loss_d1 * batch_num_fake

                if i % 10 == 0:
                    log_loss = "epoch [{}/{}], L2 Loss(mean): {}, Class Discriminator Loss(mean): {}.".\
                        format(epoch + 1, num_epochs, loss_r.data, loss_d1.data)
                    tqdm.write(log_loss)
                pbar.update(1)

                if epoch == num_epochs - 1:
                    if not os.path.exists("imgs_ae"):
                        os.makedirs("imgs_ae")
                    if i % 100 == 0:
                        save_image(images[0], "imgs_ae/img{}.png".format(i))
                        save_image(real_outputs[0], "imgs_ae/img_ae{}.png".format(i))
            epoch_log = "epoch [{}/{}], epoch_loss_r(mean): {:.6f}, epoch_loss_d1(mean): {:.6f}".\
                        format(epoch + 1, num_epochs, epoch_loss_r / epoch_num_real, epoch_loss_d1 / epoch_num_fake)
            print(epoch_log)
            with open("epoch_loss.txt", "a", encoding="utf-8") as f:
                f.write(epoch_log)
                f.write("\n")

            if not os.path.exists("ae_trained"):
                os.makedirs("ae_trained")
            torch.save(ae.state_dict(), f"ae_trained/ae_epoch{epoch}.pt")
