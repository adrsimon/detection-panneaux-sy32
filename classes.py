import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import torch.nn.functional as F

class VideosDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, label_encoder, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_name}: {e}")
            return None, None

        annotation_name = img_name.replace('.jpg', '.csv')
        annotation_path = os.path.join(self.annotation_dir, annotation_name)

        try:
            annotation = pd.read_csv(annotation_path, header=None)
        except Exception as e:
            print(f"Error reading {annotation_name}: {e}")
            annotation = pd.DataFrame()

        if annotation.empty:
            label = self.label_encoder.transform(['empty'])[0]
            if self.transform:
                image = self.transform(image)
            return image, label

        rois = []
        for _, line in annotation.iterrows():
            try:
                label_str = line[4]
                if label_str != 'empty':
                    x1 = int(line[0])
                    y1 = int(line[1])
                    x2 = int(line[2])
                    y2 = int(line[3])
                    roi = image.crop((x1, y1, x2, y2))
                else:
                    roi = image
            except Exception as e:
                print(f"Error processing annotation {annotation_name}: {e}")
                return None, None

            if self.transform:
                roi = self.transform(roi)

            label = self.label_encoder.transform([label_str])[0]

            rois.append((roi, label))

        return rois


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout3 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = x.view(-1, 128 * 16 * 16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x