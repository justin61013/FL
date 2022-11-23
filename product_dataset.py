# load data set 'C:\Users\USER\Downloads\data_set'
from torchvision import datasets,transforms
import os

imgsz = 240
folder = 'data_white'
data_transforms = {
    folder : transforms.Compose([
        transforms.Resize((imgsz, imgsz)),  # resize
        transforms.CenterCrop((imgsz, imgsz)),  # 中心裁剪
        transforms.RandomRotation(45),  # 随機旋轉，旋轉範圍為【-45,45】
        transforms.ToTensor(),  # 轉換為張量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
    ]),

    "val": transforms.Compose([
        transforms.Resize((imgsz, imgsz)),  # resize
        transforms.CenterCrop((imgsz, imgsz)),  # 中心裁剪
        transforms.ToTensor(),  # 張量轉換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}
image_datasets = {x: datasets.ImageFolder(root=os.path.join('C:/Users/USER/Desktop/FL_fin/FL_fin/FL/data_set',x), transform=data_transforms[x]) for x in [folder, 'val']}