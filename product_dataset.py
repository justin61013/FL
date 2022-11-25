# load data set 'C:\Users\USER\Downloads\data_set'
from torchvision import datasets,transforms
import os

imgsz = 224
# data_white, data_black, data_glass
folder = 'data_white'
data_transforms = {
    folder : transforms.Compose([
        transforms.Resize((imgsz, imgsz)),  # resize
        transforms.RandomRotation(45),  # random rotation
        transforms.ToTensor(),  # 轉換為張量
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 正規化
        
    ]),

    "val": transforms.Compose([
        transforms.Resize((imgsz, imgsz)),  # resize
        transforms.ToTensor(),  # 張量轉換
    ])
}
image_datasets = {x: datasets.ImageFolder(root=os.path.join('C:/Users/USER/Desktop/FL_fin/FL_fin/FL/data_set',x), transform=data_transforms[x]) for x in [folder, 'val']}