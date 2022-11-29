from efficient import *
import cifar
from thop import profile
from collections import OrderedDict
import time
import product_dataset
import pruning

model = EfficientNetB0()

pr = pruning.pruning(model, 'efficientnet-b0.pth')
pr.process()

def load_path(model, path):
    new_state_dict = OrderedDict()
    for key,value in torch.load(path).items():
        if 'weight_orig' in key:
            title = key.split('.')
            name = title[0]+'.weight'
            new_state_dict[name] = value
            value1 = value
            print(key)
        elif 'weight_mask' in key:
            title = key.split('.')
            name = title[0]+'.weight'
            new_state_dict[name] = value1
            print(key)
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)

DEVICE = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
model = EfficientNetB0()
# model.load_state_dict(torch.load('efficientnet-b0_74%_new.pth'))
model.eval()
model.to(DEVICE)
load_path(model, 'efficientnet-b0.pth')
test_set = product_dataset.image_datasets
testloader = torch.utils.data.DataLoader(test_set[p], batch_size=16, shuffle=True, drop_last=True)
# start test
criterion = nn.CrossEntropyLoss()
correct = 0
total = 0
time_start = time.time()

with torch.no_grad():
    for times in range(1):

        for data in testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of test images: %d %%' % (
            100 * correct / total))
time_end = time.time()
print('time cost', time_end - time_start, 's')

# caculate FLOPs
flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(DEVICE),))
print('FLOPs: %.2fM, Params: %.2fM' % (flops / 1e6, params / 1e6))