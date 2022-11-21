#套件
import torch
import torch.nn as nn
from time import sleep
import torch.nn.utils.prune as prune


#模型剪枝
class pruning():
    def __init__(self, model, path) -> None:
        super().__init__()
        self.path = path
        self.model = model
    # 定義模型
    def process(self):
        state_dict = torch.load(self.path)
        self.model.load_state_dict(state_dict)

        # 取得每一層的名稱和參數
        name_list = [n for n,p in self.model.named_parameters()]

        for name in name_list:

            #只剪weight檔
            if 'weight' in name:
                m = getattr(self.model, name.split('.')[0])

                #只剪維度大於一的weight
                try:
                    prune.ln_structured(m,name="weight", amount=0.1,n=1,dim=0)
                    a = m.weight.data
                    b = m.weight_mask.data

                    m.weight.data =m.weight.data.mul(m.weight_mask.data)       
                    c = m.weight_mask.data

                    for name, module in self.model.named_modules():
                        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                            print(module)
                            prune.remove(module, 'weight')

                #維度唯一的weight沒有剪所以沒有mask檔，不用remove
                except:
                    try:
                        
                        prune.remove(module, 'weigh_mask')
                    except:
                        pass
        # save model
        torch.save(self.model.state_dict(), 'efficientnet-b0.pth')
