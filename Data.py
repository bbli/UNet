import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from skimage import io
from skimage import img_as_float
from skimage.transform import downscale_local_mean
from sklearn.preprocessing import StandardScaler
from functools import partial

import debug as D


def toTorch(image):
    new_image = torch.from_numpy(image).float()
    new_image = new_image.view(-1,1,667,667)
    print(type(new_image))
    return new_image

class Normalize():
    def __init__(self):
        self.scalar = StandardScaler()
    def __call__(self, image):
        self.scalar.fit(image)
        return self.scalar.transform(image)
    

class ParhyaleDataset(Dataset):
    def __init__(self,path,transform=None):
        self.image = img_as_float(io.imread(path)) 
        self.img = img_as_float(io.imread(path)) 

        
        # self.image=self.image/255
        if transform:
            self.image = transform(self.image)
    def __len__(self):
        return 1
    def __getitem__(self,index):
        return self.pic


if __name__=='__main__':
    path = '/data/bbli/gryllus_disk_images/cmp_1_1_T0000.tif'

    downscale = partial(downscale_local_mean, factors=(3,3))
    transforms = Compose([downscale, Normalize(), toTorch])
    # transforms = Compose ([ToTensor(),Normalize(0,1)])

    dataset = ParhyaleDataset(path,transforms)
