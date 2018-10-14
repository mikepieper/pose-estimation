import scipy.io
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    """
    The Leeds sport pose dataset.
    (http://sam.johnson.io/research/lsp.html.)
    
    Parameters
    ----------
    split : {'train', 'test'}
    """
    def __init__(self, split):
        self.indices = np.genfromtxt(f'data/lspset_dataset/{split}_indices.csv', delimiter=",").astype(int)
        data = torch.from_numpy(scipy.io.loadmat('data/lspset_dataset/joints.mat')['joints'])
        data = data.permute(2, 0, 1)
        self.joints = data[:, :, :2].reshape(-1, 28)
        self.occlusion_mask = data[:, :, 2].unsqueeze(-1).expand((-1, 14, 2)).reshape(-1, 28)
        self.transform = transform = transforms.Compose(
            [transforms.Resize((220, 220)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        old_image = Image.open(f'data/lspset_dataset/images/im{index+1:05d}.jpg')
        old_width, old_height = old_image.size
        image = self.transform(old_image)
        
        joints = self.joints[index, :]
        joints[0:28:2] *= 220 / old_width
        joints[1:28:2] *= 220 / old_height 
        sample = {'image': image, 'joints': self.joints[index, :], 
                  'occlusion_mask': self.occlusion_mask[index, :]}
        return sample

# trainloader = torch.utils.data.DataLoader(Dataset('train'), batch_size=32,
#                                           shuffle=True, num_workers=2)
# testloader = torch.utils.data.DataLoader(Dataset('test'), batch_size=32,
#                                           shuffle=True, num_workers=2)