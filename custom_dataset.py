from torchvision.datasets.folder import *
import numpy as np
import torch

class CustomDataset(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, target_num=None):
        super(CustomDataset, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
        self.target_num = target_num

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.target_num is None:
            while True:
                path2, target2 = self.samples[np.random.choice(len(self.samples), 1)[0]]
                if target == target2:
                    pass
                else:
                    break
            sample2 = self.loader(path2)
            if self.transform is not None:
                sample2 = self.transform(sample2)
            if self.target_transform is not None:
                target2 = self.target_transform(target2)

            sample = torch.cat((sample, sample2), 0)
            target = torch.LongTensor([target, target2])

        else:
            samples_array = np.array(self.samples)
            target_list = samples_array[samples_array[:, 1].astype(np.int) == target]
            idx = np.random.choice(target_list.shape[0], self.target_num - 1)
            path2, target2 = target_list[idx, 0], target_list[idx, 1]
            target = torch.LongTensor([target])
            for p, t in zip(path2, target2):
                sample2 = self.loader(p)
                if self.transform is not None:
                    sample2 = self.transform(sample2)
                if self.target_transform is not None:
                    target2 = self.target_transform(target2)

                sample = torch.cat((sample, sample2), 0)
                target = torch.cat((target, torch.LongTensor(target2.astype(np.long))), 0)

        return sample, target