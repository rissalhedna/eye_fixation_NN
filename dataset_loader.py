import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imageio.v2 as imageio
import matplotlib.pyplot as plt

def read_text_file(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip() 
            lines.append(line)
    return lines

class FixationDataset(Dataset):
    def __init__(self, root_dir, image_file, fixation_file, image_transform=None, fixation_transform=None):
        self.root_dir = root_dir
        self.image_files = read_text_file(image_file)
        self.fixation_files = read_text_file(fixation_file)
        self.image_transform = image_transform
        self.fixation_transform = fixation_transform
        assert len(self.image_files) == len(self.fixation_files), "lengths of image files and fixation files do not match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = imageio.imread(img_name)

        fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
        fix = imageio.imread(fix_name)

        sample = {"image": image, "fixation": fix}

        if self.image_transform:
            sample["image"] = self.image_transform(sample["image"])
        if self.fixation_transform:
            sample["fixation"] = self.fixation_transform(sample["fixation"])

        return sample


class TestFixationDataset(Dataset):
    def __init__(self, root_dir, image_file, image_transform=None):
        self.root_dir = root_dir
        self.image_files = read_text_file(image_file)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = imageio.imread(img_name)

        sample = {"image": image}

        if self.image_transform:
            sample["image"] = self.image_transform(sample["image"])

        return sample


def get_datasets(root_dir=os.getcwd()):
    image_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    fixation_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_image_file = os.path.join(root_dir, "txt/train_images.txt")
    train_fixation_file = os.path.join(root_dir, "txt/train_fixations.txt")

    val_image_file = os.path.join(root_dir, "txt/val_images.txt")
    val_fixation_file = os.path.join(root_dir, "txt/val_fixations.txt")

    test_image_file = os.path.join(root_dir, "txt/test_images.txt")

    train_dataset = FixationDataset(root_dir=root_dir,
                                    image_file=train_image_file,
                                    fixation_file=train_fixation_file,
                                    image_transform=image_transform,
                                    fixation_transform=fixation_transform)

    val_dataset = FixationDataset(root_dir=root_dir,
                                  image_file=val_image_file,
                                  fixation_file=val_fixation_file,
                                  image_transform=image_transform,
                                  fixation_transform=fixation_transform)

    test_dataset = TestFixationDataset(root_dir=root_dir,
                                       image_file=test_image_file,
                                       image_transform=image_transform)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":

    def visualize_samples(dataset, num_samples=5):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        
        for i, sample in enumerate(dataloader):
            if i >= num_samples:
                break
            
            image = sample["image"].squeeze().permute(1, 2, 0)  
            fixation = sample["fixation"].squeeze()  

            plt.subplot(2, num_samples, i + 1)
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"Image {i + 1}")

            plt.subplot(2, num_samples, num_samples + i + 1)
            plt.imshow(fixation, cmap="gray") 
            plt.axis("off")
            plt.title(f"Fixation {i + 1}")

        plt.show()

    train_dataset, _, _ = get_datasets()
    visualize_samples(train_dataset, num_samples=5)
