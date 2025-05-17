from medmnist import INFO
import medmnist
from torchvision import transforms
from torch.utils.data import DataLoader

def get_medmnist_dataloaders(name, batch_size=64, resize=224):
    info = INFO[name]
    DataClass = getattr(medmnist, info['python_class'])

    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = DataClass(split='train', download=True, transform=transform)
    test_dataset = DataClass(split='test', download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
