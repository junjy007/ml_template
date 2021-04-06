import torchvision.transforms as transforms
from torchvision.datasets import MNIST

def prepare_data(data_dir):
    print(f"Prepare data in dir {data_dir}")
    MNIST(root=data_dir, train=True, download=True)
    MNIST(root=data_dir, train=False, download=True)

def get_trainval_data(data_dir):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
    return MNIST(root=data_dir, transform=transform, train=True)