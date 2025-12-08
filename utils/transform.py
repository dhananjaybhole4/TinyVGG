from torchvision import transforms



transform = transforms.Compose([
transforms.Resize(size = (128,128)),
transforms.ToTensor()
])

