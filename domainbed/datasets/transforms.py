from torchvision import transforms as T

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def basic(size=224):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        _NORM,
    ])


def aug(size=224):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        _NORM,
    ])