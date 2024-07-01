from torchvision import datasets, transforms

# Applying transforms to the data
mean =  [0.5732364, 0.5861441, 0.4746769]
std = [0.1434643,  0.16687445, 0.15344492]

normalize = transforms.Normalize(mean=mean, std=std)
image_transforms = {
    'name': 'image_transforms_normal_3',
   'train': transforms.Compose([transforms.Resize((96,96)), 
                                transforms.RandomAffine(degrees=(0,90), translate=(0.5,0.5), scale=(0.2, 2)),
                                transforms.RandomCrop(96, padding=4, padding_mode='reflect'),
                                transforms.ToTensor(), normalize]),

    'valid': transforms.Compose([transforms.Resize((96,96)),
                                 transforms.RandomAffine(degrees=(0,90), translate=(0.5,0.5), scale=(0.2, 2)), 
                                 transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                transforms.ToTensor(), normalize]),
    # 'train': transforms.Compose([
    #     transforms.Resize(100),
    #     transforms.RandomCrop(96, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
    #                          [0.1434643, 0.16687445, 0.15344492]),
    # ]),
    # 'valid': transforms.Compose([
    #     transforms.Resize(100),
    #     transforms.RandomCrop(96, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
    #                          [0.1434643, 0.16687445, 0.15344492]),
    # ]),
}