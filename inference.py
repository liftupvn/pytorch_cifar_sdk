from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from bluetraining import MLOps
from PIL import Image
from models import *

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch
import os


transform_test = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class CustomImageDataset(Dataset):
    def __init__(self, image_folder='test_set/cifar_test_dataset'):
        self.label_2_int = {'plane': 0,
                        'automobile': 1,
                        'bird': 2,
                        'cat': 3,
                        'deer': 4,
                        'dog': 5,
                        'frog': 6,
                        'horse': 7,
                        'ship': 8,
                        'truck': 9
                        }
        self.transform = transform_test
        self.labels = []
        self.image_paths = []
        self.list_labels = os.listdir(image_folder)
        if '.DS_Store' in self.list_labels:
            self.list_labels.remove('.DS_Store')

        for label in self.list_labels:
            elements = os.listdir(os.path.join(image_folder, label))
            if '.DS_Store' in elements:
                elements.remove('.DS_Store')
            for item in elements:

                self.labels.append(label)
                self.image_paths.append(os.path.join(image_folder, label, item))
        # print(self.image_paths)
        # print(self.labels)
        assert len(self.labels) == len(self.image_paths)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # image = read_image(self.image_paths[idx])
        image = Image.open(self.image_paths[idx]).convert('RGB')

        image = self.transform(image)
        label = self.labels[idx]
        label = self.label_2_int[label]
        return image, label

testing_data = CustomImageDataset()
test_dataloader = DataLoader(testing_data, batch_size=4, shuffle=True, num_workers=2, drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'



#---------------------------------------------------------------------------------------------------------------
net = SimpleDLA()
# net = MobileNet()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])

net.eval()

correct = 0
total = 0
y_true = []
y_pred = []

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        # print('intput: ', inputs.shape)
        # targets = list(targets)
        # print(type(targets))        
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)

        y_true += targets.tolist()
        

        _, predicted = outputs.max(1)
        y_pred += predicted.tolist()
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
print(correct)
print(total)

print(multilabel_confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5,6,7,8,9]))

benchmark = {"accuracy": correct/total}

MLOps.init('testing', name='test set A')
MLOps.log.add_test_results(
    '6098ef68e5c7f148f6b361dc',
    'test_set_A',
    benchmark,
    name='testing SimpleDLA')