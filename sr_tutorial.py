import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from time import time
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import cv2



#=================================================
# HYPERPARAMETERS HERE...
img_transforms = transforms.Compose([
    transforms.Resize((224,224)),    
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45)
    ])
target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ) 
])
input_transform = transforms.Compose([
    transforms.Resize((56, 56)),#((224,224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ) 
    ])
if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")
loss_func = torch.nn.SmoothL1Loss()
epochs = 150
batch_size=4
lr = 0.001
datapath = 'catdog/srgan_datatrain'
train_data = os.listdir(datapath)


#=================================================
# CLASS AND FUNCTION HERE...

def read_batch(datapath, imgname):
    imginput = []
    imgtarget = []
    for i in range(len(imgname)):
        img = Image.open(os.path.join(datapath, imgname[i]))
        img = img_transforms(img)
        imgtgt = target_transform(img)
        imgtarget.append(imgtgt)
        img = input_transform(img)
        imginput.append(img)
    imginput = torch.stack([x for x in imginput], dim=0).to(device)
    imgtarget = torch.stack([x for x in imgtarget], dim=0).to(device)
    # print('imginput.size(): ', imginput.size())
    # print('imgtarget.size(): ', imgtarget.size())
    return imginput, imgtarget

# here comes the training function!
def train(model, optimizer, loss_fn, train_data, datapath=None, batch_size = 5, epochs=20, device="cpu"):
    lowest_train_loss = 1e+6
    train_num = np.arange(len(train_data))
    train_data = np.array(train_data)
    print('=======================')
    print('Training data number: ', len(train_data))
    print('Start Training...')
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        start = time()
        model.train()
        np.random.shuffle(train_num)
        ctr = batch_size
        start = time()
        for b in range(0, len(train_num), batch_size):
            batch_data = train_data[train_num[b:b+ctr]]
            ctr += b
            optimizer.zero_grad()

            inputs, targets = read_batch(datapath, batch_data)
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_num)

        print('Epoch: {}, time: {:.2f}s, Lowest train loss: {:.2f}, Training Loss: {:.2f}'.format(epoch, 
                                                        time()-start, lowest_train_loss, training_loss))

        if training_loss < lowest_train_loss:
            lowest_train_loss = training_loss
            if epochs > 10:
                torch.save(model.state_dict(), r'D:\pytorch_tutorial\catdog\srgan_trained.pth')

class OurFirstSRNet(nn.Module):
    def __init__(self):
        super(OurFirstSRNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256,256,kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,192,kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192,128,kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,64,kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x

# test image transform
img = Image.open(os.path.join(datapath, train_data[0]))
img = img_transforms(img)
img = input_transform(img).unsqueeze(0).to(device)

# test forward propagation
model = OurFirstSRNet()
model.to(device)
print('=======================')
print('Example input-output...')
print('input: ', img.size())
output = model(img)
print('output:', output.size())
# define Backprop Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
model.load_state_dict(torch.load( r'catdog/srgan_trained.pth') )

# training
if sys.argv[1] ==  'training':
    print('...You pick mode training...')
    train(model, optimizer, loss_func, train_data, datapath=datapath, 
        batch_size = batch_size, epochs=epochs, device=device)

# testing
if sys.argv[1] ==  'detecting':
    print('...You pick mode detecting...')
    model.load_state_dict(torch.load( r'catdog/srgan_trained.pth') )
    model.eval()
    with torch.no_grad():
        output = model(img).cpu().squeeze(0).numpy()
        print('output:', output.shape)
        # cv2.imwrite('output.png', output[...,::-1].astype(np.uint8))






