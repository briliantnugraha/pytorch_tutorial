help = '''
    Author: Brilian
    Target: Pytorch tutorial data augmentation Chapter 9
    How to use it:
    - Go to the main folder (pytorch_tutorial folder)
    - open cmd in that folder (or you could win+R -> cmd -> go to this folder)
    - Type python mixup_segmentation.py {training or mixup_augmentation} #without bracket {} -> for mixup segmentation
    - Type python mixup_segmentation.py training (label_smooth) -> for label smoothing
    - Type python mixup_segmentation.py plot_training -> for plotting training output (with or without label_smooth)
    - TYpe python mixup_segmentation.py detecting (adversarial) -> for using adversarial
'''


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

#=================================================
# CLASS AND FUNCTION HERE...

# here comes the training function!
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu", debug_mixup=True):
    val_loss_list = []
    val_acc_list  = []
    lowest_val_loss = 1e+6
    log_type = ''
    if len(sys.argv) > 2 and sys.argv[2] == 'label_smooth':
        log_type = sys.argv[2]
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        start = time()
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            if epochs == 1:
                print('check0: ', targets, targets[0])
                targets[0] = targets[0] + 1. if targets[0] < 4 else targets
                print('check1: ', targets, targets[0])
            
            # if debug_mixup:
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        start_val = time()
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)
        end_val = time()

        val_acc_list.append(num_correct*100 / num_examples)
        val_loss_list.append(valid_loss)

        print('Epoch: {}, Runtime(train-val): {:.2f}s-{:.2f}s, Lowest val loss: {:.2f}, Training Loss: {:.2f}, '
                'Validation Loss: {:.2f}, Validation acc = {}%'.format(epoch, start_val-start, 
                                                            end_val-start_val, lowest_val_loss,
                                                            training_loss,
                                                            valid_loss, val_acc_list[-1]))

        if valid_loss < lowest_val_loss:
            lowest_val_loss = valid_loss
            if epochs > 10:
                torch.save(transfer_model.state_dict(), r'D:\pytorch_tutorial\catdog\resnext50_32x4d_trained{}.pth'.format(log_type))
    if epochs > 10: np.save('log_train{}.npy'.format(log_type), [val_acc_list, val_loss_list])

# check image
def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, output, target):
        num_classes = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = (-log_preds.sum(dim=-1)).mean()
        nll = F.nll_loss(log_preds, target)
        final_loss = self.epsilon * loss / num_classes + (1-self.epsilon) * nll
        return final_loss

# fast gradient single attack
def fgsm(input_tensor, labels, epsilon=0.02, loss_function=None, model=None):
    input_tensor.requires_grad = True
    outputs = model(input_tensor)
    loss = loss_function(outputs, labels)
    loss.backward(retain_graph=True)
    fgsm_out = torch.sign(input_tensor.grad) * epsilon
    perturbed_image = torch.clamp(fgsm_out.squeeze(0).to(device) + input_tensor, 0, 1).to(device)
    print("Mean fgsm: ", fgsm_out.cpu().numpy().mean())
    print("Mean perturbed_image: ", perturbed_image.cpu().detach().numpy().mean())
    return perturbed_image

def predict_image(transfer_model, imgfool):
    transfer_model.eval()
    with torch.no_grad():
        out = transfer_model(imgfool)
    out = predict_softmax(out)
    out = out.cpu().numpy()[0]
    return out

def plot_inputs(lbl, lblsmooth, path):
    plt.figure(figsize=[15,10])
    plt.plot(lbl, label='no_smooth')
    plt.plot(lblsmooth, label='with_smooth')
    plt.legend()
    plt.savefig(path)

def load_model(trained=False, label_smooth=False):
    # load our base model
    transfer_model = models.resnext50_32x4d(pretrained=True if not trained else False)
    # change last fully connected layer
    transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500),
                                                nn.ReLU(),                                 
                                                nn.Dropout(), nn.Linear(500, len(os.listdir(train_data_path)))
                                            )#2)
    if not trained:
        transfer_model.load_state_dict(torch.load( r'D:\pytorch_tutorial\catdog\resnext50_32x4d_base.pth') )
        # torch.save(transfer_model.state_dict(), r'D:\pytorch_tutorial\catdog\resnext50_32x4d_base.pth')
    if trained:
        if label_smooth:
            transfer_model.load_state_dict(torch.load( r'D:\pytorch_tutorial\catdog\resnext50_32x4d_trained.pth') )
        else:
            transfer_model.load_state_dict(torch.load( r'D:\pytorch_tutorial\catdog\resnext50_32x4d_trainedlabel_smooth.pth') )
    return transfer_model

#=================================================
# HYPERPARAMETER HERE...
classname = ['cat', 'dog', 'fish', 'frog', 'scorpion']
epochs = 50
batch_size=8
lr = 0.001
train_data_path = "./catdog/small_train/"
val_data_path = "./catdog/small_val/"
loss_func = torch.nn.CrossEntropyLoss()
if len(sys.argv) > 2 and sys.argv[2] == 'label_smooth':
    loss_func = LabelSmoothingCrossEntropyLoss()
    print('...You pick mode training label_smooth...')
if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")
predict_softmax = nn.Softmax(dim=-1).to(device)
epsilon_fgsm = 1e-2

# make data augmentation
img_transforms = transforms.Compose([
    transforms.Resize((64, 64)),#((224,224)),    
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    
    transforms.ToPILImage() if sys.argv[1] ==  'mixup_augmentation' \
                    else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ) 
    ])

#=================================================
# MODE OPERATION HERE...
if sys.argv[1] ==  'mixup_augmentation':
    print('...You pick mode mixup_augmentation...')
    pathcat = os.path.join(train_data_path, 'cat', os.listdir(os.path.join(train_data_path, 'cat'))[0] )
    pathdog = os.path.join(train_data_path, 'dog', os.listdir(os.path.join(train_data_path, 'dog'))[0]  )
    imgcat = Image.open(pathcat)
    imgdog = Image.open(pathdog)
    imgcat.save('imgcat.png')
    imgdog.save('imgdog.png')
    imgcat = img_transforms(imgcat)
    imgdog = img_transforms(imgdog)
    # imgcat = imgcat.unsqueeze(0)
    # imgdog = imgdog.unsqueeze(0)
    # print('array size: ', imgdog.size(), imgcat.size())
    imgcat.save('imgcat_trans.png')
    imgdog.save('imgdog_trans.png')

    # you could customize the mixup ratio here
    mixups = [0.2*(i+1) for i in range(4)]
    for mixup in mixups:
    # mixup = 0.5
        inputs_mixed = (mixup * np.array(imgcat)) + ((1-mixup) * np.array(imgdog))
        inputs_mixed = Image.fromarray(inputs_mixed.astype(np.uint8))
        inputs_mixed.save('inputs_mixed{:.1f}.png'.format(mixup))
        print('Done mixup:', mixup)


elif sys.argv[1] ==  'training':
    print('...You pick mode training...')
    # load our base model
    # transfer_model = models.resnext50_32x4d(pretrained=True)
    # # torch.save(transfer_model.state_dict(), r'D:\pytorch_tutorial\catdog\resnext50_32x4d_base.pth')
    # transfer_model.load_state_dict(torch.load( r'D:\pytorch_tutorial\catdog\resnext50_32x4d_base.pth') )
    # # change last fully connected layer
    # transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500),
    #                                             nn.ReLU(),                                 
    #                                             nn.Dropout(), nn.Linear(500, len(os.listdir(train_data_path)))#2)
                                    # )
    transfer_model = load_model(trained=False)
    # retrained needs batchnorm to be retrained too!
    for name, param in transfer_model.named_parameters():
        if("bn" not in name):
            param.requires_grad = False
    train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms, is_valid_file=check_image)
    val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=img_transforms, is_valid_file=check_image)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader   = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
        
    # print length of training and validation
    print('train_data_loader len: ', len(train_data_loader.dataset))
    print('val_data_loader len: ', len(val_data_loader.dataset))

    # put model into GPU
    transfer_model.to(device)
    # define Backprop Optimizer
    optimizer = optim.Adam(transfer_model.parameters(), lr=lr)

    # start training
    train(transfer_model, optimizer, loss_func, train_data_loader,
        val_data_loader, epochs=epochs, device=device)


elif sys.argv[1] ==  'plot_training':
    print('...You pick mode plot_training...')
    # this is to save mixup_segmentation label smooth
    import matplotlib.pyplot as plt
    labelsmooth = np.load(r'D:\pytorch_tutorial\log_trainlabel_smooth.npy')
    labels = np.load(r'D:\pytorch_tutorial\log_train.npy')
    plot_inputs(labels[0], labelsmooth[0], r'D:\pytorch_tutorial\train_acc.png')
    plot_inputs(labels[1], labelsmooth[1], r'D:\pytorch_tutorial\train_loss.png')

elif sys.argv[1] ==  'detecting':
    print('...You pick mode detecting...')
    transfer_model = load_model(trained=True, label_smooth=False)
    
    # put model into GPU
    transfer_model.to(device)

    # define path image input
    foolpick = 0#np.random.randint(2)
    fool = classname[foolpick]
    fooldir = os.listdir(os.path.join(train_data_path, fool))
    foolnumber = 19#np.random.randint(len(fooldir))
    pathfool = os.path.join(train_data_path, fool, fooldir[foolnumber] )
    imgfool = Image.open(pathfool)
    imgfool = img_transforms(imgfool).unsqueeze(0).to(device)
    print('Try to predict input label: ', fool, foolnumber)
    
    # predict without adversarial
    out = predict_image(transfer_model, imgfool)
    print('(without adversarial) Predict acc : {}, Highest accuracy class: {}'.format(out, classname[np.argmax(out)]))
    
    # predict with adversarial
    if len(sys.argv) > 2 and sys.argv[2] ==  'adversarial':
        print('...You pick mode detecting adversarial...')
        labelfool = torch.tensor([foolpick]).to(device)
        model_to_break = transfer_model# load our model to break here
        imgfool = fgsm(imgfool,
                    labelfool,
                    epsilon_fgsm,
                    loss_func,
                    model_to_break)
        out = predict_image(transfer_model, imgfool)
        print('(with adversarial) Predict acc: {}, Highest accuracy class: {}'.format(out, classname[np.argmax(out)]))

else:
    print('...No mode for your input: {}...'.format(sys.argv[1]))
    print('...Here is the tips: ', help)


