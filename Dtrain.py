import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
# from data import x_train, y_train
from read_data import x_train,x_val,y_train,y_val
from transforms import Rotate, ToCDHW
# from model import *
from model.model import *
from datas_define import GetDataset
from tqdm import tqdm

save_model_path = './model_save/resnet101/'
save_model_name = '3DPET_CNN'
num_epochs = 50
batch_size = 8
LR = 1e-4
weight_decay = 1e-5
epoch_losses = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet101(1) 
# model = generate_model(121)
# model.fc = nn.Linear(model.fc.in_features, 1)
model.to(device)  

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=1)


# criterion = nn.BCEWithLogitsLoss() 
# criterion = nn.CrossEntropyLoss() 
# criterion = nn.FocalLoss()
# criterion = nn.BCELoss()

train_dataset = GetDataset(x_train,y_train, transform=Compose([Rotate(), ToCDHW()]))
valid_dataset = GetDataset(x_val, y_val, transform=Compose([ToCDHW()]))

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_dataset, batch_size=1)


def calculate_accuracy(outputs, targets):
    preds = outputs.round()
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0.0

    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float).unsqueeze(1)  # 确保targets的形状正确
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += calculate_accuracy(outputs, targets) * targets.size(0)

    # epoch_loss = running_loss / len(train_loader.dataset)
    # epoch_losses.append(epoch_loss)
    # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / len(train_loader.dataset)
    epoch_losses.append(epoch_loss)
    epoch_accuracies.append(epoch_acc)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')


    lr_scheduler.step()
    epochs = epoch + 1
    if (epochs % 30 == 0) & (epochs >= 10):
        # if (epoch % 10 == 0) :
        save_path = save_model_path
        torch.save(model.state_dict(),
                   save_path + save_model_name + '_Global1' + '_epochs_' + str(epochs) + '.pkl')
        print('Global_Branch_model already save!')
    # time_elapsed = time.time() - since
    # print('Training one epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
with open('./train_loss_acc/1.txt', 'w') as f:
    f.write("Epoch\tTrain Loss\tTrain Accuracy\n")
    for i in range(num_epochs):
        f.write(f"{i+1}\t{epoch_losses[i]:.4f}\t{epoch_accuracies[i]:.4f}\n")

plt.figure()
plt.plot(epoch_losses, label='Loss')
plt.plot(epoch_accuracies, label='Train Accuracy')
plt.title('Loss and Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig('./train_loss_acc/1.png')
plt.show()

