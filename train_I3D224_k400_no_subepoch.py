# by Ruikang Luo
import os
import copy
from pathlib2 import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from i3d_src.utils import *
from i3d_src.opts import parser
from i3d_src.i3d_s224 import I3D, Unit3Dpy
from i3d_src.DataLoader_i3d_k400_DS1_ge128 import RGB_jpg_train_Dataset, RGB_jpg_val_Dataset
Num_Frames = 32
Batch_Size = 16
Num_Workers = 12
s0 = "i3dm_s224_d1_f{}_k400_".format(Num_Frames)

data_dir = Path("data/k400_imgs")    
data_json_dir = "data/k400_imgs/data_json_ge128"
num_sub_dataset = 1

weight_path = "weights/i3d_s224_k400_d1_f32_val_acc_0.6100.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log = Log(dir='log')
log_loss = Log_Loss(dir='log')

file_name =  os.path.basename(__file__)
log("\nstart running {}".format(file_name))
log_loss('epoch,sub_epoch,lr,train_loss,train_acc,val_loss,val_acc')
log("device = {}".format(device))
log("cpu_count = {}".format(os.cpu_count()))
log("batch_size = {}".format(Batch_Size))    
log("out_frame_num = {}".format(Num_Frames))    
log("data_dir = {}".format(data_dir))   
log("data_json_dir = {}".format(data_json_dir)) 
log("num_sub_dataset = {}".format(num_sub_dataset)) 

save_path = "./model_i3d/"
if not os.path.exists(save_path):
    os.mkdir(save_path)   
 
def train_model(opt,save_path, model, criterion, optimizer, lr_i, lr_i_max, lr_list, num_epochs, \
            train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size):
       
    since = time.time()

    data_loaders = {}
    dataset_sizes = {}
    data_loaders['val'] = val_data_loader
    dataset_sizes['val'] = val_dataset_size

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_acc = 0.0
    optimizer.param_groups[0]['lr'] = lr_list.pop(0)
    lr_pre = optimizer.param_groups[0]['lr']
    len_train_loader = len(train_data_loaders)
    
    for epoch in range(num_epochs):
        for i in range(len_train_loader):       
            data_loaders['train'] = train_data_loaders[i]
            dataset_sizes['train'] = train_dataset_sizes[i]         
                           
            lr_cur = optimizer.param_groups[0]['lr']

            lr_i += 1
            
            if lr_i > lr_i_max :
                if lr_list: 
                    optimizer.param_groups[0]['lr'] = lr_list.pop(0)
                    lr_cur = optimizer.param_groups[0]['lr']
                    lr_i = 1
                else:
                    log("lr_list is empty")
                    return
                
            if lr_cur != lr_pre :
                model.load_state_dict(best_model_wts)  
                optimizer.load_state_dict(best_optimizer_state) 
                optimizer.param_groups[0]['lr'] = lr_cur                         
                lr_pre = lr_cur   

                        
            #print("\n")
            log('-' * 40)
            log('Epoch={}, i={}, lr={}, lr_i={}, lr_i_max={}'.format(epoch, i, lr_cur, lr_i, lr_i_max))
            

            # Each epoch has a training and validation phase        
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_correct = 0

                # Iterate over data.
                progress = tqdm(data_loaders[phase])
                for idx, (data,labels) in enumerate(progress):                                    
                    data = data.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        out_softmax, out_logits = model(data)
                        #out_softmax = torch.nn.functional.softmax(out_logits, 1)
                        probs, preds = torch.max(out_softmax.data.cpu(), 1)
                        loss = criterion(out_softmax.cpu(), labels.cpu())                    

                        if phase == "train":
                            loss.backward()
                            optimizer.step()                       

                        # statistics
                        running_loss += loss.item() * data.shape[0]
                        running_correct += torch.sum(preds == labels.data.cpu())
                       
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_correct.double() / dataset_sizes[phase]                    

                if phase == 'train':
                    #scheduler.step()
                    train_acc = epoch_acc
                    train_loss = epoch_loss

                log('Epoch={}, i={}, phase={}, Loss: {:.4f} Acc: {:.4f}'.format(
                    epoch, i, phase, epoch_loss, epoch_acc))

                if phase == 'val':
                    d_t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    s1 = s0 + d_t + "_"
                                                    
                    temp_model_path = save_path+s1+"{}_lr{:e}_e{}_acc{:.4f}_{:.4f}.pth".\
                        format(opt, lr_cur, epoch, train_acc, epoch_acc)
                    torch.save(model.state_dict(), temp_model_path) 
                    log("model saved path: \n"+temp_model_path)                    
                                       
                    log_loss('{}, {}, {:e}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.\
                        format(epoch,i,lr_cur,train_loss,train_acc,epoch_loss,epoch_acc))                                      
                    

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_train_acc = train_acc
                    best_model_wts = copy.deepcopy(model.state_dict()) 
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    lr_i = 0              
                    log('*************** best_model_wt is from e_{} i_{}***************'.format(epoch, i))

            print()

    time_elapsed = time.time() - since
    log('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
  
    log('Best train Acc: {:4f}, Best val Acc: {:4f}'.format(best_train_acc, best_acc))

    return 


def load_and_freeze_model(model, num_classes, num_freeze=0):
    if model.num_classes != num_classes :
        model.num_classes = num_classes
        print("new num_classes = ", num_classes)
        model.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024,
                                    out_channels=num_classes,
                                    kernel_size=(1, 1, 1),
                                    activation=None,
                                    use_bias=True,
                                    use_bn=False)
        
    #freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    counter = 0
    print("num_freeze = ", num_freeze)
    for child in model.children():
        counter += 1
        #print("layer number = ", counter)
        if counter <= num_freeze:
            log("L{}: Layer {} frozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = False
        else:
            log("L{}: Layer {} unfrozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = True   
    return model   

def main(train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size):

    Num_Epochs = 100
    Num_Classes=400

    DropoutProb = 0
    
    lr_list = [1.0e-3, 1.0e-4, 1.0e-5]
    lr = lr_list[0]
    lr_i = 0
    lr_i_max = 3

    opt_method = "SGD"
    weight_decay = 0
    momentum = 0.9
    
    model = I3D(num_classes=400, modality='rgb', dropout_prob=DropoutProb)
    print("dropout_prob = ", DropoutProb)
    model = load_and_freeze_model(model, Num_Classes, 0)
    model.to(device)      
    
    model.load_state_dict(torch.load(weight_path))
    log("i3dm_k400_model \n{} \nloaded successfully!".format(weight_path))
    
    print("model.num_classes = ", model.num_classes)    

    optim_paras = filter(lambda p: p.requires_grad, model.parameters())
    
    if opt_method == 'Adam' :
        optimizer = optim.Adam(optim_paras, lr=lr, weight_decay=weight_decay)
        log("adam: lr={}, weight_decay={:e}".format(lr, weight_decay)) 
        opt='adam_wd{:e}'.format(weight_decay)
    elif opt_method == 'SGD' :
        optimizer = optim.SGD(optim_paras, lr=lr, momentum=momentum, dampening=0, weight_decay=weight_decay) 
        log("sgd: lr={:e}, momentum={:.1f}, weight_decay={:e}".format(lr, momentum, weight_decay))
        opt='sgd_mm{:.1f}'.format(momentum)
       
    criterion = nn.CrossEntropyLoss()    
      
    optimizer.param_groups[0]['lr'] = lr          
 
    train_model(opt,save_path, model, criterion, optimizer, lr_i, lr_i_max, lr_list, Num_Epochs, \
            train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size)   


if __name__ == "__main__":
    args = parser.parse_args()
    classes_path= data_dir / "classes.txt"
    class_names = [i.strip() for i in open(classes_path)]
    class_dicts = {k: v for v, k in enumerate(class_names)}

    x = 'train'
    train_datasets = []
    train_data_loaders = []
    train_dataset_sizes = []
    if not os.path.exists(data_json_dir):
        os.makedirs(data_json_dir) 
    data_pairs_file_root = data_json_dir + "/train_data_pairs"        
    for i in range(num_sub_dataset):
        data_pairs_file = data_pairs_file_root+"_"+str(num_sub_dataset)+"_"+str(i+1)+".json"
        train_datasets.append(RGB_jpg_train_Dataset(data_pairs_file, data_dir/x, class_dicts,
                                            out_frame_num=Num_Frames, x=x))  

        train_data_loaders.append(torch.utils.data.DataLoader(\
            train_datasets[i], batch_size=Batch_Size,shuffle=True, num_workers=Num_Workers))

        train_dataset_sizes.append(len(train_datasets[i]))
        
        log("train_datasets_sizes[{}] = {}".format(i, train_dataset_sizes[i]))    
            
    x = 'val'
    data_pairs_file = data_json_dir + "/val_data_pairs.json"
    val_dataset = RGB_jpg_val_Dataset(data_pairs_file, data_dir/x, class_dicts,
                                        out_frame_num=Num_Frames, x=x)
    val_data_loader = torch.utils.data.DataLoader(\
        val_dataset, batch_size=Batch_Size,shuffle=True, num_workers=Num_Workers)          
        
    val_dataset_size = len(val_dataset)
    log("val_dataset_size = {}".format(val_dataset_size))    
    
    main(train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size)