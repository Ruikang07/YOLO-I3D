# by Ruikang Luo
import os
import time
import copy
from pathlib2 import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from i3d_src.utils import *
from yolo_i3d_s224 import I3D, Unit3Dpy, YOLO_3D_Bottom
from i3d_src.DataLoader_i3d_h51_DS2_ge32 import RGB_jpg_train_Dataset, RGB_jpg_val_Dataset

from i3d_src.opts import parser

def modify_and_freeze_i3d_top(model, num_classes, num_freeze=10):
    if model.num_classes != num_classes :
        model.num_classes = num_classes
        model.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024,
                                    out_channels=num_classes,
                                    kernel_size=(1, 1, 1),
                                    activation=None,
                                    use_bias=True,
                                    use_bn=False)  

    counter = 0
    for child in model.children():
        counter += 1
        #print("model layer number = ", counter)
        if counter > num_freeze:
            log("I3D: Layer{}( {} ) was unfrozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = True
        else:
            log("I3D: Layer{}( {} )  was frozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = False
    return model   

       
def train_model(model_i3d, model_yolo_3d_bottom, opt,save_path, \
            criterion, optimizer, lr_i, lr_i_max, lr_list, num_epochs, \
            train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size):
    
    since = time.time()

    data_loaders = {}
    dataset_sizes = {}
    data_loaders['val'] = val_data_loader
    dataset_sizes['val'] = val_dataset_size

    best_model_wts = copy.deepcopy(model_i3d.state_dict())
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
                model_i3d.load_state_dict(best_model_wts)  
                optimizer.load_state_dict(best_optimizer_state) 
                optimizer.param_groups[0]['lr'] = lr_cur                         
                lr_pre = lr_cur                

            log('Epoch={}, i={}, lr={}, lr_i={}, lr_i_max={}'.format(epoch, i, lr_cur, lr_i, lr_i_max))
            log('-' * 50)

            # Each epoch has a training and validation phase        
            for phase in ['train', 'val']:
                if phase == 'train':
                    model_i3d.train()  # Set model to training mode
                    model_yolo_3d_bottom.eval()
                else:
                    model_i3d.eval()  # Set model to evaluate mode
                    model_yolo_3d_bottom.eval()

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
                        out_softmax, out_logits = model_i3d(model_yolo_3d_bottom(data))
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

                log('Epoch={}, i={},  Loss: {:.4f} Acc: {:.4f}'.format(
                    epoch, i, epoch_loss, epoch_acc))

                if phase == 'val':
                    d_t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    s1 = "yolo_i3d_s224_h51_DS2_ge32_f16_"+d_t+"_"
                    temp_model_path = save_path+s1+"{}_lr{:e}_e{}_i{}_acc{:.4f}_{:.4f}.pth".\
                        format(opt, lr_cur, epoch, i, train_acc, epoch_acc)
                    torch.save(model_i3d.state_dict(), temp_model_path) 
                    log("model saved path: \n"+temp_model_path)
                                       
                                                           
                    log_loss('{}, {}, {:e}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.\
                        format(epoch,i,lr_cur,train_loss,train_acc,epoch_loss,epoch_acc))                    
                    

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_train_acc = train_acc
                    best_model_wts = copy.deepcopy(model_i3d.state_dict())   
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    lr_i = 0         
                    log('*************** best_model_wt is from e_{} i_{}***************'.format(epoch, i))

            print()

    time_elapsed = time.time() - since
    log('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
  
    log('Best train Acc: {:4f}, Best val Acc: {:4f}'.format(best_train_acc, best_acc))

    return 


def main(model_i3d, model_yolo_3d_bottom, \
        train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size):     
    
    Num_Epochs = 200
       
    lr_list = [8.0e-4, 4.0e-4, 2.0e-4, 1.0e-4, 4.0e-5, 2.0e-5, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8]
    
    lr = lr_list[0]
    lr_i = 0
    lr_i_max = 3

    opt_method = "Adam"
    weight_decay = 2.0e-6
    momentum = 0.9    
    
    optim_paras = filter(lambda p: p.requires_grad, model_i3d.parameters())  
    if opt_method == 'Adam' :
        optimizer = optim.Adam(optim_paras, lr=lr, weight_decay=weight_decay)
        log("adam: lr={:e}, weight_decay={:e}".format(lr, weight_decay)) 
        opt='adam_wd{:e}'.format(weight_decay)
    elif opt_method == 'SGD' :
        optimizer = optim.SGD(optim_paras, lr=lr, momentum=momentum) 
        log("sgd: lr={:e}, momentum={:.2f}".format(lr, momentum))
        opt='sgd_mm{:.2f}'.format(momentum)    
          
    save_path = "./model_i3d/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)     
      
    optimizer.param_groups[0]['lr'] = lr   
    criterion = nn.CrossEntropyLoss()        
 
    train_model(model_i3d, model_yolo_3d_bottom, opt,save_path, \
            criterion, optimizer, lr_i, lr_i_max, lr_list, Num_Epochs, \
            train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size)  


if __name__ == "__main__":    
        
    Batch_Size = 17
    out_frame_num = 16
    Num_Workers = 12
    
    log = Log(dir='log')
    log_loss = Log_Loss(dir='log')

    args = parser.parse_args()
    log("start running {}".format(parser.prog))
    
    DropoutProb = 0.4
    weight_yolo = "weights/yolov5l.pt"   
    weight_i3d = "weights/yolo_i3d_top_s224_k400_d2_f16_val_acc_0.6242.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_yolo_3d_bottom = YOLO_3D_Bottom(device, weight_yolo) 
    model_i3d = I3D(num_classes=400, modality='rgb', dropout_prob=DropoutProb)
    model_i3d.to(device) 
    model_i3d.load_state_dict(torch.load(weight_i3d))

    data_dir = Path('data/hmdb51')
    data_json_dir = "data/hmdb51/data_json_ge32"
    num_sub_dataset = 1 

    log("device = {}".format(device))
    log_loss("epoch,sub_epoch,lr,train_loss,train_acc,val_loss,val_acc")

    log("YOLOv5 model: \n{} loaded successfully!".format(weight_yolo))
    log("I3D model: \n{} loaded successfully!".format(weight_i3d))  
    log("DropoutProb = {}".format(DropoutProb))

    log("cpu_count = {}".format(os.cpu_count()))
    log("Batch_Size = {}".format(Batch_Size))    
    log("out_frame_num = {}".format(out_frame_num)) 
    log("data_dir = {}".format(data_dir))   
    log("data_json_dir = {}".format(data_json_dir)) 
    log("number of sub_dataset = {}".format(num_sub_dataset))     
    
    ###############################################################################
    Num_Classes=51
    model_i3d = modify_and_freeze_i3d_top(model_i3d, Num_Classes, 17)
    model_i3d.to(device) 
    log("model.num_classes = ", model_i3d.num_classes)     

    classes_path= data_dir / "classes.txt"
    class_names = [i.strip() for i in open(classes_path)]
    class_dicts = {k: v for v, k in enumerate(class_names)}
        
    #Prepare DataLoader  
    log("start processing train dataset")
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
                                            out_frame_num=out_frame_num, x=x))  
        train_data_loaders.append(torch.utils.data.DataLoader(\
            train_datasets[i], batch_size=Batch_Size,shuffle=True, num_workers=Num_Workers))
        train_dataset_sizes.append(len(train_datasets[i]))
        log("train_datasets_sizes[{}] = {}".format(i, train_dataset_sizes[i]))    
            

    log("start processing val dataset")
    x = 'val'
    data_pairs_file = data_json_dir + "/val_data_pairs.json"
    val_dataset = RGB_jpg_val_Dataset(data_pairs_file, data_dir/x, class_dicts,
                                        out_frame_num=out_frame_num, x=x)
    val_data_loader = torch.utils.data.DataLoader(\
        val_dataset, batch_size=Batch_Size,shuffle=True, num_workers=Num_Workers)                
    val_dataset_size = len(val_dataset)
    log("val_dataset_size = {}".format(val_dataset_size))      
    
    main(model_i3d, model_yolo_3d_bottom, \
        train_data_loaders, val_data_loader, train_dataset_sizes, val_dataset_size)         
