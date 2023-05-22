# by Ruikang Luo
import os
import sys
import time
import copy
from pathlib2 import Path
from tqdm import tqdm

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

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
            print("I3D: Layer{}( {} ) was unfrozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = True
        else:
            print("I3D: Layer{}( {} )  was frozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = False
    return model   


def val_model(model_i3d, model_yolo_3d_bottom, criterion, val_data_loader, val_dataset_size):      
    data_loaders = {}
    dataset_sizes = {}
    data_loaders['val'] = val_data_loader
    dataset_sizes['val'] = val_dataset_size
    
    # Each epoch has a training and validation phase        
    for phase in ['val']:
        model_i3d.eval()  # Set model to evaluate mode
        running_loss = 0.0
        running_correct = 0

        # Iterate over data.
        progress = tqdm(data_loaders[phase])
        for idx, (data,labels) in enumerate(progress):                                    
            data = data.to(device)
            labels = labels.to(device)

            # forward
            with torch.set_grad_enabled(phase == 'train'):
                out_softmax, out_logits = model_i3d(model_yolo_3d_bottom(data))
                probs, preds = torch.max(out_softmax.data.cpu(), 1)
                loss = criterion(out_softmax.cpu(), labels.cpu())                    
            
                # statistics
                running_loss += loss.item() * data.shape[0]
                running_correct += torch.sum(preds == labels.data.cpu())
                
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_correct.double() / dataset_sizes[phase]                    

        print('phase={}, Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                               
    return 


if __name__ == "__main__":    

    Num_Classes=51
    Batch_Size = 32
    out_frame_num = 16
    Num_Workers = 12

    weight_yolo = "weights/yolov5l.pt"    
    weight_i3d = "weights/yolo_i3d_top_s224_h51_d2_f16_second_round_val_acc_0.7098.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    model_yolo_3d_bottom = YOLO_3D_Bottom(device, weight_yolo) 
    model_i3d = I3D(num_classes=Num_Classes, modality='rgb')
    
    model_i3d.to(device) 
    model_i3d.load_state_dict(torch.load(weight_i3d))

    data_dir = Path('data/hmdb51')
    data_json_dir = "data/hmdb51/data_json_ge32"

    print("device = {}".format(device))

    print("YOLOv5 model: \n{} loaded successfully!".format(weight_yolo))
    print("I3D model: \n{} loaded successfully!".format(weight_i3d))  

    print("cpu_count = {}".format(os.cpu_count()))
    print("Batch_Size = {}".format(Batch_Size))    
    print("out_frame_num = {}".format(out_frame_num)) 
    print("data_dir = {}".format(data_dir))   
    print("data_json_dir = {}".format(data_json_dir))   
    
    model_i3d = modify_and_freeze_i3d_top(model_i3d, Num_Classes, 21)
    model_i3d.to(device) 
    print("model.num_classes = ", model_i3d.num_classes)     

    classes_path= data_dir / "classes.txt"
    class_names = [i.strip() for i in open(classes_path)]
    class_dicts = {k: v for v, k in enumerate(class_names)}

    x = 'val'
    data_pairs_file = data_json_dir + "/val_data_pairs.json"
    val_dataset = RGB_jpg_val_Dataset(data_pairs_file, data_dir/x, class_dicts,
                                        out_frame_num=out_frame_num, x=x)
    val_data_loader = torch.utils.data.DataLoader(\
        val_dataset, batch_size=Batch_Size,shuffle=True, num_workers=Num_Workers)                
    val_dataset_size = len(val_dataset)
    print("val_dataset_size = {}".format(val_dataset_size))      
    
    criterion = nn.CrossEntropyLoss() 

    val_model(model_i3d, model_yolo_3d_bottom, criterion, val_data_loader, val_dataset_size)      
