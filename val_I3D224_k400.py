# by Ruikang Luo
import os
from pathlib2 import Path
from tqdm import tqdm

import torch
import torch.nn as nn

from i3d_src.utils import *
from i3d_src.opts import parser
from i3d_src.i3d_s224 import I3D, Unit3Dpy
from i3d_src.DataLoader_i3d_k400_DS1_ge128 import RGB_jpg_train_Dataset, RGB_jpg_val_Dataset

def val_model(model, criterion, val_data_loader, val_dataset_size):      
    data_loaders = {}
    dataset_sizes = {}
    data_loaders['val'] = val_data_loader
    dataset_sizes['val'] = val_dataset_size
    
    # Each epoch has a training and validation phase        
    for phase in ['val']:
        model.eval()  # Set model to evaluate mode
        running_loss = 0.0
        running_correct = 0

        # Iterate over data.
        progress = tqdm(data_loaders[phase])
        for idx, (data,labels) in enumerate(progress):                                    
            data = data.to(device)
            labels = labels.to(device)

            # forward
            with torch.set_grad_enabled(phase == 'train'):
                out_softmax, out_logits = model(data)
                probs, preds = torch.max(out_softmax.data.cpu(), 1)
                loss = criterion(out_softmax.cpu(), labels.cpu())                    
            
                # statistics
                running_loss += loss.item() * data.shape[0]
                running_correct += torch.sum(preds == labels.data.cpu())
                
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_correct.double() / dataset_sizes[phase]                    

        print('phase={}, Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                               
    return 


def load_and_freeze_model(model, num_classes, num_freeze=20):
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
            print("L{}: Layer {} frozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = False
        else:
            print("L{}: Layer {} unfrozen!".format(counter, child._get_name()))
            for param in child.parameters():
                param.requires_grad = True   
                param.requires_grad = False
    return model 


def main(device, val_data_loader, val_dataset_size):
    Num_Classes=400
    DropoutProb = 0
    weight_path = "weights/i3d_s224_k400_d1_f32_val_acc_0.6100.pth"
    
    model = I3D(num_classes=Num_Classes, modality='rgb', dropout_prob=DropoutProb)
    print("dropout_prob = ", DropoutProb)
    model = load_and_freeze_model(model, Num_Classes, 20)
    model.to(device)      
    
    model.load_state_dict(torch.load(weight_path))
    print("i3dm_k400_model \n{} \nloaded successfully!".format(weight_path))
    
    print("model.num_classes = ", model.num_classes)    
   
    criterion = nn.CrossEntropyLoss()             
 
    val_model(model, criterion, val_data_loader, val_dataset_size)   


if __name__ == "__main__":
    args = parser.parse_args()

    Num_Frames = 32
    Batch_Size = 32
    Num_Workers = 12
    
    data_dir = Path("data/k400_imgs")    
    data_json_dir = "data/k400_imgs/data_json_ge128"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    file_name =  os.path.basename(__file__)
    print("start running {}".format(file_name))
    print("device = {}".format(device))
    print("cpu_count = {}".format(os.cpu_count()))
    print("batch_size = {}".format(Batch_Size))    
    print("out_frame_num = {}".format(Num_Frames))    
    print("data_dir = {}".format(data_dir))   
    print("data_json_dir = {}".format(data_json_dir))     
    
    classes_path= data_dir / "classes.txt"
    class_names = [i.strip() for i in open(classes_path)]
    class_dicts = {k: v for v, k in enumerate(class_names)}

    with Timer('main_of_training'):              
        x = 'val'
        data_pairs_file = data_json_dir + "/val_data_pairs.json"
        val_dataset = RGB_jpg_val_Dataset(data_pairs_file, data_dir/x, class_dicts,
                                            out_frame_num=Num_Frames, x=x)
        val_data_loader = torch.utils.data.DataLoader(\
            val_dataset, batch_size=Batch_Size,shuffle=True, num_workers=Num_Workers)          
            
        val_dataset_size = len(val_dataset)
        print("val_dataset_size = {}".format(val_dataset_size))    
        
        main(device, val_data_loader, val_dataset_size)