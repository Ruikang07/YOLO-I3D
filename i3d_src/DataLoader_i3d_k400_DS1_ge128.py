import os
import json
import random
from PIL import Image
import torch
from pathlib2 import Path
from torch.utils.data import Dataset

import torchvision.transforms as transforms

Min_Frame_Num = 128
DownSample = 1
Image_Size_L = 256
Image_Size_S = 224

def train_Dataset_split(data_pairs_file, len_i, root_dir, class_dict):

    sub_dirs = [i for i in root_dir.iterdir() if i.is_dir() and not i.stem.startswith('.')]
    class_names = [i.stem for i in sub_dirs]
    data_pairs = []
    file_names = []
    # Iterate over a sequence of numbers from 0 to 9
    for i in range(len_i):
        # In each iteration, add an empty list to the main list
        data_pairs.append([])
        file_names.append(data_pairs_file+"_"+str(len_i)+"_"+str(i+1)+".json")
        print("file_names[{}] = {}".format(i, file_names[i]))
     
    for sub_dir in sub_dirs:
        item_dirs = [i for i in sub_dir.iterdir() if i.is_dir() and not i.stem.startswith('.')]
        for item_dir in item_dirs:
            contents = [i.as_posix() for i in item_dir.iterdir() if i.is_file() and not i.stem.startswith('.')] 
            contents.sort()
            if len(contents) >= Min_Frame_Num : 
                if len_i>1 :
                    i = random.randint(0, len_i-1)
                    data_pairs[i].append((contents, class_dict[sub_dir.stem]))   
                else:
                    data_pairs[0].append((contents, class_dict[sub_dir.stem]))     
    
    for i in range(len_i):    
        with open(file_names[i], "w") as f:
            json.dump(data_pairs[i], f)    
        
        print("train: len(data_pairs[{})] = {}".format(i, len(data_pairs[i])))
       
        
pil_2_tensor = transforms.Compose(
    [transforms.ToTensor(), #[0-255] to [0.0-1.0]
    ])

spacial_train_transform = transforms.Compose(
    [transforms.Resize(Image_Size_L),
    transforms.CenterCrop(Image_Size_L), 
    transforms.RandomCrop(Image_Size_S),
    transforms.RandomHorizontalFlip(), #p – probability of the image being flipped. Default=0.5
    ])

spacial_val_transform = transforms.Compose(
    [transforms.Resize(Image_Size_L),
    transforms.CenterCrop(Image_Size_S),
    ])

def temporal_train_transform(frames, sample_num):
    frame_num = len(frames) 
    if frame_num >= DownSample*sample_num:
        frames = frames[::DownSample]  
    frame_num = len(frames)      

    if sample_num <= frame_num :
        start_frame = random.randint(0, (frame_num - sample_num))
        end_frame = start_frame + sample_num
        rgb_sample = frames[start_frame:end_frame]   
        return rgb_sample

    if sample_num > 4*frame_num:
        len2 = sample_num - 4*frame_num
        rgb_sample = frames + frames + frames + frames + frames[0:len2] 
    elif sample_num > 3*frame_num:
        len2 = sample_num - 3*frame_num
        rgb_sample = frames + frames + frames + frames[0:len2]  
    elif sample_num > 2*frame_num:
        len2 = sample_num - 2*frame_num
        rgb_sample = frames + frames + frames[0:len2]          
    else:
        len2 = sample_num - frame_num
        rgb_sample = frames + frames[0:len2]     
    return rgb_sample

def temporal_val_transform(frames, sample_num):
    frame_num = len(frames) 
    if frame_num >= DownSample*sample_num:
        frames = frames[::DownSample]  
    frame_num = len(frames) 

    if sample_num <= frame_num :
        start_frame = int((frame_num - sample_num)/2)
        end_frame = start_frame + sample_num
        rgb_sample = frames[start_frame:end_frame]
        return rgb_sample

    if sample_num > 4*frame_num:
        len2 = sample_num - 4*frame_num
        rgb_sample = frames + frames + frames + frames + frames[0:len2] 
    elif sample_num > 3*frame_num:
        len2 = sample_num - 3*frame_num
        rgb_sample = frames + frames + frames + frames[0:len2]  
    elif sample_num > 2*frame_num:
        len2 = sample_num - 2*frame_num
        rgb_sample = frames + frames + frames[0:len2]          
    else:
        len2 = sample_num - frame_num
        rgb_sample = frames + frames[0:len2]     
    return rgb_sample


class RGB_jpg_train_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_pairs_file, root_dir, class_dict, out_frame_num=32, x=''):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = Path(root_dir)
        self.sub_dirs = [i for i in self.root_dir.iterdir() if i.is_dir() and not i.stem.startswith('.')]
        self.class_names = [i.stem for i in self.sub_dirs]
        self.data_pairs = []
        self.data_pairs_file = data_pairs_file
        self.out_frame_num = out_frame_num

        if os.path.isfile(self.data_pairs_file):
                try:
                    with open(self.data_pairs_file, "r") as f:
                        self.data_pairs = json.load(f)
                except:
                    print("can not open " + self.data_pairs_file)  
                    exit(0)
        else:            
            data_pairs_file = Path(data_pairs_file)
            data_pairs_root = str(data_pairs_file.parent)
            file_name = data_pairs_file.stem
            parts = file_name.split("_")
            len_i = int(parts[3])
            data_pairs_root = data_pairs_root+"/"+parts[0]+"_"+parts[1]+"_"+parts[2]
            print("root_dir = ", root_dir)
            train_Dataset_split(data_pairs_root, len_i, root_dir, class_dict)   
            try:
                with open(self.data_pairs_file, "r") as f:
                    self.data_pairs = json.load(f)
            except:
                print("can not open " + self.data_pairs_file)  
                exit(0)      

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        images_list = self.data_pairs[idx][0]
        images_list = temporal_train_transform(images_list, self.out_frame_num)
        rgb_data = self.images_loader(images_list)
        #print("rgb_data.shape = ", rgb_data.shape)
        #rgb_data.shape =  (T,Ch,H,W,)     
        
        rgb_data = spacial_train_transform(rgb_data)
        #print("rgb_data.shape = ", rgb_data.shape)
        #rgb_data.shape = (T,Ch,H,W) for Pytorch

        rgb_data = rgb_data.permute(1,0,2,3)
        #rgb_data.shape = (Ch,T,H,W) for I3D
        #i3d input.shape = (B,Ch,T,H,W)
        
        return rgb_data, self.data_pairs[idx][1]
        
    def images_loader(self, images_list):
        images_set = []
        for image_path in images_list:
            image = Image.open(Path(image_path).as_posix()).convert('RGB')
            images_set.append(pil_2_tensor(image))
        return torch.stack(images_set)


class RGB_jpg_val_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_pairs_file, root_dir, class_dict, out_frame_num=32, x=''):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = Path(root_dir)
        self.sub_dirs = [i for i in self.root_dir.iterdir() if i.is_dir() and not i.stem.startswith('.')]
        self.class_names = [i.stem for i in self.sub_dirs]
        self.data_pairs = []
        self.data_pairs_file = data_pairs_file
        self.out_frame_num = out_frame_num    

        if os.path.isfile(self.data_pairs_file):
                try:
                    with open(self.data_pairs_file, "r") as f:
                        self.data_pairs = json.load(f)
                except:
                    print("can not open " + self.data_pairs_file)   
                    exit(0)
        else:          
            for sub_dir in self.sub_dirs:     
                item_dirs = [i for i in sub_dir.iterdir() if i.is_dir() and not i.stem.startswith('.')]
                for item_dir in item_dirs:
                    contents = [i.as_posix() for i in item_dir.iterdir() if i.is_file() and not i.stem.startswith('.')] 
                    contents.sort()
                    if len(contents) >= Min_Frame_Num : 
                        self.data_pairs.append((contents, class_dict[sub_dir.stem]))
            #write data to json File.
            with open(self.data_pairs_file, "w") as f:
                json.dump(self.data_pairs, f)    

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        images_list = self.data_pairs[idx][0]
        images_list = temporal_val_transform(images_list, self.out_frame_num)
        rgb_data = self.images_loader(images_list)
        #rgb_data.shape =  (T,Ch,H,W)

        rgb_data = spacial_val_transform(rgb_data)
        #rgb_data.shape = (T,Ch,H,W) for Pytorch

        rgb_data = rgb_data.permute(1,0,2,3)
        #rgb_data.shape = (Ch,T,H,W) for I3D
        #i3d input.shape = (B,Ch,T,H,W)

        return rgb_data, self.data_pairs[idx][1]
        
    def images_loader(self, images_list):
        images_set = []
        for image_path in images_list:
            image = Image.open(Path(image_path).as_posix()).convert('RGB')
            images_set.append(pil_2_tensor(image))
        return torch.stack(images_set)

