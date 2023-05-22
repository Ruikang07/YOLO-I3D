
# YOLO-I3D

YOLO-I3D is a hybrid 2D/3D structure. The YOLOv5 Bottom is used to extract the spatial features and the I3D Top is used to extract spatial-temporal features and classify the action type by using spatial-temporal features. <br>
<br>
This code is based on Ultralytics's [YOLOv5](https://github.com/ultralytics/yolov5) and Miracleyoo's [Trainable-i3d-pytorch](https://github.com/miracleyoo/Trainable-i3d-pytorch)
<br><br>
<img src="YOLO-I3D.png">
&nbsp;              &nbsp;Figure 1: Architecture of proposed YOLO_I3D.

<br><br>


## Setup

```shell
git clone https://github.com/Ruikang07/YOLO-I3D.git
cd i3d_light

conda create --name yolo_i3d python=3.7
conda activate yolo_i3d
pip install -r requirements.txt
```


## Sub-epoch and Sub-dataset
Since the dataset Kinetics400 is very big, each training epoch takes about a few hours. In order to check the effect of hyper-parameters optimization and avoid wasting too much time with wrong hyper-parameters, we use sub-epoch-based training instead of ordinary epoch-based training. The details of the sub-epoch runners are as follows.

### Sub-epoch runner

Step 1: Loading and preprocessing data. <br>
1)	Load training dataset and validation dataset.<br>
2)	Divide the whole training dataset into N (such as 10) subsets randomly with equal probability. Keep validation dataset undivided. <br>
3)	Create N sub-data-loaders corresponding to the N sub-datasets for training and one data loader for validation dataset. <br><br>

Step 2: Training. <br>
1)	Divide one epoch into N sub-epochs. Each sub-epoch performs a training on the corresponding sub-dataset and a validation on the whole validation dataset. <br>
2)	If the validation accuracy is not improved after a specified number of training epochs, the learning rate is updated based on a given schedule and model weights are set to the values which produced the best validation accuracy until that time point.<br><br>

Step 3: Stop the training if epoch number reaches a specified value or early stop condition is satisfied. Otherwise go back to Step 2.<br><br>

In order to save the time of creating sub-datasets, the information of sub-datasets can be saved in json files.<br><br>



## Dataset Folder Structure

```
dataset_in_imgs
├── classes.txt
├── train
│   ├── action1
│   │   ├── video1
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   ├── video2
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   └── ...
│   ├── action2
│   │   ├── video1
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   ├── video2
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── img ...
│   │   └── ...
│   └── ...
│
├── val
│   ├── action1
│   │   ├── video1
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   ├── video2
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   └── ...
│   ├── action2
│   │   ├── video1
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   ├── video2
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── img ...
│   │   └── ...
│   └── ...
│
└── json_file_dir
```

<br><br>

## lr scheduler
For simplicity and flexibility, we use a list of predefined lr values as a lr scheduler.
