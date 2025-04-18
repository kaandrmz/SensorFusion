import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from appnet.utils.transform import NormalizeInverse, Patching, RandomPair, Resize ,Pair,\
    RandomCrop,Resizescale,PhotoMetricDistortion,Pad,RandomFlip,Normalize,mmRandomPair,RandomCrop_O,ImageToTensor,to_tensor
from PIL import Image
import mmcv
from mmcv.parallel import DataContainer as DC
class BaseDataset(data.Dataset):
    """Base dataset generator

    Args:
        opt (Namespace): arguments to parse

    Attributes:
        phase (str): train, val phase
        root (str): root directory of dataset
        data (List([dict])): list of data information
        scales (List([int, int])): List of scale (w, h)
        crop_size (Tuple([int, int])): crop size (w, h)
        input_size (Tuple([int, int])): input size (w, h)
        ignore_label (int): index of ignored label
        label2color (Dict): mapping between label and color
        label_reading_mode (enum): label reading mode
        cropping_transform (object): transformation for training
        cropping_transforms (List(object)): list of transformations for validation
        image_transform (object): image transformation
        inverse_transform (object): inverse transformation to reconstruct image
    """

    def __init__(self, opt):
        super().__init__()

        self.phase = opt.phase
        self.root = opt.root
        self.dataset = opt.dataset
        self.model = opt.model
        if opt.phase == "test":
            self.city_test = opt.test_city
        # Parse the file datalist
        self.data = []
        if os.path.isfile(opt.datalist):
            with open(opt.datalist, "r") as f:
                for line in f.readlines():
                    if opt.dataset=="pascal":
                        self.data += [line]
                    else:
                        self.data += [self.parse_info(line)]

        self.scales = opt.scales  # Scale to this size

        self.crop_size = opt.crop_size  # Crop to this size
        self.input_size = opt.input_size  # Resize to this size
        self.nclass = opt.num_classes
        # For label parsing
        self.ignore_label = -1
        self.label2color = {}
        self.label_reading_mode = cv2.IMREAD_COLOR
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
        # Cropping transformation
        if self.phase == "train":
            self.up = Resize([256,256])
            # self.resize = Resizescale(img_scale=(2048, opt.scales[0][0]),ratio_range=(0.5, 2.0))
            # self.resize = Resizescale(img_scale=(2048, opt.scales[0][0]))
            self.resize = Resizescale(img_scale=opt.resize_size)

            self.crop = RandomCrop(crop_size=self.input_size, cat_max_ratio=0.75)
            self.flip = RandomFlip()
            self.photometricdistortion = PhotoMetricDistortion()
            self.pad = Pad(size=self.input_size, pad_val=0, seg_pad_val=255)
            self.normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

            self.cropping_transforms = []

            for scale in self.scales[1:]:
                self.cropping_transforms += [Patching(scale, self.crop_size, Resize(scale), Resize(self.input_size))]
            if "convnext" in self.model:
                self.cropping_transform = mmRandomPair(self.scales[1:], self.crop_size, self.input_size)
            else:
                # For training
                self.cropping_transform = RandomPair(self.scales[1:], self.crop_size, self.input_size)
        elif self.phase == "train_ostg":
            self.cropping_transform = Pair(self.scales[1:],self.crop_size, self.input_size)
        elif self.phase == "trainconvnext" or self.phase == "testconv":
            self.cropping_transform = Resize(self.input_size)
            self.resize = Resizescale(img_scale=(2048, 512), ratio_range=(0.5, 2.0))
            self.crop = RandomCrop(crop_size=self.crop_size, cat_max_ratio=0.75)
            self.flip = RandomFlip()
            self.photometricdistortion = PhotoMetricDistortion()
            self.pad = Pad(size=self.crop_size, pad_val=0, seg_pad_val=255)
            self.normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
            self.tensor = transforms.ToTensor()
        else:
            # For testing
            self.crop = RandomCrop(crop_size=self.input_size, cat_max_ratio=1)
            self.pad = Pad(size=self.input_size, pad_val=0, seg_pad_val=255)
            self.normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
            self.cropping_transforms = []
            self.tensor = ImageToTensor()
            for scale in self.scales:
                self.cropping_transforms += [Patching(scale, self.crop_size, Resize(scale), Resize(self.input_size))]
            self.resize = Resizescale(img_scale=opt.resize_size)

        # For image transformation
        self.image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.inverse_transform = NormalizeInverse([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def parse_info(self, line):
        """Parse information of each line in the filelist

        Args:
            line (str): line in filelist

        Returns:
            dict([str, str]): information of each data item
        """
        info = {}
        tokens = line.strip().split()
        info["image"] = os.path.join(self.root, tokens[0])
        info["label"] = os.path.join(self.root, tokens[1])
        return info

    def __len__(self):
        return len(self.data)

    def image2class(self, label):
        """Convert image to class matrix

        Args:
            label (np.array): h x w x C (= 1 or = 3)
                label image

        Returns:
            np.array: h x w
                class matrix
        """
        if label is None:
            return label
        l, w = label.shape[0], label.shape[1]
        classmap = np.zeros(shape=(l, w), dtype=np.uint8)

        for classnum, color in self.label2color.items():
            indices = np.where(np.all(label == tuple(color[::-1]), axis=-1))
            classmap[indices[0].tolist(), indices[1].tolist()] = classnum
        return classmap

    def class2bgr(self, label):
        """Convert class matrix to BGR image

        Args:
            label (np.array): h x w
                class matrix

        Returns:
            np.array: h x w x 3
                BGR image
        """
        l, w = label.shape[0], label.shape[1]
        bgr = np.zeros(shape=(l, w, 3)).astype(np.uint8)
        for classnum, color in self.label2color.items():
            indices = np.where(label == classnum)
            bgr[indices[0].tolist(), indices[1].tolist(), :] = color[::-1]
        return bgr

    def augment(self, image, label):
        """Augment image and label

        Args:
            image (np.array): h x w x 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label image

        Returns:
            np.array: h x w x 3
                image after augmentation
            np.array: h x w x 3 or h x w x 1
                label image after augmentation
        """
        return image, label
    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image
    def slice_image(self, image):
        """Slice image to patches

        Args:
            image (np.array): H x W x 3
                image to be sliced

        Returns:
            torch.Tensor: N x 3 x h x w
                patches
            torch.Tensor: N
                indices of patches
        """
        image_patches = []
        scale_idx = []

        for scale_id, cropping_transform in enumerate(self.cropping_transforms):

            # Crop data for each scale
            image_crops = cropping_transform(image)

            image_patches += list(map(lambda x: self.image_transform(x), image_crops))
            scale_idx += [scale_id for _ in range(len(image_crops))]

        image_patches = torch.stack(image_patches)
        scale_idx = torch.tensor(scale_idx)
        return image_patches, scale_idx

    def __getitem__(self, index):
        if self.phase == "output":
            image_path = self.data[index]["image"]
            image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            image_patches, scale_idx = self.slice_image(image)
            return {
                "image_patches": image_patches,
                "scale_idx": scale_idx,
                #"label": label,
                "name": image_path.split("/")[-1],
            }

        # Get data information
        if self.dataset == 'pascal':
            path = self.data[index].strip().replace(' ', '').replace('\n', '')
            image_path = os.path.join(self.root+"/Image",path+".jpg")
            label_path = os.path.join(self.root+"/label",path+".png")
        else:
            image_path = self.data[index]["image"]

            label_path = self.data[index]["label"]
        if self.dataset == 'ade20k' or self.dataset == 'cocostuff' or self.dataset == 'pascal':
            label = np.array(Image.open(label_path).convert('P'))
            label = np.array(label)
        else:
            label = cv2.imread(label_path, self.label_reading_mode)
       
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Read label
        # label = cv2.imread(label_path, self.label_reading_mode)

        # Augment data for training phase
        if self.phase == "train":
            image, label = self.augment(image, label)

        
        if self.phase == "train":
            
            label = self.image2class(label)
            image,label = self.resize(image,label)
            image,label = self.crop(image,label)
            image,label = self.flip(image,label)
            image,label = self.pad(image,label)
            
            image = self.image_transform(image)
            
            label_edge = cv2.Laplacian(label, cv2.CV_64F, ksize=5)
            label = torch.from_numpy(label).type(torch.LongTensor)
            label_edge = torch.from_numpy(label_edge).type(torch.LongTensor)
            return {
                "image": image,
                "label_edge":label_edge,
                "label": label,
                 "key_weight":key_weight,
            }
        # Augment data for training phase
        if self.phase == "train1024":
            image, label = self.augment(image, label)

        # For training
        if self.phase == "trainconvnext" or self.phase == "testconv":
            label = self.image2class(label)
            image,label = self.resize(image,label)
            image,label = self.crop(image,label)
            image = self.photometricdistortion(image)
            image = self.normalize(image)
            image,label = self.pad(image,label)
            image = self.tensor(image)
            label_edge = cv2.Laplacian(label, cv2.CV_64F,ksize=3)
            label = torch.from_numpy(label).type(torch.LongTensor)
            label_edge = torch.from_numpy(label_edge).type(torch.LongTensor)
            return {
                "image": image,
                "label": label,
                "label_edge":label_edge,
                "name": image_path.split("/")[-1]
            }

        if self.phase == "train1024":

            image = self.image_transform(image)

            label = self.image2class(label)
            label_edge = cv2.Laplacian(label, cv2.CV_64F,ksize=3)
            label = torch.from_numpy(label).type(torch.LongTensor)
            label_edge = torch.from_numpy(label_edge).type(torch.LongTensor)
            return {
                "image": image,
                "label": label,
                "label_edge":label_edge,
                "name": image_path.split("/")[-1]
            }
        if self.phase == "train_ostg":
            # # Random scale, crop, resize image and label
            coarse_image, _, image_patches, label_patches, info_sum = self.cropping_transform(image, label)
            coarse_image = self.image_transform(coarse_image)
            for i in range(len(image_patches)):

                image_patches[i] = self.image_transform(image_patches[i])

                label_patches[i] = self.image2class(label_patches[i])
                label_patches[i] = torch.from_numpy(label_patches[i]).type(torch.LongTensor)

            return {
                "coarse_image": coarse_image,
                "image_patches": torch.stack(image_patches),
                "label_patches": torch.stack(label_patches),
                # "fine_label_edge": fine_label_edge,
                "coord": torch.tensor(info_sum).unsqueeze(1),
            }
        
        if self.dataset == "pascal" and "hrnet" in self.model:
            label = self.image2class(label)
            image = cv2.resize(image, (520, 520), interpolation=cv2.INTER_LINEAR)
            image = self.image_transform(image)
        elif self.dataset == "ade20k" or self.dataset == "pascal" or self.dataset == "cocostuff":
            label = self.image2class(label)
            image ,_ = self.resize(image,label)
            
            image = self.image_transform(image)
            
        elif self.dataset == "cityscapes":
            
            label = self.image2class(label)
            image = self.image_transform(image)




        return {
            "image":image,
            "label": label,
            "name": image_path.split("/")[-1],
        }














