import math
from numpy import random
import mmcv
from mmcv.utils import deprecated_api_warning, is_tuple_of
import cv2
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')
class SegCompose(object):
    """Compose of augmentations applied to segmentation data

    Args:
        augmenters (List([object])): list of augmenters
    """

    def __init__(self, augmenters):
        super().__init__()
        self.augmenters = augmenters

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        for augmenter in self.augmenters:
            image, label = augmenter(image, label)
        return image, label


class OneOf(object):
    """Choose one of augmentations and apply to image and label

    Args:
        augmenters (List([object])): list of augmenters
    """

    def __init__(self, augmenters):
        super().__init__()
        self.augmenters = augmenters

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        augmenter = random.choice(self.augmenters)
        return augmenter(image, label)


class Resize(object):
    """Resize image and label

    Args:
        size (Tuple([int, int])): size to resize (width, height)
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        width, height = self.size
        #print(width,height)
        h, w = image.shape[0], image.shape[1]
        #print(h,w)
        if width == -1:
            width = int(height / h * w)
        if height == -1:
            height = int(width / w * h)

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        #print(image.shape)
        label = label if label is None else cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
        return image, label


class Patching(object):
    """Patch processing image

    Args:
        scale (Tuple([int, int])): scale size (width, height)
        crop (Tuple([int, int])): crop size (width, height)
        preaugmenter (object, optional): augmenter applied before cropping. Defaults to None.
        post_augmenter (object, optional): augmenter applied after cropping. Defaults to None.
    """

    def __init__(self, scale, crop, pre_augmenter=None, post_augmenter=None):
        super().__init__()
        crops = []
        n_x = math.ceil(scale[0] / crop[0])
        # print(n_x)
        step_x = int(crop[0] - (n_x * crop[0] - scale[0]) / max(n_x - 1, 1.0))

        n_y = math.ceil(scale[1] / crop[1])
        # print(n_y)
        step_y = int(crop[1] - (n_y * crop[1] - scale[1]) / max(n_y - 1, 1.0))
        #print(n_x,n_y,step_x,step_y)
        for x in range(n_x):
            for y in range(n_y):
                crops += [(x * step_x, y * step_y, x * step_x + crop[0], y * step_y + crop[1])]
        self.crops = crops
        self.pre_augmenter = pre_augmenter
        self.post_augmenter = post_augmenter

    def __call__(self, image):
        """Apply augmentation

        Args:
            image (np.array): H x W 3
                image

        Returns:
            List([np.array]): [h x w x 3]
                list of crops
        """
        if self.pre_augmenter is not None:
            image, _ = self.pre_augmenter(image, None)
        #print(image.shape)
        image_crops = []
        for crop in self.crops:
            x, y, xmax, ymax = crop
            im = image[y:ymax, x:xmax]
            if self.post_augmenter is not None:
                im, _ = self.post_augmenter(im, None)

            image_crops += [im]
        # print(len(image_crops))

        return image_crops


class RandomCrop_O(object):
    """Random crop image and label

    Args:
        size (Tuple([int, int])): size of input
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        # print(label.shape)
        max_x = image.shape[1] - self.size[0]
        max_y = image.shape[0] - self.size[1]

        x = np.random.randint(0, max_x) if max_x >0 else 0
        y = np.random.randint(0, max_y) if max_y >0 else 0
        # print(x,y)
        image = image[y : y + self.size[1], x : x + self.size[0]]
        label = label if label is None else label[y : y + self.size[1], x : x + self.size[0]]
        # print(image.shape,label.shape)
        return image, label


class RandomRotate(object):
    """Random rotate image and label

    Args:
        max_angle (int): maximum angle when applying rotation
    """

    def __init__(self, max_angle):
        super().__init__()
        self.max_angle = max_angle

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        angle = random.randint(0, self.max_angle * 2) - self.max_angle

        image = Image.fromarray(image)
        image = image.rotate(angle, resample=Image.BILINEAR)
        image = np.array(image)

        if label is not None:
            label = Image.fromarray(label)
            label = label.rotate(angle, resample=Image.NEAREST)
            label = np.array(label)

        return image, label


class RandomFlip(object):
    """Random flip image and label"""

    def __init__(self):
        super().__init__()

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        if np.random.rand() > 0.5:
            # image = cv2.flip(image, 1)
            image = mmcv.imflip(
                image, direction="horizontal")
            # label = cv2.flip(label, 1)
            label = mmcv.imflip(
                label, direction="horizontal")

        return image, label


class RandomPair(object):
    """Random generate pairs for training

    Args:
        scales (List([Tuple([int, int])])): list of scales
        crop_size (Tuple([int, int])): crop size (width, height)
        input_size (Tuple([int, int])): input size (width, height)
    """

    def __init__(self, scales, crop_size, input_size):
        super().__init__()
        self.scales = [Resize(scale) for scale in scales]
        self.crop_size = crop_size
        self.resize = Resize(input_size)

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            np.array: h x w x 3
                coarse image
            np.array: h x w x 3 or h x w x 1
                coarse label
            np.array: h x w x 3
                fine image
            np.array: h x w x 3 or h x w x 1
                fine label
            Tuple([float, float, float, float]): cropping information
        """
        # Resize to get coarse scale
        coarse_image, coarse_label = self.resize(image, label)
        #print(image.shape,label.shape)
        #print(coarse_image.shape,coarse_label.shape,'f1')

        # Pick random scale
        fine_image, fine_label = random.choice(self.scales)(image, label)

        #print(fine_image.shape[1],fine_image.shape[0],self.crop_size[0],self.crop_size[1],'f2')
        # Random crop
        max_x = fine_image.shape[1] - self.crop_size[0]
        max_y = fine_image.shape[0] - self.crop_size[1]
        #print(max_y,max_x)
        x = np.random.randint(0, max_x) if max_x >0 else 0
        y = np.random.randint(0, max_y) if max_y >0 else 0

        xmin = x * 1.0 / fine_image.shape[1] * coarse_image.shape[1]
        ymin = y * 1.0 / fine_image.shape[0] * coarse_image.shape[0]
        xmax = xmin + (self.crop_size[0] * 1.0 / fine_image.shape[1] * coarse_image.shape[1])
        ymax = ymin + (self.crop_size[1] * 1.0 / fine_image.shape[0] * coarse_image.shape[0])

        fine_image = fine_image[y : y + self.crop_size[1], x : x + self.crop_size[0]]
        fine_label = fine_label[y : y + self.crop_size[1], x : x + self.crop_size[0]]
        #print(fine_image.shape,fine_label.shape)
        info = (xmin, ymin, xmax, ymax)
        #print(info)
        # Resize fine image
        fine_image, fine_label = self.resize(fine_image, fine_label)
        #print(fine_image.shape,fine_label.shape,'f3')

        return coarse_image, coarse_label, fine_image, fine_label, info

class mmRandomPair(object):
    """Random generate pairs for training

    Args:
        scales (List([Tuple([int, int])])): list of scales
        crop_size (Tuple([int, int])): crop size (width, height)
        input_size (Tuple([int, int])): input size (width, height)
    """

    def __init__(self, scales, crop_size, input_size):
        super().__init__()
        self.scales = [Resize(scale) for scale in scales]
        self.crop_size = crop_size
        self.resize = Resize(input_size)
        self.randomcrop = RandomCrop_O(input_size)

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            np.array: h x w x 3
                coarse image
            np.array: h x w x 3 or h x w x 1
                coarse label
            np.array: h x w x 3
                fine image
            np.array: h x w x 3 or h x w x 1
                fine label
            Tuple([float, float, float, float]): cropping information
        """
        # Resize to get coarse scale
        # print(image.shape,label.shape)
        image,label = self.randomcrop(image,label)
        print(label.shape)
        coarse_image, coarse_label = self.resize(image, label)
        # print(coarse_image[:,1,1],coarse_label[1,:])
        # print(coarse_image.shape,coarse_label.shape,'f1')

        # Pick random scale
        fine_image, fine_label = random.choice(self.scales)(image, label)
        # print(fine_image.shape)
        # print(fine_image.shape[1],fine_image.shape[0],self.crop_size[0],self.crop_size[1],'f2')
        # Random crop
        max_x = fine_image.shape[1] - self.crop_size[0]
        max_y = fine_image.shape[0] - self.crop_size[1]
        # print(max_y,max_x)
        x = np.random.randint(0, max_x) if max_x > 0 else 0
        y = np.random.randint(0, max_y) if max_y > 0 else 0

        xmin = x * 1.0 / fine_image.shape[1] * coarse_image.shape[1]
        ymin = y * 1.0 / fine_image.shape[0] * coarse_image.shape[0]
        xmax = xmin + (self.crop_size[0] * 1.0 / fine_image.shape[1] * coarse_image.shape[1])
        ymax = ymin + (self.crop_size[1] * 1.0 / fine_image.shape[0] * coarse_image.shape[0])

        fine_image = fine_image[y: y + self.crop_size[1], x: x + self.crop_size[0]]
        fine_label = fine_label[y: y + self.crop_size[1], x: x + self.crop_size[0]]
        # print(fine_image.shape,fine_label.shape)
        info = (xmin, ymin, xmax, ymax)
        # print(info)
        # Resize fine image
        fine_image, fine_label = self.resize(fine_image, fine_label)
        # print(fine_image.shape,fine_label.shape,'f3')

        return coarse_image, coarse_label, fine_image, fine_label, info
class Pair(object):
    """Random generate pairs for training

    Args:
        scales (List([Tuple([int, int])])): list of scales
        crop_size (Tuple([int, int])): crop size (width, height)
        input_size (Tuple([int, int])): input size (width, height)
    """

    def __init__(self,scales, crop_size, input_size):
        super().__init__()

        self.scales = scales
        self.crop_size = crop_size
        self.resize = Resize(input_size)

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            np.array: h x w x 3
                coarse image
            np.array: h x w x 3 or h x w x 1
                coarse label
            np.array: h x w x 3
                fine image
            np.array: h x w x 3 or h x w x 1
                fine label
            Tuple([float, float, float, float]): cropping information
        """
        # Resize to get coarse scale
        coarse_image, coarse_label = self.resize(image, label)
        #print(image.shape,label.shape)
        #print(coarse_image.shape,coarse_label.shape,'f1')

        # Pick  scale
        image_patches = []
        label_patches = []
        info_sum = []
        scales = self.scales
        for scale in scales:

            # image = cv2.resize(image, scale, interpolation=cv2.INTER_LINEAR)
            # label = cv2.resize(label, scale, interpolation=cv2.INTER_NEAREST)
            # print(image.shape)


            #print(fine_image.shape[1],fine_image.shape[0],self.crop_size[0],self.crop_size[1],'f2')
            # Random crop
            max_x = image.shape[1] - scale[0]
            max_y = image.shape[0] - scale[1]
            #print(max_y,max_x)
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            xmin = x * 1.0 / image.shape[1] * self.crop_size[0]
            ymin = y * 1.0 / image.shape[0] * self.crop_size[1]
            xmax = xmin + (scale[0] * 1.0 / image.shape[1] * self.crop_size[0])
            ymax = ymin + (scale[1] * 1.0 / image.shape[0] * self.crop_size[1])
            fine_image = image[y : y + scale[1], x : x + scale[0]]
            fine_label = label[y : y + scale[1], x : x + scale[0]]
            image = fine_image
            label = fine_label
            fine_image, fine_label = self.resize(fine_image, fine_label)
            image_patches.append(fine_image)
            label_patches.append(fine_label)
            #print(fine_image.shape,fine_label.shape)
            info = (xmin, ymin, xmax, ymax)
            info_sum.append(info)
            #print(info)
            # Resize fine image
            #print(fine_image.shape,fine_label.shape,'f3')

        return coarse_image, coarse_label, image_patches, label_patches, info_sum


class NormalizeInverse(TF.Normalize):
    """Undoes the normalization and returns the reconstructed images in the input domain.

    Args:
        mean (List([float, float, float])): mean value
        std (List([float, float, float])): standard deviation
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class Resizescale(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, img):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = img.shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError
        # print(scale)

        return scale

    def _resize_img(self, img, scale):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            # print(img.shape)
            img, scale_factor = mmcv.imrescale(
                img, scale, return_scale=True)


        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
        return img


    def _resize_seg(self, seg, scale):
        """Resize semantic segmentation map with ``results['scale']``."""

        if self.keep_ratio:
            gt_seg = mmcv.imrescale(
                seg, scale, interpolation='nearest')
        else:
            gt_seg = mmcv.imresize(
                seg, scale, interpolation='nearest')
        return gt_seg

    def __call__(self, img, seg):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        scale = self._random_scale(img)
        # print(scale)
        img = self._resize_img(img,scale)
        seg = self._resize_seg(seg,scale)

        return img, seg


class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        # print(img.shape)
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, img, seg):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """


        crop_bbox = self.get_crop_bbox(img)
        # print(crop_bbox)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(seg, crop_bbox)
                # print(seg_temp.shape)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)
        # print(crop_bbox)
        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        # print(img_shape)
        # crop semantic seg
        seg = self.crop(seg, crop_bbox)
        return img,seg

class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, img):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        # img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)


        return img

class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, img):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val)

        return padded_img, padded_img.shape

    def _pad_seg(self, seg, shape):
        """Pad masks according to ``results['pad_shape']``."""
        if seg is None:
            return None
        seg = mmcv.impad(
            seg,
            shape=shape[:2],
            pad_val=self.seg_pad_val)
        return seg

    def __call__(self, img, seg):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        img, shape = self._pad_img(img)
        seg = self._pad_seg(seg,shape)
        return img, seg
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, img):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        img = mmcv.imnormalize(img, self.mean, self.std,
                                          self.to_rgb)

        return img
class ImageToTensor(object):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self):
        self.keys = 1

    def __call__(self, img):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = to_tensor(img.transpose(2, 0, 1))
        return img