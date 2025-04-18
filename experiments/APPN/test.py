import math
import os
import time
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from appnet.dataset import get_dataset_with_name
from appnet.model import get_model_with_name
from appnet.model.refinement_msaugcpel import RefinementAppNet
from appnet.options.test import TestOptions
from appnet.utils.blur import MedianBlur,GaussianBlur
from appnet.utils.geometry import (
    calculate_certainty,
    ensemble,
    get_patch_coords,
    get_uncertain_point_coords_on_grid,
    point_sample,
)
from appnet.utils.transform import Patching,Resize
from appnet.utils.metrics import confusion_matrix, get_freq_iou, get_overall_iou
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
from tqdm import tqdm

from thop import profile
from thop import clever_format

from mmseg.ops import resize
image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
@torch.no_grad()
def get_batch_predictions(model, sub_batch_size, patches, another=None):
    """Inference model with batch

    Args:
        model (nn.Module): model to inference
        sub_batch_size (int): batch size
        patches (torch.Tensor): B x C x H x W
            patches to infer
        another (torch.Tensor, optional): B x C x H x W, another inputs. Defaults to None.

    Returns:
        torch.Tensor: B x C x H x W
            predictions (after softmax layer)
    """
    preds = []
    n_patches = patches.shape[0]
    n_batches = math.ceil(n_patches / sub_batch_size)

    # Process each batch
    for batch_idx in range(n_batches):
        max_index = min((batch_idx + 1) * sub_batch_size, n_patches)
        batch = patches[batch_idx * sub_batch_size : max_index]
        with torch.no_grad():
            if another is None:
                pred,_ = model(batch)
                preds += [pred]
            else:
                preds += [model(batch, another[batch_idx * sub_batch_size : max_index])]
    preds = torch.cat(preds, dim=0)
    return preds
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def id2trainId(label, id_to_trainid, reverse=False):
    label_copy = label.copy()
    if reverse:
        for v, k in id_to_trainid.items():
            label_copy[label == k] = v
    else:
        for k, v in id_to_trainid.items():
            label_copy[label == k] = v
    return label_copy
def patches_inference(crop_img,model):
    up_crop_img = F.interpolate(crop_img, scale_factor=2, mode="bilinear", align_corners=True)
    up_crop_img1 = up_crop_img[:, :, :crop_img.shape[2], :crop_img.shape[3]]
    up_crop_img2 = up_crop_img[:, :, crop_img.shape[2]:, crop_img.shape[3]:]
    up_crop_img3 = up_crop_img[:, :, crop_img.shape[2]:, :crop_img.shape[3]]
    up_crop_img4 = up_crop_img[:, :, :crop_img.shape[2], crop_img.shape[3]:]
    up_crop_imgs = torch.cat([up_crop_img1, up_crop_img2, up_crop_img3, up_crop_img4], dim=0)
    preds = []
    for i in range(4):
        pred = model(up_crop_imgs[i].unsqueeze(0))
        preds += [pred]
    middle_map = F.interpolate(torch.zeros_like(pred), scale_factor=2, mode="bilinear",align_corners=False)
    middle_map[:, :, :pred.shape[2], :pred.shape[3]] = preds[0]
    middle_map[:, :, pred.shape[2]:, pred.shape[3]:] = preds[1]
    middle_map[:, :, pred.shape[2]:, :pred.shape[3]] = preds[2]
    middle_map[:, :, :pred.shape[2], pred.shape[3]:] = preds[3]
    return middle_map


def slide_inference(model, refinement_models,stride,img,scales,tmp,num_classes,device, rescale=None):
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """
    h_stride, w_stride = stride
    h_crop, w_crop = scales[0][::-1]
    batch_size, _, h_img, w_img = img.size()
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    co_preds = img.new_zeros((batch_size, num_classes, h_img, w_img)).to(device)
    pa_preds = img.new_zeros((batch_size, num_classes, h_img, w_img)).to(device)
    lo_preds = img.new_zeros((batch_size, num_classes, h_img, w_img)).to(device)
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img)).to(device)
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            coarse_pred = model(crop_img.to(device))
            middle_map = patches_inference(crop_img.to(device), model)
            coarse_preds = resize(
                input=coarse_pred,
                size=crop_img.shape[2:],
                mode='bilinear',
                align_corners=rescale)
            middle_preds = resize(
                input=middle_map,
                size=crop_img.shape[2:],
                mode='bilinear',
                align_corners=rescale)
            co_preds += F.pad(coarse_preds,
                              (int(x1), int(co_preds.shape[3] - x2), int(y1), int(co_preds.shape[2] - y2)))
            pa_preds += F.pad(middle_preds,
                              (int(x1), int(pa_preds.shape[3] - x2), int(y1), int(pa_preds.shape[2] - y2)))
            certainty_mask = torch.where(middle_preds.argmax(1) != coarse_preds.argmax(1),
                                         torch.ones_like(middle_preds.argmax(1)),
                                         torch.zeros_like(middle_preds.argmax(1)))
            certainty_mask_sum = torch.sum(certainty_mask.contiguous().view(-1) != 0)
            if certainty_mask_sum <= int(scales[0][0]*scales[0][1]/tmp):
                fine_pred = refinement_models[0](coarse_pred, middle_preds)
            else:
                fine_pred = coarse_preds
            logit_pred = resize(
                input=fine_pred,
                size=crop_img.shape[2:],
                mode='bilinear',
                align_corners=False)
            lo_preds += F.pad(logit_pred,
                              (int(x1), int(lo_preds.shape[3] - x2), int(y1),
                               int(lo_preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    coarse_pred = (co_preds / count_mat)
    logit_pred = (lo_preds / count_mat)
    middle_pred = (pa_preds / count_mat)
    return coarse_pred,logit_pred,middle_pred

@torch.no_grad()
def main():
    # Parse arguments
    opt = TestOptions().parse()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Create dataset
    dataset = get_dataset_with_name(opt.dataset)(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    # Create model
    if 'convnext' in opt.model or 'vitadapter' in opt.model:
        model = get_model_with_name(opt.model)(opt.num_classes, opt.channels).to(device)
        print("Load model weight from", opt.pretrained)
        state_dict = torch.load(opt.pretrained)
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['state_dict'].items()})
    elif 'beitadapter' in opt.model or 'maskadapter' in opt.model:
        model = get_model_with_name(opt.model)(opt.num_classes, opt.channels).to(device)
        print("Load model weight from", opt.pretrained)
        state_dict = torch.load(opt.pretrained)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict['state_dict'].items()}
        model.load_state_dict(state_dict=state_dict,strict=False)
    elif 'condnet' in opt.model:
        model = get_model_with_name(opt.model)(opt.num_classes).to(device)
        print("Load model weight from", opt.pretrained)
        state_dict = torch.load(opt.pretrained)
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['state_dict'].items()})
    elif opt.model == "hrnet48+ocr-1024" or "hrnet" in opt.model:
        model = get_model_with_name(opt.model)(opt.num_classes).to(device)
        print("Load model weight from", opt.pretrained)
        state_dict = torch.load(opt.pretrained)
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in state_dict.items()
                           if k[6:] in model_dict.keys()}
        model.load_state_dict(pretrained_dict)
    _ = model.eval()
    # Create refinement models
    pretrained_weight = [opt.pretrained_refinement]
    if isinstance(opt.pretrained_refinement, list):
        assert len(opt.scales) - 1 == len(
            opt.pretrained_refinement
        ), "The number of refinement weights must match (no.scales - 1)"
        pretrained_weight = opt.pretrained_refinement
    refinement_models = []

    # Load pretrained weight of refinement modules
    for weight_path in pretrained_weight:
        refinement_model = RefinementAppNet(opt.num_classes,opt.input_size, use_bn=True).to(device)
        state_dict = torch.load(weight_path)
        refinement_model.load_state_dict(state_dict, strict=False)
        _ = refinement_model.eval()
        refinement_models += [refinement_model]

    # Patch coords
    patch_coords = []
    cropping_transforms = []
    for scale in opt.scales:
        patch_coords += [torch.tensor(get_patch_coords(scale, opt.crop_size)).to(device)]
        cropping_transforms += [Patching(scale, opt.crop_size, Resize(scale), Resize(opt.input_size))]

    # Allocate prediction map
    _, H, W = opt.num_classes, opt.scales[-1][1], opt.scales[-1][0]
    final_output = None

    # Blur operator
    median_blur = MedianBlur(kernel_size=(opt.smooth_kernel, opt.smooth_kernel)).to(device)
    # median_blur = GaussianBlur(3,kernel_size=(opt.smooth_kernel, opt.smooth_kernel),sigma=(1.0,1.0)).to(device)
    median_blur.eval()

    # Confusion matrix
    conf_mat = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)
    refined_conf_mat = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)
    logit_conf_mat = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)

    # Test dataloader
    pbar = tqdm(total=len(dataset), ascii=True)
    for idx, data in enumerate(dataloader):
        pbar.update(1)
        execution_time = {}
        description = ""
        img = data["image"]
        label = data["label"].numpy()
        name = data["name"]
        total_time = time.time()
        if opt.slide_test is True:
            coarse_pred,logit_pred,middle_pred = slide_inference(model,refinement_models,opt.stride_size,img,opt.scales,opt.tmp,
                                                                 opt.num_classes,device,rescale=opt.rescale)
        else:
            coarse_pred = model(img.to(device))
            middle_pred = patches_inference(img.to(device),model)
            logit_pred = refinement_models[0](coarse_pred,middle_pred)
            coarse_pred = resize(
                input=coarse_pred,
                size=middle_pred.shape[2:],
                mode='bilinear',
                align_corners=False)
            logit_pred = resize(
                input=logit_pred,
                size=middle_pred.shape[2:],
                mode='bilinear',
                align_corners=False)
        certainty_mask = torch.where(middle_pred.argmax(1)!=coarse_pred.argmax(1),torch.ones_like(middle_pred.argmax(1)),
                                                                                                  torch.zeros_like(middle_pred.argmax(1)))
        certainty_mask_sum = torch.sum(certainty_mask.contiguous().view(-1)!=0)
        certainty_score = calculate_certainty(logit_pred.softmax(1))
        uncertainty_score = 1-calculate_certainty(coarse_pred.softmax(1))
        error_score = uncertainty_score*certainty_score*certainty_mask.unsqueeze(1)

        certainty_score_map = (certainty_mask.unsqueeze(1).float()*255)
        certainty_score_map =  F.interpolate(certainty_score_map, (label.shape[1],label.shape[2]), mode="bilinear",
                                             align_corners=False).cpu().numpy().astype('int')
        uncertainty_score_map = (uncertainty_score * 255)
        uncertainty_score_map = F.interpolate(uncertainty_score_map, (label.shape[1], label.shape[2]), mode="bilinear",
                                            align_corners=False).cpu().numpy().astype('int')
        error_score_map = (error_score*255)
        error_score_map = F.interpolate(error_score_map, (label.shape[1],label.shape[2]), mode="bilinear", align_corners=False).cpu().numpy().astype('int')
        del certainty_score, uncertainty_score

        if opt.n_points > 1.0:
            n_points = int(certainty_mask_sum/1.5)
        else:
            n_points = int(scale[0] * scale[1] * opt.n_points) / 2

        # Get point coordinates
        error_point_indices, error_point_coords = get_uncertain_point_coords_on_grid(error_score, n_points)
        del error_score

        error_point_indices = error_point_indices.unsqueeze(1).expand(-1, opt.num_classes, -1)

        # Get refinement prediction
        fine_pred = point_sample(logit_pred, error_point_coords, align_corners=False)
        # Replace points with new prediction
        final_output = coarse_pred.clone()
        final_output = (
            final_output.reshape(1, opt.num_classes, coarse_pred.shape[2] * coarse_pred.shape[3])
            .scatter_(2, error_point_indices, fine_pred)
            .view(1, opt.num_classes, coarse_pred.shape[2], coarse_pred.shape[3])
        )
        execution_time["time"] = time.time() - total_time
        # Compute IoU for coarse prediction
        coarse_pred = F.interpolate(coarse_pred, (label.shape[1],label.shape[2]), mode="bilinear", align_corners=False).argmax(1).cpu().numpy()
        mat = confusion_matrix(label, coarse_pred, opt.num_classes, ignore_label=dataset.ignore_label)
        conf_mat += mat
        description += "Coarse IoU: %.2f, " % (get_freq_iou(mat, opt.dataset) * 100)
        Coarse_IoU = (get_freq_iou(mat, opt.dataset) * 100)
        # Compute IoU for fine prediction
        final_output = (
            F.interpolate(middle_pred,(label.shape[1],label.shape[2]), mode="bilinear", align_corners=False).argmax(1).cpu().numpy()
        )
        mat = confusion_matrix(label, final_output, opt.num_classes, ignore_label=dataset.ignore_label)
        refined_conf_mat += mat
        description += "Refinement IoU: %.2f" % (get_freq_iou(mat, opt.dataset) * 100)
        Refinement_IoU = (get_freq_iou(mat, opt.dataset) * 100)
        fine_output = (
            F.interpolate(logit_pred, (label.shape[1],label.shape[2]), mode="bilinear", align_corners=False).argmax(
                1).cpu().numpy()
        )
        mat = confusion_matrix(label, fine_output, opt.num_classes, ignore_label=dataset.ignore_label)
        logit_conf_mat += mat
        description += "logit IoU: %.2f" % (get_freq_iou(mat, opt.dataset) * 100)
        logit_IoU = (get_freq_iou(mat, opt.dataset) * 100)
        if opt.save_pred:
            # Transform tensor to images
            img = dataset.inverse_transform(image_patches[0])
            img = np.array(to_pil_image(img))[:, :, ::-1]

            # Ignore label
            if dataset.ignore_label is not None:
                coarse_pred[label == dataset.ignore_label] = dataset.ignore_label
                final_output[label == dataset.ignore_label] = dataset.ignore_label

            # Convert predictions to images
            label = dataset.class2bgr(label[0])
            coarse_pred = dataset.class2bgr(coarse_pred[0])
            fine_pred = dataset.class2bgr(final_output[0])

            # Combine images, gt, predictions
            h = 512
            w = int((h * 1.0 / img.shape[0]) * img.shape[1])
            save_image = np.zeros((h, w * 4 + 10 * 3, 3), dtype=np.uint8)
            save_image[:, :, 2] = 255

            save_image[:, :w] = cv2.resize(img, (w, h))
            save_image[:, w + 10 : w * 2 + 10] = cv2.resize(label, (w, h))
            save_image[:, w * 2 + 20 : w * 3 + 20] = cv2.resize(coarse_pred, (w, h))
            save_image[:, w * 3 + 30 :] = cv2.resize(fine_pred, (w, h))

            # Save predictions
            os.makedirs(opt.save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(opt.save_dir, data["name"][0]), save_image)
        

        description += "".join([", %s: %.2f" % (k, v) for k, v in execution_time.items() if v > 0.01])
        pbar.set_description(description)

    pbar.write("-------SUMMARY-------")
    pbar.write("Coarse IoU: %.2f" % (get_overall_iou(conf_mat, opt.dataset) * 100))
    pbar.write("Refinement IoU: %.2f" % (get_overall_iou(refined_conf_mat, opt.dataset) * 100))
    pbar.write("logit IoU: %.2f" % (get_overall_iou(logit_conf_mat, opt.dataset) * 100))

if __name__ == "__main__":
    main()
