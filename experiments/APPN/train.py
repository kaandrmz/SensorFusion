import json
import os
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from appnet.dataset import get_dataset_with_name
from appnet.model import get_model_with_name
from appnet.model.refinement import RefinementAppNet
from appnet.options.train import TrainOptions
from appnet.utils.blur import GaussianBlur,MedianBlur
from appnet.utils.geometry import calculate_certainty, get_uncertain_point_coords_on_grid, point_sample
from appnet.utils.loss import (OhemCrossEntropy,RectifyCrossEntropy,IoUloss,ERRORLoss,DiceLoss,
                               CertaintyLoss,FocalLoss,LabelSmoothLoss,AsymmetricLossOptimized,
                                ClassLoss,PixelContrastLoss,)
from appnet.utils.metrics import confusion_matrix, get_freq_iou, get_overall_iou
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from appnet.utils.geometry import (
    get_patch_coords,
    merge,
)
from tqdm import tqdm
import torch.nn.functional as F
from mmseg.ops import resize


@torch.no_grad()
def patches_inference(crop_img,model):
    up_crop_img = F.interpolate(crop_img, scale_factor=2, mode="bilinear", align_corners=True)
    up_crop_img1 = up_crop_img[:, :, :crop_img.shape[2], :crop_img.shape[3]]
    up_crop_img2 = up_crop_img[:, :, crop_img.shape[2]:, crop_img.shape[3]:]
    up_crop_img3 = up_crop_img[:, :, crop_img.shape[2]:, :crop_img.shape[3]]
    up_crop_img4 = up_crop_img[:, :, :crop_img.shape[2], crop_img.shape[3]:]
    up_crop_imgs = torch.cat([up_crop_img1.unsqueeze(0), up_crop_img2.unsqueeze(0), up_crop_img3.unsqueeze(0), up_crop_img4.unsqueeze(0)], dim=0)
    preds = []
    for i in range(4):
        pred = model(up_crop_imgs[i])
        preds += [pred]
    middle_map = F.interpolate(torch.zeros_like(pred), scale_factor=2, mode="bilinear",align_corners=False)
    middle_map[:, :, :pred.shape[2], :pred.shape[3]] = preds[0]
    middle_map[:, :, pred.shape[2]:, pred.shape[3]:] = preds[1]
    middle_map[:, :, pred.shape[2]:, :pred.shape[3]] = preds[2]
    middle_map[:, :, :pred.shape[2], pred.shape[3]:] = preds[3]
    return middle_map



def main():
    # Parse arguments
    opt = TrainOptions().parse()  # opt: shurucanshu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("gpu")

    # Create logger
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(opt.log_dir, opt.task_name, date_time)
    writer = SummaryWriter(logdir=log_dir)

    # Save config
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(opt), indent=4))
        
    # Create dataset
    dataset = get_dataset_with_name(opt.dataset)(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
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
        model.load_state_dict(state_dict=state_dict, strict=False)
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

    # gaussianblur = MedianBlur(kernel_size=(opt.smooth_kernel, opt.smooth_kernel)).to(device)
    gaussianblur = GaussianBlur(3,kernel_size=(opt.smooth_kernel, opt.smooth_kernel),sigma=(1,1)).to(device)
    gaussianblur.eval()

    model.eval()
    model = torch.nn.DataParallel(model)
    for scale in opt.scales[1:]:
        patch_coords = torch.tensor(get_patch_coords(scale, opt.crop_size)).to(device)
    # Create refinement module
    refinement_model = RefinementAppNet(opt.num_classes, opt.input_size, use_bn=True).to(device)
    if os.path.isfile(opt.pretrained_refinement):
        print("Load refinement weight from", opt.pretrained_refinement)
        state_dict = torch.load(opt.pretrained_refinement)
        refinement_model.load_state_dict(state_dict, strict=False)


    # Create optimizer
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, refinement_model.parameters()),
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.decay,
    )

    

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)

    # Loss function
    criteria = OhemCrossEntropy(ignore_label=dataset.ignore_label)
    class_intra_loss = PixelContrastLoss(ignorelabel= dataset.ignore_label)
    edeloss = nn.CrossEntropyLoss()
    aug_criteria = OhemCrossEntropy(ignore_label=dataset.ignore_label)
    IoUcriteria = IoUloss(ignore_label=dataset.ignore_label)
    errorcriteria = OhemCrossEntropy(ignore_label=dataset.ignore_label)
    maskcriteria = DiceLoss(ignore_label=dataset.ignore_label)
    certaintyloss = CertaintyLoss(ignore_label=dataset.ignore_label)
    fcoalloss = FocalLoss(ignore_label=dataset.ignore_label,gamma=0)
    smothcriteria = LabelSmoothLoss(ignore_index=dataset.ignore_label)
    classcriteria = ClassLoss()
    class_criteria = AsymmetricLossOptimized()
    global_step = 0
    best_mIoU = 0
    for epoch in range(opt.epochs):

        # Training
        refinement_model.train()

        pbar = tqdm(total=len(dataloader))

        # Metrics
        epoch_mat_coarse = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)
        epoch_mat_fine = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)
        epoch_mat_aggre = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)
        epoch_mat_logits = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)
        epoch_mask_mat = np.zeros((2, 2), dtype=np.float)
        mean_loss = []
        ori_mean_loss = []
        refine_mean_loss = []
        co = []
        fi = []
        for idx, data in enumerate(dataloader):
            
            image = data["image"].to(device)
            
            label = data["label"].to(device)
            label_edge = data["label_edge"].to(device)
            label_edge[label_edge != 0] = 1
            
            L_image = F.interpolate(image,scale_factor=2,mode="bilinear",align_corners=True)
            
            with torch.no_grad():
                coarse_pred = model(image)
                middle_pre = patches_inference(image,model)
                
            crop_preds = coarse_pred
            
            # Refinement forward
            optimizer.zero_grad()

            logits,augout,aug = refinement_model(coarse_pred,middle_pre, aug=True)
            if epoch > 80:
                loss = criteria(aug,label,mask)+edeloss(logits,label_edge) 
            else:
                loss = criteria(augout, label,ohem=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss1 = criteria(coarse_pred, label,ohem=True)
            description = "loss: %.2f, " % (loss)
            description += "ori_loss: %.2f, " % (loss1)
            mean_loss += [float(loss)]
            refine_mean_loss += [float(loss)]
            ori_mean_loss += [float(loss1)]

            writer.add_scalar("step_loss", loss, global_step)
            writer.add_scalar("ori_step_loss", loss1, global_step)

            description += "lr: " + str(optimizer.param_groups[0]["lr"]) + ", "

            crop_preds = F.interpolate(crop_preds, (label.size(1), label.size(2)),
                                       mode="bilinear", align_corners=False)
            fine_pred = F.interpolate(augout, (label.size(1), label.size(2)),
                                      mode="bilinear", align_corners=True)
            logits = F.interpolate(aug, (label.size(1), label.size(2)),
                                   mode="bilinear", align_corners=False)
            fine_label = label.cpu().numpy()
            coarse_mat = confusion_matrix(fine_label, crop_preds.argmax(1).cpu().numpy(), opt.num_classes,
                                          ignore_label=dataset.ignore_label)
            epoch_mat_coarse += coarse_mat
            logits_mat = confusion_matrix(fine_label, logits.argmax(1).cpu().numpy(), opt.num_classes,
                                          ignore_label=dataset.ignore_label)
            epoch_mat_logits += logits_mat
            
            aggre_mat = confusion_matrix(
                fine_label, fine_pred.argmax(1).cpu().numpy(), opt.num_classes, ignore_label=dataset.ignore_label
            )
            epoch_mat_aggre += aggre_mat
            IoU_coarse = get_freq_iou(coarse_mat, opt.dataset)
            description += "IoU coarse: %.2f, " % (IoU_coarse * 100)
            co.append(IoU_coarse)


            IoU_aggre = get_freq_iou(aggre_mat, opt.dataset)
            description += "IoU aggre: %.2f," % (IoU_aggre * 100)
            IoU_logits = get_freq_iou(logits_mat, opt.dataset)
            description += "IoU logits: %.2f" % (IoU_logits * 100)

            writer.add_scalars("step_IoU", {"coarse": IoU_coarse, "aggre": IoU_aggre,"logits": IoU_logits},
                               global_step)
            description = "Epoch {}/{}: ".format(epoch + 1, opt.epochs) + description

            pbar.set_description(description)
            pbar.update(1)
            global_step += 1

        # Update learning rate
        lr_scheduler.step()
        writer.add_scalar("epoch_loss", sum(mean_loss) / len(mean_loss), global_step=epoch + 1)
        writer.add_scalar("ori_epoch_loss", sum(ori_mean_loss) / len(ori_mean_loss), global_step=epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=epoch + 1)
        writer.add_scalars(
            "epoch_IoU",
            {
                "coarse": get_overall_iou(epoch_mat_coarse, opt.dataset),
                "aggre": get_overall_iou(epoch_mat_aggre, opt.dataset),
                'logits': get_overall_iou(epoch_mat_logits, opt.dataset),
            },
            global_step=epoch + 1,
        )
        mean_Iou_coarse = get_overall_iou(epoch_mat_coarse, opt.dataset)
        print("coarse_IOU {0}:".format(get_overall_iou(epoch_mat_coarse, opt.dataset)))
        print("logits_IOU {0}:".format(get_overall_iou(epoch_mat_logits, opt.dataset)))

        mean_Iou_refine = get_overall_iou(epoch_mat_aggre, opt.dataset)
        print("refine_IOU {0}:".format(mean_Iou_refine))
        if mean_Iou_refine-mean_Iou_coarse > best_mIoU:
            best_mIoU = mean_Iou_refine - mean_Iou_coarse
            print("best mIoU change to {},mean_loss change to {}".format(mean_Iou_refine,
                                                                         sum(refine_mean_loss) / len(refine_mean_loss)))
            with open(os.path.join(log_dir, "best.json"), "w") as f:
                f.write("best mIoU change to {},mean_loss change to {},epoch is {}".format(mean_Iou_refine,
                                                                         best_mIoU,epoch+1))
            torch.save(refinement_model.state_dict(), os.path.join(log_dir, "best.pth"))
        # Save model
        torch.save(refinement_model.state_dict(), os.path.join(log_dir, "epoch{}.pth".format(epoch + 1)))


if __name__ == "__main__":
    main()
