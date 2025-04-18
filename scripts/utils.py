import os
import sys
import random
import clip
import torch.nn.functional as F

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import cv2

from scripts.losses import fusion_prompt_loss

low_light_prompt_path = "./SensorFusion/dataset/EMS_dataset/vis_Low_Light/train/text.txt"
assert os.path.exists(low_light_prompt_path), "text prompt root: {} does not exist.".format(low_light_prompt_path)
with open(low_light_prompt_path, 'r', encoding='utf-8') as file:
    low_light_lines = file.readlines()

over_exposure_prompt_path = "./SensorFusion/dataset/EMS_dataset/vis_Exposure/train/text.txt"
assert os.path.exists(over_exposure_prompt_path), "text prompt root: {} does not exist.".format(over_exposure_prompt_path)
with open(over_exposure_prompt_path, 'r', encoding='utf-8') as file:
    over_exposure_lines = file.readlines()

ir_low_contrast_prompt_path = "./SensorFusion/dataset/EMS_dataset/IR_Low_Contrast/train/text.txt"
assert os.path.exists(ir_low_contrast_prompt_path), "text prompt root: {} does not exist.".format(ir_low_contrast_prompt_path)
with open(ir_low_contrast_prompt_path, 'r', encoding='utf-8') as file:
    ir_low_contrast_lines = file.readlines()

ir_noise_prompt_path = "./SensorFusion/dataset/EMS_dataset/IR_Random_noise/train/text.txt"
assert os.path.exists(ir_noise_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_prompt_path)
with open(ir_noise_prompt_path, 'r', encoding='utf-8') as file:
    ir_noise_lines = file.readlines()

ir_stripe_noise_prompt_path = "./SensorFusion/dataset/EMS_dataset/IR_Stripe_noise/train/text.txt"
assert os.path.exists(ir_stripe_noise_prompt_path), "text prompt root: {} does not exist.".format(ir_stripe_noise_prompt_path)
with open(ir_stripe_noise_prompt_path, 'r', encoding='utf-8') as file:
    ir_stripe_noise_lines = file.readlines()

vis_blur_prompt_path = "./SensorFusion/dataset/EMS_dataset/vis_Blur/train/text.txt"
assert os.path.exists(vis_blur_prompt_path), "text prompt root: {} does not exist.".format(vis_blur_prompt_path)
with open(vis_blur_prompt_path, 'r', encoding='utf-8') as file:
    vis_blur_lines = file.readlines()

vis_haze_prompt_path = "./SensorFusion/dataset/EMS_dataset/vis_Haze/train/text.txt"
assert os.path.exists(vis_haze_prompt_path), "text prompt root: {} does not exist.".format(vis_haze_prompt_path)
with open(vis_haze_prompt_path, 'r', encoding='utf-8') as file:
    vis_haze_lines = file.readlines()

vis_rain_prompt_path = "./SensorFusion/dataset/EMS_dataset/vis_Rain/train/text.txt"
assert os.path.exists(vis_rain_prompt_path), "text prompt root: {} does not exist.".format(vis_rain_prompt_path)
with open(vis_rain_prompt_path, 'r', encoding='utf-8') as file:
    vis_rain_lines = file.readlines()

vis_random_noise_prompt_path = "./SensorFusion/dataset/EMS_dataset/vis_Random_noise/train/text.txt"
assert os.path.exists(vis_random_noise_prompt_path), "text prompt root: {} does not exist.".format(vis_random_noise_prompt_path)
with open(vis_random_noise_prompt_path, 'r', encoding='utf-8') as file:
    vis_random_noise_lines = file.readlines()

def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "eval")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    train_images_visible_path = []
    train_images_infrared_path = []
    train_images_visible_gt_path = []
    train_images_infrared_gt_path = []
    val_images_visible_path = []
    val_images_infrared_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF'] 

    train_visible_root = os.path.join(train_root, "Visible")
    train_infrared_root= os.path.join(train_root, "Infrared")

    train_visible_gt_root = os.path.join(train_root, "Visible_gt")
    train_infrared_gt_root= os.path.join(train_root, "Infrared_gt")

    val_visible_root = os.path.join(val_root, "Visible")
    val_infrared_root = os.path.join(val_root, "Infrared")

    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_gt_path = [os.path.join(train_visible_gt_root, i) for i in os.listdir(train_visible_gt_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_gt_path = [os.path.join(train_infrared_gt_root, i) for i in os.listdir(train_infrared_gt_root)
                  if os.path.splitext(i)[-1] in supported]

    val_visible_path = [os.path.join(val_visible_root, i) for i in os.listdir(val_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    val_infrared_path = [os.path.join(val_infrared_root, i) for i in os.listdir(val_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_path.sort()
    train_infrared_path.sort()
    train_visible_gt_path.sort()
    train_infrared_gt_path.sort()
    val_visible_path.sort()
    val_infrared_path.sort()

    assert len(train_visible_path) == len(train_infrared_path),' The length of train dataset does not match. low:{}, high:{}'.\
                                         format(len(train_visible_path),len(train_infrared_path))
    assert len(val_visible_path) == len(val_infrared_path),' The length of val dataset does not match. low:{}, high:{}'.\
                                          format(len(val_visible_path),len(val_infrared_path))
    print("Visible and Infrared images check finish")

    for index in range(len(train_visible_path)):
        img_visible_path=train_visible_path[index]
        img_infrared_path=train_infrared_path[index]
        train_images_visible_path.append(img_visible_path)
        train_images_infrared_path.append(img_infrared_path)

        img_visible_gt_path=train_visible_gt_path[index]
        img_infrared_gt_path=train_infrared_gt_path[index]
        train_images_visible_gt_path.append(img_visible_gt_path)
        train_images_infrared_gt_path.append(img_infrared_gt_path)

    for index in range(len(val_visible_path)):
        img_visible_path=val_visible_path[index]
        img_infrared_path=val_infrared_path[index]
        val_images_visible_path.append(img_visible_path)
        val_images_infrared_path.append(img_infrared_path)

    total_dataset_nums = len(train_visible_path) + len(train_infrared_path) + len(train_visible_gt_path) + len(train_infrared_gt_path) \
                         + len(val_visible_path) + len(val_infrared_path)
    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} visible images for training.".format(len(train_visible_path)))
    print("{} infrared images for training.".format(len(train_infrared_path)))
    print("{} visible gt images for training.".format(len(train_visible_gt_path)))
    print("{} infrared gt images for training.".format(len(train_infrared_gt_path)))
    print("{} visible images for validation.".format(len(val_visible_path)))
    print("{} infrared images for validation.\n".format(len(val_infrared_path)))

    train_low_light_path_list = [train_visible_path, train_infrared_path, train_visible_gt_path, train_infrared_gt_path]
    val_low_light_path_list = [val_visible_path, val_infrared_path]
    return train_low_light_path_list, val_low_light_path_list

def get_low_light_prompt():
    return random.choice(low_light_lines).strip()

def get_over_exposure_prompt():
    return random.choice(over_exposure_lines).strip()

def get_ir_low_contrast_prompt():
    return random.choice(ir_low_contrast_lines).strip()

def get_ir_noise_prompt():
    return random.choice(ir_noise_lines).strip()

def get_ir_stripe_noise_prompt():
    return random.choice(ir_stripe_noise_lines).strip()

def get_vis_blur_prompt():
    return random.choice(vis_blur_lines).strip()

def get_vis_haze_prompt():
    return random.choice(vis_haze_lines).strip()

def get_vis_rain_prompt():
    return random.choice(vis_rain_lines).strip()

def get_vis_random_noise_prompt():
    return random.choice(vis_random_noise_lines).strip()

def train_one_epoch(model, model_clip, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    model_clip.eval()
    loss_function_prompt = fusion_prompt_loss()

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)

    ##################################################################################
    # # 4 times score maps
    # def compute_prompt_loss(image_A, image_B, score_map1, score_map2, score_map3, score_map4, task, text_features=None, temperature=0.07):
    #     """
    #     Compute prompt loss by treating score_map as a low-res fusion result
    #     Args:
    #         image_A: tensor of shape [B, C, H, W] (visible image)
    #         image_B: tensor of shape [B, C, H, W] (infrared image)
    #         score_map: tensor of shape [B, C, H, W] (score map)
    #         task: list of task types
    #         temperature: temperature coefficient for scaling
    #     Returns:
    #         loss: scalar tensor
    #     """
    #     print("Shapes before processing:")
    #     print(f"image_A: {image_A.shape}")
    #     print(f"image_B: {image_B.shape}")
    #     print(f"score_map1: {score_map1.shape}")
    #     print(f"text_features: {text_features.shape if text_features is not None else None}")
    #     losses=[]
    #     weights = [0.4, 0.3, 0.2, 0.1]
    #     window_size = 48  # Starting window size
    #     min_window_size = 3  # Prevent window becoming too small
        
    #     # Debug NaN/Inf values
    #     for i, score_map in enumerate([score_map1, score_map2, score_map3, score_map4]):
    #         if torch.isnan(score_map).any() or torch.isinf(score_map).any():
    #             print(f"WARNING: score_map{i+1} contains NaN or Inf values")
    #             score_map = torch.nan_to_num(score_map, nan=0.0, posinf=1.0, neginf=0.0)
        
    #     # Scale and normalize score_map
    #     for score_map, weight in zip([score_map1, score_map2, score_map3, score_map4], weights):
    #         score_map = score_map / temperature
    #         # score_map = torch.sigmoid(score_map)
            
    #         # Get score_map dimensions
    #         h, w = score_map.shape[2:]
            
    #         # Resize input images to match score_map resolution
    #         image_A_small = F.interpolate(image_A, size=(h, w), mode='bilinear', align_corners=False)
    #         image_B_small = F.interpolate(image_B, size=(h, w), mode='bilinear', align_corners=False)
            
    #         # Create a new loss function with the current window size
    #         loss_function_with_window = fusion_prompt_loss(window_size=window_size)
    #         if torch.cuda.is_available():
    #             loss_function_with_window = loss_function_with_window.to(device)
            
    #         # Calculate all fusion losses using the score_map as the "fused" image
    #         # loss, ssim_loss, max_loss, color_loss, grad_loss = loss_function_with_window(
    #         #     image_A_small, 
    #         #     image_B_small, 
    #         #     score_map,
    #         #     task
    #         # )
    #         loss, ssim_loss, max_loss, color_loss, grad_loss = loss_function_with_window(
    #             image_A_small, 
    #             image_B_small, 
    #             score_map,
    #             task,
    #             text_features
    #         )
    #         losses.append(weight*loss)
            
    #         # Halve the window size for next iteration
    #         window_size = max(window_size // 2, min_window_size)  # Prevent too small windows
        
    #     return sum(losses)
    ##################################################################################


    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)

    # accu_original_loss = torch.zeros(1).to(device)
    # accu_ptm_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, _, task, _ = data
        text_line = []

        for index in range(len(task)):
        # default type degradation in vis image
            if task[index] == "low_light":
                text_line.append(get_low_light_prompt())
            elif task[index] == "over_exposure":
                text_line.append(get_over_exposure_prompt())
            elif task[index] == "ir_low_contrast":
                text_line.append(get_ir_low_contrast_prompt())
            elif task[index] == "ir_noise":
                text_line.append(get_ir_noise_prompt())
            elif task[index] == "ir_stripe_noise":
                text_line.append(get_ir_stripe_noise_prompt())
            elif task[index] == "vis_blur":
                text_line.append(get_vis_blur_prompt())
            elif task[index] == "vis_haze":
                text_line.append(get_vis_haze_prompt())
            elif task[index] == "vis_rain":
                text_line.append(get_vis_rain_prompt())
            elif task[index] == "vis_random_noise":
                text_line.append(get_vis_random_noise_prompt())
            else:
                text_line.append("This is unknown to the image fusion task.")
        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)

        
        # I_fused = model(I_A, I_B, text) # For original
        I_fused, text_features = model(I_A, I_B, text)
        # I_fused, text_features, score_map1, score_map2, score_map3, score_map4 = model(I_A, I_B, text)

        # loss_original, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task)
        # loss_original, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task, text_features)
        # loss_ptm = compute_prompt_loss(I_A_gt, I_B_gt, score_map1, score_map2, score_map3, score_map4, task, text_features)
        # loss = loss_ptm + loss_original

        # loss, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task) # original	
        loss, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task, text_features) 
        loss.backward()

        accu_total_loss += loss.detach()
        # accu_original_loss += loss_original.detach() # original loss, added for denseclip 
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text.detach()
        # accu_ptm_loss += loss_ptm.detach()

        lr = optimizer.param_groups[0]["lr"]

        #original
        data_loader.desc = "[train epoch {}] loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), lr)
        
        # # add ptm loss
        # data_loader.desc = "[train epoch {}] loss: {:.3f} original loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f} ptm loss: {:.3f} lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
        #     accu_original_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), accu_ptm_loss.item() / (step + 1), lr)
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), lr
    # return accu_total_loss.item() / (step + 1), accu_original_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), accu_ptm_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lr, filefold_path):
    loss_function_prompt = fusion_prompt_loss()

    model.eval()
    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)

    # accu_original_loss = torch.zeros(1).to(device)
    # accu_ptm_loss = torch.zeros(1).to(device)

    save_epoch = 1
    save_length = 60
    cnt = 0
    save_RGB_fuse = True

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)
    
    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    ###########################################################################
    # # 4 times score maps
    # def compute_prompt_loss(image_A, image_B, score_map1, score_map2, score_map3, score_map4, task, text_features=None, temperature=0.07):
    #     """
    #     Compute prompt loss by treating score_map as a low-res fusion result
    #     Args:
    #         image_A: tensor of shape [B, C, H, W] (visible image)
    #         image_B: tensor of shape [B, C, H, W] (infrared image)
    #         score_map: tensor of shape [B, C, H, W] (score map)
    #         task: list of task types
    #         temperature: temperature coefficient for scaling
    #     Returns:
    #         loss: scalar tensor
    #     """
    #     print("Shapes before processing:")
    #     print(f"image_A: {image_A.shape}")
    #     print(f"image_B: {image_B.shape}")
    #     print(f"score_map1: {score_map1.shape}")
    #     print(f"text_features: {text_features.shape if text_features is not None else None}")
    #     losses=[]
    #     weights = [0.4, 0.3, 0.2, 0.1]
    #     window_size = 48  # Starting window size
    #     min_window_size = 3  # Prevent window becoming too small
        
    #     # Debug NaN/Inf values
    #     for i, score_map in enumerate([score_map1, score_map2, score_map3, score_map4]):
    #         if torch.isnan(score_map).any() or torch.isinf(score_map).any():
    #             print(f"WARNING: score_map{i+1} contains NaN or Inf values")
    #             score_map = torch.nan_to_num(score_map, nan=0.0, posinf=1.0, neginf=0.0)
        
    #     # Scale and normalize score_map
    #     for score_map, weight in zip([score_map1, score_map2, score_map3, score_map4], weights):
    #         score_map = score_map / temperature
    #         # score_map = torch.sigmoid(score_map)
            
    #         # Get score_map dimensions
    #         h, w = score_map.shape[2:]
            
    #         # Resize input images to match score_map resolution
    #         image_A_small = F.interpolate(image_A, size=(h, w), mode='bilinear', align_corners=False)
    #         image_B_small = F.interpolate(image_B, size=(h, w), mode='bilinear', align_corners=False)
            
    #         # Create a new loss function with the current window size
    #         loss_function_with_window = fusion_prompt_loss(window_size=window_size)
    #         if torch.cuda.is_available():
    #             loss_function_with_window = loss_function_with_window.to(device)
            
    #         # Calculate all fusion losses using the score_map as the "fused" image
    #         # loss, ssim_loss, max_loss, color_loss, grad_loss = loss_function_with_window(
    #         #     image_A_small, 
    #         #     image_B_small, 
    #         #     score_map,
    #         #     task
    #         # )
    #         loss, ssim_loss, max_loss, color_loss, grad_loss = loss_function_with_window(
    #             image_A_small, 
    #             image_B_small, 
    #             score_map,
    #             task,
    #             text_features
    #         )
    #         losses.append(weight*loss)
            
    #         # Halve the window size for next iteration
    #         window_size = max(window_size // 2, min_window_size)  # Prevent too small windows
        
    #     return sum(losses) * 0.5 # to make it more comparable with original loss
    ##################################################################################


    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name = data
        text_line = []
        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the low light degradation.")
            elif task[index] == "over_exposure":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the overexposure degradation.")
            elif task[index] == "ir_low_contrast":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the low contrast degradation.")
            elif task[index] == "ir_noise":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the noise degradation.")
            elif task[index] == "ir_stripe_noise":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have stripe noise degradation.")
            elif task[index] == "vis_blur":
                text_line.append("This is the infrared-visible light fusion task. Visible images have blur degradation.")
            elif task[index] == "vis_haze":
                text_line.append("This is the infrared-visible light fusion task. Visible images have haze degradation.")
            elif task[index] == "vis_rain":
                text_line.append("This is the infrared-visible light fusion task. Visible images have rain degradation.")
            elif task[index] == "vis_random_noise":
                text_line.append("This is the infrared-visible light fusion task. Visible images have random noise degradation.")
            else:
                text_line.append("This is unknown to the image fusion task.")

        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_full = I_full.to(device)

        # I_fused = model(I_A, I_B, text) # for original
        I_fused, text_features = model(I_A, I_B, text) # for weight embeddings
        # I_fused, score_map1, score_map2, score_map3, score_map4 = model(I_A, I_B, text)
        # I_fused, text_features, score_map1, score_map2, score_map3, score_map4 = model(I_A, I_B, text) # for weight embeddings and denseclip



        if epoch % save_epoch == 0:
            if cnt <= save_length:
                # fused_img_Y = tensor2numpy(I_fused_rectified)
                fused_img_Y = tensor2numpy(I_fused)
                img_full = tensor2numpy(I_full)
                img_ir = tensor2numpy(I_B_gt)
                save_pic(fused_img_Y, evalfold_path, str(name[0]))
                if save_RGB_fuse == True:
                    save_pic(img_full, evalfold_path, str(name[0]) + "vis")
                    save_pic(img_ir, evalfold_path, str(name[0]) + "ir")
                cnt += 1
        
        # loss_original, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task) # denseclip
        # loss_ptm = compute_prompt_loss(I_A_gt, I_B_gt, score_map1, score_map2, score_map3, score_map4, task)

        # loss_original, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task, text_features)
        # loss_ptm = compute_prompt_loss(I_A_gt, I_B_gt, score_map1, score_map2, score_map3, score_map4, task, text_features)

        # loss = loss_ptm + loss_original

        # loss, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task) # original
        loss, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task, text_features) # for weight embeddings        
        
        accu_total_loss += loss

        # accu_original_loss += loss_original.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text
        # accu_ptm_loss += loss_ptm.detach()

        # original
        data_loader.desc = "[val epoch {}] loss: {:.3f} ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), lr)
        
        # # denseclip
        # data_loader.desc = "[val epoch {}] loss: {:.3f} original loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f}  ptm loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
        #     accu_original_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), accu_ptm_loss.item() / (step + 1), lr)

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1)
    # return accu_total_loss.item() / (step + 1), accu_original_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), accu_ptm_loss.item() / (step + 1)

def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])

    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])

    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index + ".png")
    cv2.imwrite(save_path, outputpic)

def show_img(images,imagesl, B):
    for index in range(B):
        img = images[index, :]
        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(1)
        plt.imshow(img_np)
        img = imagesl[index, :]

        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(2)
        plt.imshow(img_np)
        plt.show(block=True)

def tensor2numpy(R_tensor):
    R = R_tensor.squeeze(0).cpu().detach().numpy()
    R = np.transpose(R, [1, 2, 0])
    return R

def tensor2numpy_single(L_tensor):
    L = L_tensor.squeeze(0)
    L_3 = torch.cat([L, L, L], dim=0)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3