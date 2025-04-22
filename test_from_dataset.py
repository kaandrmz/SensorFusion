import os
import numpy as np
from PIL import Image
import cv2
import clip
import torch
from torchvision.transforms import functional as F
from model.Text_IF_model import Text_IF as create_model
import argparse
from utils.Evaluator import Evaluator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main(args):
    # Set GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    root_path = args.dataset_path
    save_path = args.save_path
    dataset_type = args.dataset_type
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']
    text_line = args.input_text

    visible_root = os.path.join(root_path, "Visible")
    infrared_root = os.path.join(root_path, "Infrared")

    visible_path = [os.path.join(visible_root, i) for i in os.listdir(visible_root)
                  if os.path.splitext(i)[-1] in supported]
    infrared_path = [os.path.join(infrared_root, i) for i in os.listdir(infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    visible_path.sort()
    infrared_path.sort()

    print("Find the number of visible image: {},  the number of the infrared image: {}".format(len(visible_path), len(infrared_path)))
    assert len(visible_path) == len(infrared_path), "The number of the source images does not match!"

    print("Begin to run!")
    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device=device)
        model = create_model(model_clip).to(device)

        model_weight_path = args.weights_path
        model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
        model.eval()

    # Set up metrics based on dataset type
    if dataset_type == "EMS":
        print("Using EMS dataset metrics: EN, SD, SF, MI, SCD, VIFF, Qabf, SSIM")
        metric_names = ['EN', 'SD', 'SF', 'MI', 'SCD', 'VIFF', 'Qabf', 'SSIM']
        num_metrics = len(metric_names)
    elif dataset_type == "MSRS":
        print("Using MSRS dataset metrics: CLIP-IQA, EN, NIQE")
        metric_names = ['CLIP-IQA', 'EN', 'NIQE']
        num_metrics = len(metric_names)
    elif dataset_type == "LLVIP":
        print("Using LLVIP dataset metrics: EN, NIQE, MUSIQ")
        metric_names = ['EN', 'NIQE', 'MUSIQ']
        num_metrics = len(metric_names)
    elif dataset_type == "MFNET":
        print("Using MFNET dataset metrics: SD, EN, MUSIQ")
        metric_names = ['SD', 'EN', 'MUSIQ']
        num_metrics = len(metric_names)
    elif dataset_type == "all":
        print("Using all dataset metrics: EN, SD, SF, MI, SCD, VIFF, Qabf, SSIM, NIQE, MUSIQ, BRISQUE, CLIP-IQA")
        metric_names = ['EN', 'SD', 'SF', 'MI', 'SCD', 'VIFF', 'Qabf', 'SSIM', 'NIQE', 'MUSIQ', 'BRISQUE', 'CLIP-IQA']
        num_metrics = len(metric_names)
    elif dataset_type == "rest":
        print("Using rest dataset metrics: NIQE, MUSIQ, BRISQUE, CLIP-IQA")
        metric_names = ['NIQE', 'MUSIQ', 'BRISQUE', 'CLIP-IQA']
        num_metrics = len(metric_names)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Must be 'EMS' or 'MSRS' or 'LLVIP' or 'MFNET' or 'all' or 'rest'")

    metric_result = np.zeros((num_metrics))
    num_images = len(visible_path)

    for i in range(len(visible_path)):
        ir_path = infrared_path[i]
        vi_path = visible_path[i]

        img_name = vi_path.replace("\\", "/").split("/")[-1]
        assert os.path.exists(ir_path), "file: '{}' dose not exist.".format(ir_path)
        assert os.path.exists(vi_path), "file: '{}' dose not exist.".format(vi_path)

        ir = Image.open(ir_path).convert(mode="RGB")
        vi = Image.open(vi_path).convert(mode="RGB")

        height, width = vi.size
        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        ir = ir.resize((new_height, new_width))
        vi = vi.resize((new_height, new_width))
        ir = F.to_tensor(ir)
        vi = F.to_tensor(vi)

        ir = ir.unsqueeze(0).cuda()
        vi = vi.unsqueeze(0).cuda()
        with torch.no_grad():
            text = clip.tokenize(text_line).to(device)
            fused, _ = model(vi, ir, text) 
            # fused = model(vi, ir, text) # original  
            fused_img_Y = tensor2numpy(fused)
            
            save_pic(fused_img_Y, save_path, img_name)

        # Convert images to grayscale format for evaluation
        ir_gray = cv2.cvtColor(tensor2numpy(ir), cv2.COLOR_RGB2GRAY)
        vi_gray = cv2.cvtColor(tensor2numpy(vi), cv2.COLOR_RGB2GRAY)
        fused_gray = cv2.cvtColor(fused_img_Y, cv2.COLOR_RGB2GRAY)

        # Calculate metrics based on dataset type
        if dataset_type == "EMS":
            metrics = np.array([
                Evaluator.EN(fused_gray),
                Evaluator.SD(fused_gray),
                Evaluator.SF(fused_gray),
                Evaluator.MI(fused_gray, ir_gray, vi_gray),
                Evaluator.SCD(fused_gray, ir_gray, vi_gray),
                Evaluator.VIFF(fused_gray, ir_gray, vi_gray),
                Evaluator.Qabf(fused_gray, ir_gray, vi_gray),
                Evaluator.SSIM(fused_gray, ir_gray, vi_gray)
            ])
        elif dataset_type == "MSRS":  # MSRS
            metrics = np.array([
                Evaluator.CLIP_IQA(fused_img_Y),
                Evaluator.EN(fused_gray),
                Evaluator.NIQE(fused_gray)
            ])
        elif dataset_type == "LLVIP":
            metrics = np.array([
                Evaluator.EN(fused_gray),
                Evaluator.NIQE(fused_gray),
                Evaluator.MUSIQ(fused_img_Y)
            ])
        elif dataset_type == "MFNET":
            metrics = np.array([
                Evaluator.SD(fused_gray),
                Evaluator.EN(fused_gray),
                Evaluator.MUSIQ(fused_img_Y)
            ])
        elif dataset_type == "all":
            metrics = np.array([
                Evaluator.EN(fused_gray),
                Evaluator.SD(fused_gray),
                Evaluator.SF(fused_gray),
                Evaluator.MI(fused_gray, ir_gray, vi_gray),
                Evaluator.SCD(fused_gray, ir_gray, vi_gray),
                Evaluator.VIFF(fused_gray, ir_gray, vi_gray),
                Evaluator.Qabf(fused_gray, ir_gray, vi_gray),
                Evaluator.SSIM(fused_gray, ir_gray, vi_gray),
                Evaluator.NIQE(fused_gray),
                Evaluator.MUSIQ(fused_img_Y),
                Evaluator.BRISQUE(fused_img_Y),
                Evaluator.CLIP_IQA(fused_img_Y)
            ])
        elif dataset_type == "rest":
            metrics = np.array([
                Evaluator.NIQE(fused_gray),
                Evaluator.MUSIQ(fused_img_Y),
                Evaluator.BRISQUE(fused_img_Y),
                Evaluator.CLIP_IQA(fused_img_Y)
            ])
        metric_result += metrics

        print(f"Processed {img_name} - Current metrics:", 
              {name: f"{val:.3f}" for name, val in zip(metric_names, metrics)})

    # Calculate and print average metrics
    metric_result /= num_images
    print("\nFinal Average Results:")
    if dataset_type == "EMS":
        print("EN\t SD\t SF\t MI\tSCD\tVIFF\tQabf\tSSIM")
    elif dataset_type == "MSRS":
        print("CLIP-IQA\tEN\tNIQE")
    elif dataset_type == "LLVIP":
        print("EN\tNIQE\tMUSIQ")
    elif dataset_type == "MFNET":
        print("SD\tEN\tMUSIQ")
    elif dataset_type == "all":
        print("EN\tSD\tSF\tMI\tSCD\tVIFF\tQabf\tSSIM\tNIQE\tMUSIQ\tBRISQUE\tCLIP-IQA")
    elif dataset_type == "rest":
        print("NIQE\tMUSIQ\tBRISQUE\tCLIP-IQA")
    print('\t'.join(f"{val:.3f}" for val in metric_result))

    print("Finish! The results are saved in {}.".format(save_path))

def tensor2numpy(img_tensor):
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img

def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).detach().cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])
    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])
    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img

def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index).replace(".jpg", ".png")
    cv2.imwrite(save_path, outputpic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='test data root path')
    parser.add_argument('--weights_path', type=str, required=True, help='initial weights path')
    parser.add_argument('--save_path', type=str, default='./results', help='output save image path')
    parser.add_argument('--input_text', type=str, required=True, help='text control input')
    parser.add_argument('--dataset_type', type=str, default='EMS', choices=['EMS', 'MSRS', 'LLVIP', 'MFNET', 'all', 'rest'], help='dataset type to determine which metrics to use')

    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='5', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()
    main(opt)