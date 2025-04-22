import os
import argparse
import json
import random

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import clip
from data.prompt_dataset import PromptDataSet

from model.Text_IF_model import Text_IF as create_model
from scripts.utils import read_data, train_one_epoch, evaluate, create_lr_scheduler
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import transforms as T

def sample_dataset(path_list, sample_ratio=0.3, seed=42):
    random.seed(seed)
    total = len(path_list)
    sample_size = int(total * sample_ratio)
    indices = random.sample(range(total), sample_size)
    return [path_list[i] for i in indices]

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "experiments." + file_name
    os.makedirs(filefold_path, exist_ok=True)
    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path, exist_ok=True)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path, exist_ok=True)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    # Initialize tracking for best model
    best_info_path = os.path.join(file_weights_path, "best_model_info.json")
    best_val_loss = 1e5
    best_epoch = -1
    
    # Load historical best info if exists
    if os.path.exists(best_info_path):
        with open(best_info_path, 'r') as f:
            best_info = json.load(f)
            best_val_loss = best_info['best_val_loss']
            best_epoch = best_info['best_epoch']
            print(f"Loaded historical best: epoch {best_epoch} with val_loss {best_val_loss}")
    
    start_epoch = 0

    print("Loading IVF Fusion and Low-Light Task!")
    if args.low_light_path is not None:
        train_low_light_path_list, val_low_light_path_list = read_data(args.low_light_path)
        if args.sample:
            print(f"Sampling {args.sample_ratio*100}% of the dataset...")
            train_low_light_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in train_low_light_path_list]
            val_low_light_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in val_low_light_path_list]
    else:
        train_low_light_path_list = val_low_light_path_list = None

    print("Loading IVF Fusion and Over-Exposure Task!")
    if args.over_exposure_path is not None:
        train_over_exposure_path_list, val_over_exposure_path_list = read_data(args.over_exposure_path)
        if args.sample:
            train_over_exposure_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in train_over_exposure_path_list]
            val_over_exposure_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in val_over_exposure_path_list]
    else:
        train_over_exposure_path_list = val_over_exposure_path_list = None

    print("Loading IVF Fusion and ir_low_contrast Task!")
    if args.ir_low_contrast_path is not None:
        train_ir_low_contrast_path_list, val_ir_low_contrast_path_list = read_data(args.ir_low_contrast_path)
        if args.sample:
            train_ir_low_contrast_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in train_ir_low_contrast_path_list]
            val_ir_low_contrast_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in val_ir_low_contrast_path_list]
    else:
        train_ir_low_contrast_path_list = val_ir_low_contrast_path_list = None

    print("Loading IVF Fusion and ir_noise_path Task!")
    if args.ir_noise_path is not None:
        train_ir_noise_path_list, val_ir_noise_path_list = read_data(args.ir_noise_path)
        if args.sample:
            train_ir_noise_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in train_ir_noise_path_list]
            val_ir_noise_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in val_ir_noise_path_list]
    else:
        train_ir_noise_path_list = val_ir_noise_path_list = None

    print("Loading IVF Fusion and ir_stripe_noise_path Task!")
    if args.ir_stripe_noise_path is not None:
        train_ir_stripe_noise_path_list, val_ir_stripe_noise_path_list = read_data(args.ir_stripe_noise_path)
        if args.sample:
            train_ir_stripe_noise_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in train_ir_stripe_noise_path_list]
            val_ir_stripe_noise_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in val_ir_stripe_noise_path_list]
    else:
        train_ir_stripe_noise_path_list = val_ir_stripe_noise_path_list = None

    print("Loading IVF Fusion and vis_blur_path Task!")
    if args.vis_blur_path is not None:
        train_vis_blur_path_list, val_vis_blur_path_list = read_data(args.vis_blur_path)
        if args.sample:
            train_vis_blur_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in train_vis_blur_path_list]
            val_vis_blur_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in val_vis_blur_path_list]
    else:
        train_vis_blur_path_list = val_vis_blur_path_list = None

    print("Loading IVF Fusion and vis_haze_path Task!")
    if args.vis_haze_path is not None:
        train_vis_haze_path_list, val_vis_haze_path_list = read_data(args.vis_haze_path)
        if args.sample:
            train_vis_haze_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in train_vis_haze_path_list]
            val_vis_haze_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in val_vis_haze_path_list]
    else:
        train_vis_haze_path_list = val_vis_haze_path_list = None

    print("Loading IVF Fusion and vis_rain_path Task!")
    if args.vis_rain_path is not None:
        train_vis_rain_path_list, val_vis_rain_path_list = read_data(args.vis_rain_path)
        if args.sample:
            train_vis_rain_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in train_vis_rain_path_list]
            val_vis_rain_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in val_vis_rain_path_list]
    else:
        train_vis_rain_path_list = val_vis_rain_path_list = None

    print("Loading IVF Fusion and vis_random_noise_path Task!")
    if args.vis_random_noise_path is not None:
        train_vis_random_noise_path_list, val_vis_random_noise_path_list = read_data(args.vis_random_noise_path)
        if args.sample:
            train_vis_random_noise_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in train_vis_random_noise_path_list]
            val_vis_random_noise_path_list = [sample_dataset(path_list, args.sample_ratio) for path_list in val_vis_random_noise_path_list]
    else:
        train_vis_random_noise_path_list = val_vis_random_noise_path_list = None

    data_transform = {
        "train": T.Compose([T.RandomCrop(96),
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor()]),

        "val": T.Compose([T.Resize_16(),
                          T.ToTensor()])}

    train_dataset = PromptDataSet(train_low_light_path_list=train_low_light_path_list,
                                  val_low_light_path_list=val_low_light_path_list,
                                  train_over_exposure_path_list=train_over_exposure_path_list,
                                  val_over_exposure_path_list=val_over_exposure_path_list,
                                  train_ir_low_contrast_path_list=train_ir_low_contrast_path_list,
                                  val_ir_low_contrast_path_list=val_ir_low_contrast_path_list,
                                  train_ir_noise_path_list=train_ir_noise_path_list,
                                  val_ir_noise_path_list=val_ir_noise_path_list,
                                  train_ir_stripe_noise_path_list=train_ir_stripe_noise_path_list,
                                  val_ir_stripe_noise_path_list=val_ir_stripe_noise_path_list,
                                  train_vis_blur_path_list=train_vis_blur_path_list,
                                  val_vis_blur_path_list=val_vis_blur_path_list,
                                  train_vis_haze_path_list=train_vis_haze_path_list,
                                  val_vis_haze_path_list=val_vis_haze_path_list,
                                  train_vis_rain_path_list=train_vis_rain_path_list,
                                  val_vis_rain_path_list=val_vis_rain_path_list,
                                  train_vis_random_noise_path_list=train_vis_random_noise_path_list,
                                  val_vis_random_noise_path_list=val_vis_random_noise_path_list,
                                  phase="train",
                              transform=data_transform["train"])

    val_dataset = PromptDataSet(train_low_light_path_list=train_low_light_path_list,
                                  val_low_light_path_list=val_low_light_path_list,
                                  train_over_exposure_path_list=train_over_exposure_path_list,
                                  val_over_exposure_path_list=val_over_exposure_path_list,
                                  train_ir_low_contrast_path_list=train_ir_low_contrast_path_list,
                                  val_ir_low_contrast_path_list=val_ir_low_contrast_path_list,
                                  train_ir_noise_path_list=train_ir_noise_path_list,
                                  val_ir_noise_path_list=val_ir_noise_path_list,
                                  train_ir_stripe_noise_path_list=train_ir_stripe_noise_path_list,
                                  val_ir_stripe_noise_path_list=val_ir_stripe_noise_path_list,
                                  train_vis_blur_path_list=train_vis_blur_path_list,
                                  val_vis_blur_path_list=val_vis_blur_path_list,
                                  train_vis_haze_path_list=train_vis_haze_path_list,
                                  val_vis_haze_path_list=val_vis_haze_path_list,
                                  train_vis_rain_path_list=train_vis_rain_path_list,
                                  val_vis_rain_path_list=val_vis_rain_path_list,
                                  train_vis_random_noise_path_list=train_vis_random_noise_path_list,
                                  val_vis_random_noise_path_list=val_vis_random_noise_path_list,
                                  phase="val",
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model_clip, _ = clip.load("ViT-B/32", device=device)
    model = create_model(model_clip).to(device)

    for param in model.model_clip.parameters():
        param.requires_grad = False

    if args.use_dp == True:
        model = torch.nn.DataParallel(model).cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        print(model.load_state_dict(weights_dict, strict=False))


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_ssim_loss, train_max_loss, train_color_loss, train_text_loss, lr = train_one_epoch(model=model,
                                              model_clip=model_clip,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                lr_scheduler=lr_scheduler,
                                                device=device,
                                                epoch=epoch)

        tb_writer.add_scalar("train_total_loss", train_loss, epoch)        
        tb_writer.add_scalar("train_ssim_loss", train_ssim_loss, epoch)
        tb_writer.add_scalar("train_max_loss", train_max_loss, epoch)
        tb_writer.add_scalar("train_color_loss", train_color_loss, epoch)
        tb_writer.add_scalar("train_text_loss", train_text_loss, epoch)
            
        if epoch % args.val_every_epcho == 0 and epoch != 0:
            val_loss, val_ssim_loss, val_max_loss, val_color_loss, val_text_loss = evaluate(model=model,
                                                 data_loader=val_loader,
                                                 device=device,
                                                 epoch=epoch, lr=lr, filefold_path=file_img_path)

            tb_writer.add_scalar("val_total_loss", val_loss, epoch)
            tb_writer.add_scalar("val_ssim_loss", val_ssim_loss, epoch)
            tb_writer.add_scalar("val_max_loss", val_max_loss, epoch)
            tb_writer.add_scalar("val_color_loss", val_color_loss, epoch)
            tb_writer.add_scalar("val_text_loss", val_text_loss, epoch)

            # Create save file dict
            if args.use_dp == True:
                save_file = {"model": model.module.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            else:
                save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            
            # Save latest checkpoint
            torch.save(save_file, os.path.join(file_weights_path, "checkpoint_lastest.pth"))
            
            # Check if this is globally best model
            if val_loss < best_val_loss:
                # Save as the best checkpoint
                torch.save(save_file, os.path.join(file_weights_path, "checkpoint.pth"))
                
                # Update best model info
                best_val_loss = val_loss
                best_epoch = epoch
                
                # Save best model info to file
                best_info = {
                    'best_val_loss': float(best_val_loss),
                    'best_epoch': int(best_epoch)
                }
                
                with open(best_info_path, 'w') as f:
                    json.dump(best_info, f)
                
                print(f"New best model at epoch {epoch} with val_loss {val_loss}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=120)

    # set the appropriate batch-size value for your device
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)

    # EMS
    parser.add_argument('--low_light_path', type=str, default="SensorFusion/dataset/EMS_dataset_extended/vis_Low_Light")
    parser.add_argument('--over_exposure_path', type=str, default="SensorFusion/dataset/EMS_dataset_extended/vis_Exposure")
    parser.add_argument('--ir_low_contrast_path', type=str, default="SensorFusion/dataset/EMS_dataset_extended/IR_Low_contrast")
    parser.add_argument('--ir_noise_path', type=str, default="SensorFusion/dataset/EMS_dataset_extended/IR_Random_noise")
    parser.add_argument('--ir_stripe_noise_path', type=str, default="SensorFusion/dataset/EMS_dataset_extended/IR_Stripe_noise")
    parser.add_argument('--vis_blur_path', type=str, default="SensorFusion/dataset/EMS_dataset_extended/vis_Blur")
    parser.add_argument('--vis_haze_path', type=str, default="SensorFusion/dataset/EMS_dataset_extended/vis_Haze")
    parser.add_argument('--vis_rain_path', type=str, default="SensorFusion/dataset/EMS_dataset_extended/vis_Rain")
    parser.add_argument('--vis_random_noise_path', type=str, default="SensorFusion/dataset/EMS_dataset_extended/vis_Random_noise")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--val_every_epcho', type=int, default=2, help='val every epcho')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default = False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')
    
    # Add sampling arguments if needed 
    parser.add_argument('--sample', action='store_true', help='whether to sample the dataset')
    parser.add_argument('--sample_ratio', type=float, default=0.4, help='ratio of data to sample (0-1)')

    opt = parser.parse_args()

    main(opt)
