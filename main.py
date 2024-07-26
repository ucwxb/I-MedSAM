import argparse
from datetime import datetime
import os
join = os.path.join
import shutil
import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
import monai
import matplotlib.pyplot as plt
from tqdm import tqdm
from dist import is_primary,barrier,all_reduce_average, init_distributed_mode
from torch.utils.data.distributed import DistributedSampler
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from model import IMedSAM
from dataset import SessileDataLoader, CVCDataLoader
from utils.SurfaceDice import compute_dice_coefficient, compute_HD_distances

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
np.random.seed(2023)

# set up parser
parser = argparse.ArgumentParser("I-MedSAM", add_help=False)
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--task_name", type=str, default="MedSAM_MoELoRA")
parser.add_argument("--method", type=str, default="imedsam")
parser.add_argument("--lora_rank", type=int, default=4)
parser.add_argument("--eval_step", type=int, default=10)
parser.add_argument("--checkpoint", type=str, default="./sam_ckp/sam_vit_b_01ec64.pth")
parser.add_argument("--work_dir", type=str, default="./work_dir")
parser.add_argument("--data_path", type=str, default="./dataset/sessile-Kvasir")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--device_ids", type=int, default=[0], nargs='+')
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument("--image_size", type=int, default=384)
parser.add_argument("--label_size", type=int, default=384)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--top_k_ratio", type=float, default=0.125,)
parser.add_argument("--train_ratio", type=float, default=1.0,)
parser.add_argument("--use_amp", action="store_true", default=False)
parser.add_argument("--wo_fa", action="store_true", default=False, 
                    help="without frequency adapter")
parser.add_argument("--wo_inr", action="store_true", default=False, 
                    help="without inr")
parser.add_argument("--test_only", action="store_true", default=False, 
                    help="test_only")
parser.add_argument("--save_pic", action="store_true", default=False, 
                    help="use save_pic")
parser.add_argument("--filter", action="store_true", default=False, 
                    help="filter")
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--local_rank", type=int,default=0)
parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
args = parser.parse_args()

# init distributed mode
if args.distributed:
    init_distributed_mode(args)
    torch.backends.cudnn.benchmark = True

model_save_path = join(args.work_dir, args.task_name)
device = torch.device(args.device)


def loss_calculation(mask_pred, gt2D, criterion, epoch, B):

    loss_coarse = criterion(mask_pred[0], gt2D)
    loss_fine = criterion(mask_pred[1], gt2D)
    loss_coarse /= B
    loss_fine /= B
    advanced_ratio = min((epoch + 1)/5, 1.0)
    inversed_ratio = 1 - advanced_ratio
    loss = inversed_ratio * loss_coarse + advanced_ratio * loss_fine

    return loss

def main():
    if is_primary() and not args.test_only:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, join(model_save_path, os.path.basename(__file__))
        )

        # list and save all python scripts
        save_ext = [".py", ".sh"]
        copy_dir_directly = ["segment_anything", "model", "utils"]
        os.makedirs(join(model_save_path, "scripts"), exist_ok=True)
        for copy in copy_dir_directly:
            shutil.copytree(copy, join(model_save_path, "scripts", copy))
        for file_name in os.listdir("."):
            if os.path.splitext(file_name)[-1] in save_ext and os.path.isfile(file_name):
                shutil.copyfile(
                    file_name,
                    join(model_save_path, "scripts", file_name),
                )

    if "sessile" in args.data_path:
        train_dataset = SessileDataLoader(args.data_path, train=True, resize_size=(args.image_size,args.image_size))
    elif "CVC" in args.data_path:
        train_dataset = CVCDataLoader(args.data_path, train=True, resize_size=(args.image_size,args.image_size))
    print("Number of training samples: ", len(train_dataset)) if is_primary() else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True if not args.distributed else False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(train_dataset) if args.distributed else None,
    )

    if "sessile" in args.data_path:
        test_dataset = SessileDataLoader(args.data_path, train=False, resize_size=(args.image_size,args.image_size), label_resize_size=(args.label_size,args.label_size), train_ratio=args.train_ratio)
    elif "CVC" in args.data_path:
        test_dataset = CVCDataLoader(args.data_path, train=False, resize_size=(args.image_size,args.image_size), label_resize_size=(args.label_size,args.label_size))
    print("Number of testing samples: ", len(test_dataset)) if is_primary() else None
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=DistributedSampler(test_dataset) if args.distributed else None,
    )

    class_num = train_dataset.class_num
    image_size = train_dataset.image_size

    sam_model = sam_model_registry[args.model_type](image_size = image_size, checkpoint=args.checkpoint, class_num=class_num, top_k_ratio=args.top_k_ratio)
    if args.method == "imedsam":
        imedsam_model = IMedSAM(sam_model, args.lora_rank).to(device)
    else:
        raise NotImplementedError("Method {} not implemented!".format(args.method))
    
    if is_primary():
        print(
            "Number of total parameters: ",
            sum(p.numel() for p in imedsam_model.parameters()),
        )
        print(
            "Number of trainable parameters: ",
            sum(p.numel() for p in imedsam_model.parameters() if p.requires_grad),
        )
    all_params = imedsam_model.parameters()
    weight_params = []
    for pname, p in imedsam_model.named_parameters():
        if "seg_net" in pname:
            weight_params += [p]
    params_id = list(map(id, weight_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))

    optimizer = torch.optim.AdamW([
            {'params': weight_params , 'lr': args.lr}, 
            {'params': other_params, 'lr': 0.05*args.lr}],
            lr=args.lr,
            weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=1000)
    criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean', lambda_dice=0.8, lambda_ce=0.2)

    num_epochs = args.num_epochs
    start_epoch = 0
    losses = []
    dscs = []
    hds = []
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            imedsam_model.load_lora_parameters(checkpoint["model"])
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            for param_group in optimizer.param_groups:
                param_group['capturable'] = True # If capturable=False, state_steps should not be CUDA tensors.

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    best_metrics = [{"epoch": 0,"dsc": 0,"hd": 100}]

    sam_trans = ResizeLongestSide(imedsam_model.sam.image_encoder.img_size)

    if args.distributed:
        imedsam_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(imedsam_model)
        imedsam_model = torch.nn.parallel.DistributedDataParallel(imedsam_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)    

    if args.test_only:
        if args.save_pic:
            save_dir = os.path.join(os.path.dirname(args.resume), "test_results_%s_%d"%(os.path.basename(args.data_path).split(".")[0], args.label_size))
            os.makedirs(save_dir, exist_ok=True)

        dsc = 0
        hd = 0
        step = 0
        imedsam_model.eval()
        pbar_test = tqdm(test_dataloader) if is_primary() else test_dataloader
        pbar_test.set_description(f"Test") if is_primary() else None
        with torch.no_grad():
            for img, gt2D, box, cls, img_file, gt_file in pbar_test:
                box_np = box.numpy()
                box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :] # (B, 1, 4)
                img, gt2D = img.to(device), gt2D.to(device)
                B,H,W = img.shape[0],img.shape[2],img.shape[3]
                
                if args.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        mask_pred = imedsam_model(img, box_torch, original_size=gt2D.shape[-2:], cls=cls)
                else:
                    mask_pred = imedsam_model(img, box_torch, original_size=gt2D.shape[-2:], cls=cls)

                mask_prob = torch.sigmoid(mask_pred[1][0,0:1,...])
                mask_prob = mask_prob.squeeze(1).detach().cpu().numpy()
                mask = (mask_prob > 0.5).astype(np.uint8)

                gt2D = gt2D.squeeze(1).cpu().numpy().astype(np.uint8)
                
                dsc_iter = compute_dice_coefficient(gt2D, mask)
                hd_iter = compute_HD_distances(gt2D, mask)

                dsc += dsc_iter * B
                hd += hd_iter * B

                step += B
                pbar_test.set_postfix({"dsc": dsc_iter, "hd": hd_iter}) if is_primary() else None

                if args.save_pic:
                    mask_save_path = os.path.join(save_dir, os.path.basename(img_file[0]).split('.')[0] + ".png")
                    cv2.imwrite(mask_save_path, mask[0]*255)
        
            dsc /= step
            hd /= step
            if args.distributed:
                barrier()
            dsc = all_reduce_average(torch.tensor(dsc).to(device)).item() if args.distributed else dsc
            hd = all_reduce_average(torch.tensor(hd).to(device)).item() if args.distributed else hd
            dscs.append(dsc)
            hds.append(hd)
            print(
                f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Test data path: {args.data_path} Size: {args.label_size}, DSC: {dsc}, HD: {hd}'
            ) if is_primary() else None
            exit()

    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()
        epoch_loss = 0
        step = 0
        imedsam_model.train()
        pbar_train = tqdm(train_dataloader) if is_primary() else train_dataloader
        pbar_train.set_description(f"Epoch [{epoch+1}/{num_epochs}] Train") if is_primary() else None
        for img, gt2D, box, cls, _, _ in pbar_train:
            B = img.shape[0]
            optimizer.zero_grad()
            box_np = box.numpy()
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            img, gt2D = img.to(device), gt2D.to(device)
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    mask_pred = imedsam_model(img, box_torch, epoch_T=(epoch + 1)/20, cls=cls)
                    loss = loss_calculation(mask_pred, gt2D, criterion, epoch, B)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                mask_pred = imedsam_model(img, box_torch, cls=cls, epoch_T=(epoch + 1)/20)
                loss = loss_calculation(mask_pred, gt2D, criterion, epoch, B)
                loss.backward()
                optimizer.step()

            loss = loss if not args.distributed else all_reduce_average(loss)
            epoch_loss += loss.item()
            step += 1
            pbar_train.set_postfix({"loss": loss.item()}) if is_primary() else None

        scheduler.step(epoch=epoch)
        epoch_loss /= step
        losses.append(epoch_loss)

        print(
            f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        ) if is_primary() else None
        ## save the latest model
        if args.distributed:
            barrier()

        checkpoint = {
            "model": imedsam_model.save_lora_parameters() if not args.distributed else imedsam_model.module.save_lora_parameters(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, join(model_save_path, "model_latest.pth")) if is_primary() else None

        # plot loss
        if is_primary():
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(join(model_save_path, "train_loss.png"))
            plt.close()

        torch.cuda.empty_cache()
        if (epoch + 1) % args.eval_step == 0:
            # test
            dsc = 0
            hd = 0
            step = 0
            imedsam_model.eval()
            pbar_test = tqdm(test_dataloader) if is_primary() else test_dataloader
            pbar_test.set_description(f"Epoch [{epoch+1}/{num_epochs}] Test") if is_primary() else None
            with torch.no_grad():
                for img, gt2D, box, cls, img_file, gt_file in pbar_test:
                    box_np = box.numpy()
                    box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
                    box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                    if len(box_torch.shape) == 2:
                        box_torch = box_torch[:, None, :] # (B, 1, 4)
                    img, gt2D = img.to(device), gt2D.to(device)
                    B,H,W = img.shape[0],img.shape[2],img.shape[3]
                    
                    if args.use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            mask_pred = imedsam_model(img, box_torch, original_size=gt2D.shape[-2:], cls=cls)
                    else:
                        mask_pred = imedsam_model(img, box_torch, original_size=gt2D.shape[-2:], cls=cls)

                    mask_prob = torch.sigmoid(mask_pred[1][0,0:1,...])
                    mask_prob = mask_prob.squeeze(1).detach().cpu().numpy()
                    mask = (mask_prob > 0.5).astype(np.uint8)

                    gt2D = gt2D.squeeze(1).cpu().numpy().astype(np.uint8)
                    
                    dsc_iter = compute_dice_coefficient(gt2D, mask)
                    hd_iter = compute_HD_distances(gt2D, mask)

                    dsc += dsc_iter * B
                    hd += hd_iter * B

                    step += B
                    pbar_test.set_postfix({"dsc": dsc_iter, "hd": hd_iter}) if is_primary() else None
            
                dsc /= step
                hd /= step
                if args.distributed:
                    barrier()
                dsc = all_reduce_average(torch.tensor(dsc).to(device)).item() if args.distributed else dsc
                hd = all_reduce_average(torch.tensor(hd).to(device)).item() if args.distributed else hd
                dscs.append(dsc)
                hds.append(hd)

                print(
                    f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, DSC: {dsc}, HD: {hd}'
                ) if is_primary() else None
                
                # plot metric
                if is_primary():
                    plt.plot(dscs)
                    plt.plot(hds)
                    plt.title("Testing Metric")
                    plt.xlabel("Epoch")
                    plt.ylabel("Metric")
                    plt.legend(["DSC", "HD"])
                    plt.savefig(join(model_save_path, "test_metric.png"))
                    plt.close()

                    # if dsc > best_metrics[0]["dsc"] and hd < best_metrics[0]["hd"]:
                    if dsc > best_metrics[0]["dsc"]:
                        best_metrics[0]["dsc"] = dsc
                        best_metrics[0]["hd"] = hd
                        best_metrics[0]["epoch"] = epoch
                        checkpoint = {
                            "model": imedsam_model.save_lora_parameters() if not args.distributed else imedsam_model.module.save_lora_parameters(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "metrics": best_metrics
                        }
                        torch.save(checkpoint, join(model_save_path, "model_best_DSC_%.5f_HD_%.5f_ep_%d.pth"%(best_metrics[0]["dsc"],best_metrics[0]["hd"], epoch)))
                    print("Best Metrics: ", best_metrics)
                    print("Work dir:", args.work_dir)

if __name__ == "__main__":
    main()
