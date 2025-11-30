from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch, math, argparse, yaml

def evaluate_3d_new(loader, model, cfg, ifdist=None, val_mode=None):
    # add padding to adapt to different input size
    print(f"{val_mode} Validation begin")
    model.eval()
    total_samples = 0
    all_mDice_organ = 0
    dice_class_all = torch.zeros((cfg['nclass']-1,), device='cuda')
    stream = torch.cuda.Stream()
    with torch.no_grad(): 
        for img, mask, id in tqdm(loader):
            with torch.cuda.stream(stream):
                img, mask = img.cuda(), mask.squeeze(0).cuda()
                big = False
                total_samples += 1
                dice_class = torch.zeros((cfg['nclass']-1,), device=img.device)      # dismiss background
                _, _, d, h, w = img.shape
                patch_d, patch_h, patch_w = cfg['val_patch_size']
                score_map = torch.zeros((cfg['nclass'], d, h, w), device=img.device)
                count_map = torch.zeros(img.shape[2:], device=img.device)
                if w // patch_w > 2:
                    big = True
                if (w // patch_w) > 4:
                        patch_d, patch_h, patch_w = cfg['val_patch_size_small']
                num_h = math.ceil(h / patch_h)
                overlap_h = (patch_h * num_h - h) // (num_h - 1) if num_h > 1 else 0
                stride_h = patch_h - overlap_h
                num_w = math.ceil(w / patch_w)
                overlap_w = (patch_w * num_w - w) // (num_w - 1) if num_w > 1 else 0
                stride_w = patch_w - overlap_w
                num_d = math.ceil(d / patch_d)
                overlap_d = (patch_d * num_d - d) // (num_d - 1) if num_d > 1 else 0
                stride_d = patch_d - overlap_d
                d_starts = torch.arange(0, num_d, device=img.device) * stride_d
                d_starts = torch.clamp(d_starts, max=d-patch_d)
                h_starts = torch.arange(0, num_h, device=img.device) * stride_h
                h_starts = torch.clamp(h_starts, max=h-patch_h)
                w_starts = torch.arange(0, num_w, device=img.device) * stride_w
                w_starts = torch.clamp(w_starts, max=w-patch_w)
                d_idx = d_starts[:, None] + torch.arange(patch_d, device=img.device)
                h_idx = h_starts[:, None] + torch.arange(patch_h, device=img.device)   
                w_idx = w_starts[:, None] + torch.arange(patch_w, device=img.device)  

                # Here are a more efficent way if GPU memory is avaible, we can use vector to val.
                # img_patch = img[:, :, d_idx[:, None, None, :, None, None], h_idx[None, :, None, None,  :, None], w_idx[None, None, :,None, None, :]] 
                # pred_patch_logits = model(img_patch)
                # pred_patch_logits = torch.softmax(pred_patch_logits, dim=1)
                # pred_mask = pred_patch_logits.argmax(dim=1)
                for dd in d_idx:
                    for hh in h_idx:
                        for ww in w_idx:
                            d_id = dd.unsqueeze(1).unsqueeze(2).expand(-1, patch_h, patch_w)
                            h_id = hh.unsqueeze(0).unsqueeze(2).expand(patch_d, -1, patch_w)  
                            w_id = ww.unsqueeze(0).unsqueeze(1).expand(patch_d, patch_h, -1)    
                            input = img[:, :, d_id, h_id, w_id]
                            pred_patch_logits = model(input)
                            pred_patch = torch.softmax(pred_patch_logits, dim=1)
                            score_map[:, d_id, h_id, w_id] += pred_patch.squeeze(0)
                            count_map[d_id, h_id, w_id] += 1
                score_map /= count_map
                pred_mask = score_map.argmax(dim=0)
                del score_map, count_map, input, pred_patch_logits, pred_patch
                torch.cuda.empty_cache()

                classes = torch.arange(1, cfg['nclass'], device=mask.device)                             # (nclass-1, 1, 1, 1)
                mask_exp = mask.unsqueeze(0) == classes.unsqueeze(1).unsqueeze(2).unsqueeze(3)           # (nclass-1, D, H, W)
                pred_exp = pred_mask.unsqueeze(0) == classes.unsqueeze(1).unsqueeze(2).unsqueeze(3)      # (nclass-1, D, H, W)
                mask_exp, pred_exp = mask_exp.view(cfg['nclass']-1, -1), pred_exp.view(cfg['nclass']-1, -1)
                if big:
                    intersection = torch.zeros(mask_exp.shape[0], device=mask_exp.device)
                    union = torch.zeros(mask_exp.shape[0], device=mask_exp.device)
                    for i in range(mask_exp.shape[0]):
                        intersection[i] = (mask_exp[i] * pred_exp[i]).sum()
                        union[i] = mask_exp[i].sum() + pred_exp[i].sum()
                else:
                    intersection = (mask_exp * pred_exp).sum(dim=1)
                    union = mask_exp.sum(dim=1) + pred_exp.sum(dim=1)
                dice_class += (2. * intersection) / (union + 1e-7)
                mDice_organ = dice_class.mean()
                all_mDice_organ += mDice_organ
                dice_class_all += dice_class
                del mask_exp, pred_exp, intersection, union, pred_mask
                torch.cuda.empty_cache()
            total_samples_tensor = torch.tensor(total_samples).cuda()
    if ifdist:
        dist.all_reduce(all_mDice_organ)
        dist.all_reduce(dice_class_all)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        all_mDice_organ /= total_samples_tensor
        dice_class_all /= total_samples_tensor
    else:
        all_mDice_organ /= total_samples_tensor
        dice_class_all /= total_samples_tensor
    dice_class_all = [dice / total_samples for dice in dice_class_all.tolist()]
    print(f"{val_mode} Validation end")
    return all_mDice_organ.item(), dice_class_all

parser = argparse.ArgumentParser(description='Fixmatch_mt for Flare22')
parser.add_argument('--config', type=str, default='./configs/your_trainer.yaml')
def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    model = UNet(1, cfg['nclass'])
    valset = YourDataset()
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    mean_dice, dice_for_every_class = evaluate_3d_new(valloader, model, cfg, val_mode='model') 
    # valsampler = DistributedSampler(valset)   # for distrubuted train
    # valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)
