
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

import cv2
from scipy.ndimage import distance_transform_edt

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from .geo_wrapper import GeoDeepLabV3Plus
from .fps_eval import benchmark_pure_fps, benchmark_end_to_end_fps, count_flops, format_extra_metrics


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default='./datasets/data')
    parser.add_argument("--dataset", type=str, default='voc', choices=['voc', 'cityscapes'])
    parser.add_argument("--num_classes", type=int, default=None)

    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenetv3_large', choices=available_models)
    parser.add_argument("--separable_conv", action='store_true', default=False)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=30e3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'])
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--download", action='store_true', default=False)

    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'])

    parser.add_argument("--enable_vis", action='store_true', default=False)
    parser.add_argument("--vis_port", type=str, default='13570')
    parser.add_argument("--vis_env", type=str, default='main')
    parser.add_argument("--vis_num_samples", type=int, default=8)

    parser.add_argument("--use_aux", action='store_true', default=False,
                        help="enable prior/boundary geometric supervision heads")
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--mask_weight", type=float, default=0.65)
    parser.add_argument("--prior_weight", type=float, default=0.15)
    parser.add_argument("--boundary_weight_loss", type=float, default=0.10,
                        help="weight of the boundary_head BCE term in total loss")
    parser.add_argument("--boundary_ce_weight", type=float, default=5.0,
                        help="edge upweighting factor inside the mask CE term")
    return parser


def get_dataset(opts):
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dst = Cityscapes(root=opts.data_root, split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root, split='val', transform=val_transform)
    return train_dst, val_dst


def compute_sdt_and_boundary_batch(label_batch, num_classes, ignore_label=255, dilate_size=2):
    """
    label_batch: (N,H,W) long tensor, values in {0..num_classes-1, ignore_label}
    Returns: distance (N,num_classes,H,W) float32, boundary (N,1,H,W) float32
    CPU/numpy, called once per training batch -- this is the main
    throughput cost of the --use_aux path. If it bottlenecks your
    dataloader, consider precomputing/caching per-image (not per-crop)
    or moving this into a background worker.
    """
    label_np = label_batch.cpu().numpy()
    n, h, w = label_np.shape
    distance = np.zeros((n, num_classes, h, w), dtype=np.float32)
    boundary = np.zeros((n, 1, h, w), dtype=np.float32)

    for b in range(n):
        lm = label_np[b]
        valid = lm != ignore_label

        for c in range(num_classes):
            binary = ((lm == c) & valid).astype(np.uint8)
            if binary.sum() == 0 or binary.sum() == valid.sum():
                distance[b, c] = -1.0 if binary.sum() == 0 else 1.0
                continue
            dist_in = distance_transform_edt(binary)
            dist_out = distance_transform_edt(1 - binary)
            sdf = dist_in - dist_out
            max_val = max(abs(sdf.max()), abs(sdf.min())) + 1e-6
            distance[b, c] = (sdf / max_val).astype(np.float32)

        lm_i = lm.astype(np.int32).copy()
        lm_i[lm_i == ignore_label] = -1
        bmap = np.zeros_like(lm_i, dtype=np.uint8)
        bmap[:-1, :] |= (lm_i[:-1, :] != lm_i[1:, :])
        bmap[1:, :] |= (lm_i[:-1, :] != lm_i[1:, :])
        bmap[:, :-1] |= (lm_i[:, :-1] != lm_i[:, 1:])
        bmap[:, 1:] |= (lm_i[:, :-1] != lm_i[:, 1:])
        bmap[lm_i == -1] = 0
        if dilate_size > 1:
            kernel = np.ones((dilate_size, dilate_size), np.uint8)
            bmap = cv2.dilate(bmap, kernel, iterations=1)
        boundary[b, 0] = bmap.astype(np.float32)

    return (torch.from_numpy(distance), torch.from_numpy(boundary))


def geo_total_loss(outputs, labels, distance, boundary_gt, num_classes,
                    ignore_label=255, mask_weight=0.65, prior_weight=0.15,
                    boundary_head_weight=0.10, boundary_ce_weight=5.0, gamma=2.0):
    logits = outputs["logits"]
    mask = outputs["mask"]

    valid = (labels != ignore_label)
    label_clamped = labels.clone()
    label_clamped[~valid] = 0
    label_onehot = F.one_hot(label_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    probs = mask * valid_f
    label_onehot_v = label_onehot * valid_f
    dims = (0, 2, 3)
    intersection = (probs * label_onehot_v).sum(dims)
    union = probs.sum(dims) + label_onehot_v.sum(dims)
    present = label_onehot_v.sum(dims) > 0
    dice_per_class = (2 * intersection + 1e-6) / (union + 1e-6)
    dice = 1 - dice_per_class[present].mean() if present.sum() > 0 else torch.tensor(0.0, device=logits.device)

    ce = F.cross_entropy(logits, labels, ignore_index=ignore_label, reduction="none")
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    boundary_w = 1.0 + (boundary_ce_weight - 1.0) * boundary_gt.squeeze(1)
    valid_ce = valid.float()
    focal_ce = (focal * boundary_w * valid_ce).sum() / valid_ce.sum().clamp(min=1.0)

    mask_loss = dice + focal_ce
    loss = mask_weight * mask_loss
    logs = {"loss_dice": dice.item(), "loss_focal_ce": focal_ce.item(), "loss_mask": mask_loss.item()}

    if "prior" in outputs:
        spatial_loss = F.smooth_l1_loss(outputs["prior"], distance)
        boundary_loss = F.binary_cross_entropy(outputs["boundary"], boundary_gt)
        loss = loss + prior_weight * spatial_loss + boundary_head_weight * boundary_loss
        logs["loss_prior"] = spatial_loss.item()
        logs["loss_boundary_head"] = boundary_loss.item()

    return loss, logs


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            logits = outputs["logits"]
            preds = logits.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append((images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for j in range(len(images)):
                    image = images[j].detach().cpu().numpy()
                    target = targets[j]
                    pred = preds[j]
                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)
                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    print("use_aux (geometric supervision): %s" % opts.use_aux)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    base_model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(base_model.classifier)
    utils.set_bn_momentum(base_model.backbone, momentum=0.01)

    aux_in_channels = None
    if opts.use_aux:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            feat = base_model.backbone(dummy)
            aux_in_channels = feat['out'].shape[1]
        print(f"[main_geo] aux heads will read {aux_in_channels}-channel backbone feature")

    model = GeoDeepLabV3Plus(base_model, num_classes=opts.num_classes, use_aux=opts.use_aux, aux_in_channels=aux_in_channels)

    metrics = StreamSegMetrics(opts.num_classes)

    aux_params = []
    if opts.use_aux and model.prior_head is not None:
        aux_params = [{'params': list(model.prior_head.parameters()) + list(model.boundary_head.parameters()),
                       'lr': opts.lr}]

    optimizer = torch.optim.SGD(params=[
        {'params': model.base_model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.base_model.classifier.parameters(), 'lr': opts.lr},
    ] + aux_params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    def save_ckpt(path):
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples, np.int32) if opts.enable_vis else None
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        pure_result = benchmark_pure_fps(model, (1, 3, opts.crop_size, opts.crop_size), device)
        e2e_result = benchmark_end_to_end_fps(model, val_loader, device)
        flops_result = count_flops(model, (1, 3, opts.crop_size, opts.crop_size))
        print(format_extra_metrics(pure_result, e2e_result, flops_result))
        return

    interval_loss = 0
    while True:
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            distance = None
            distance_np, boundary_gt = compute_sdt_and_boundary_batch(
                labels, opts.num_classes, opts.ignore_label
            )
            boundary_gt = boundary_gt.to(device)
            if opts.use_aux:
                distance = distance_np.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss, logs = geo_total_loss(
                outputs, labels, distance, boundary_gt, opts.num_classes,
                ignore_label=opts.ignore_label, mask_weight=opts.mask_weight,
                prior_weight=opts.prior_weight, boundary_head_weight=opts.boundary_weight_loss,
                boundary_ce_weight=opts.boundary_ce_weight,
            )
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % opts.print_interval == 0:
                interval_loss = interval_loss / opts.print_interval
                print("Epoch %d, Itrs %d/%d, Loss=%f | %s" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss, logs))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d_aux%s.pth' %
                          (opts.model, opts.dataset, opts.output_stride, opts.use_aux))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device,
                                                  metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                pure_result = benchmark_pure_fps(model, (1, 3, opts.crop_size, opts.crop_size), device)
                e2e_result = benchmark_end_to_end_fps(model, val_loader, device)
                flops_result = count_flops(model, (1, 3, opts.crop_size, opts.crop_size))
                print(format_extra_metrics(pure_result, e2e_result, flops_result))
                if val_score['Mean IoU'] > best_score:
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d_aux%s.pth' %
                              (opts.model, opts.dataset, opts.output_stride, opts.use_aux))

                if vis is not None:
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()