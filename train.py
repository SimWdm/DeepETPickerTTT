import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    # prefer deterministic algorithms but allow warn-only for ops without deterministic kernels
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    # older torch versions may not have this API
    pass
from torch.multiprocessing import set_sharing_strategy
try:
    set_sharing_strategy('file_system')
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import os
import sys
from dataset.dataloader_DynamicLoad import Dataset_ClsBased
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.loss import DiceLoss, CELoss, BCELoss
from utils.metrics import seg_metrics
from utils.cc_loss import CCLoss3D
from utils.backup import backup_python_files  
from utils.colors import COLORS
from model.model_loader import get_model
from utils.misc import combine, cal_metrics_NMS_OneCls, get_centroids, cal_metrics_MultiCls, combine_torch
from sklearn.metrics import precision_recall_fscore_support
import time
import json

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

class UNetExperiment(pl.LightningModule):
    def __init__(self, args):
        if args.f_maps is None:
            args.f_maps = [32, 64, 128, 256]
        print(args.pad_size)

        if len(args.configs) > 0:
            with open(args.configs, 'r') as f:
                self.cfg = json.loads(''.join(f.readlines()).lstrip('train_configs='))
        else:
            self.cfg = {}
        if len(args.train_configs) > 0:
            with open(args.train_configs, 'r') as f:
                self.train_cfg = json.loads(''.join(f.readlines()).lstrip('train_configs='))
        else:
            self.train_cfg = self.cfg

        if len(args.val_configs) > 0:
            with open(args.val_configs, 'r') as f:
                self.val_config = json.loads(''.join(f.readlines()).lstrip('train_configs='))
        else:
            self.val_cfg = self.cfg

        super(UNetExperiment, self).__init__()
        self.save_hyperparameters()
        self.model = get_model(args)
        print(self.model)

        if args.loss_func_seg == 'Dice':
            self.loss_function_seg = DiceLoss(args=args)
        elif args.loss_func_seg == 'Dice4':
            self.loss_function_seg = DiceLoss(args=args, beta=4)
        elif args.loss_func_seg == 'CE':
            if args.num_classes > 1:
                self.loss_function_seg = CELoss(args=args)
            else:
                self.loss_function_seg = BCELoss(args=args)
                #raise NotImplementedError("For binary classification, please use Dice loss.")
        elif args.loss_func_seg == 'BCE':
            self.loss_function_seg = BCELoss(args=args)
        else:
            raise ValueError(f"Unsupported loss_func_seg: {args.loss_func_seg}")
        if args.denoising:
            self.loss_function_denoising = CCLoss3D(reduction="mean")
            
        # log f4 loss
        self.f4_loss = DiceLoss(args=args, beta=4)

        if 'gaussian' in self.val_cfg["label_type"]:
            self.thresholds = np.linspace(0.15, 0.45, 7)
        elif 'sphere' in self.val_cfg["label_type"]:
            self.thresholds = np.linspace(0.2, 0.80, 13)
        self.partical_volume = 4 / 3 * np.pi * (self.val_cfg["label_diameter"] / 2) ** 3
        self.args = args
    
    def on_train_epoch_start(self):
        if self.global_step == 0:
            if self.trainer.global_rank == 0:
                print("Making code backup...")
                backup_code_dir = f"{self.logger.log_dir}/code_backup"
                # src is parent of this file
                src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                backup_python_files(src=src, dest=backup_code_dir, exclude_dirs=["code_backup"])
                print("... done!")


    def forward(self, x):
        return self.model(x)
    
    def get_denoising_loss(self, batch, return_denoising_output=False):
        img_even, img_odd = batch['img_even'], batch['img_odd']
        denoising_output = self.forward(img_even)[:, -1, :, :, :].unsqueeze(1)
        loss_denoising = self.loss_function_denoising(denoising_output, img_odd)
        if return_denoising_output:
            return loss_denoising, denoising_output
        else:
            return loss_denoising

    def training_step(self, train_batch, batch_idx):
        args = self.args
        #img, label, index = train_batch
        img, label = train_batch['img'], train_batch['label']
        img = img.to(torch.float32)
                
        train_loss = 0
        denom = 0
        
        if args.train_denoising_only:
            loss_seg = 0
        else:
            seg_output = self.model.get_segmentation_output(img)
            
            if args.use_mask:
                mask = label.clone().detach()
                mask[mask > 0] = 1
                label[label < 255] = 0
                label[label > 0] = 1

                # update label and mask according to label-threshold
                label[seg_output > args.seg_tau] = 1
                mask[seg_output > args.seg_tau] = 1
                mask[seg_output < (1 - args.seg_tau)] = 1

                seg_output = seg_output * mask
            loss_seg = self.loss_function_seg(seg_output, label, trust_labels=train_batch["trust_label"])
            denom += 1
        train_loss = loss_seg

        if args.denoising:
            loss_denoising = self.get_denoising_loss(train_batch)
            train_loss = train_loss + loss_denoising
            denom += 1
            #train_loss = loss_denoising
        
        train_loss = train_loss / denom
            
        self.log('train_loss', train_loss, on_step=False, on_epoch=True)
        if args.denoising:
            self.log('train_loss_denoising', loss_denoising, on_step=False, on_epoch=True)
            self.log('train_loss_seg', loss_seg, on_step=False, on_epoch=True, prog_bar=False)
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        args = self.args
        with torch.no_grad():
            #img, label, index = val_batch
            img, label, index = val_batch['img'], val_batch['label'], val_batch['position']
            index = torch.cat([i.view(1, -1) for i in index], dim=0).permute(1, 0)
            img = img.to(torch.float32)
            self.seg_output = self.model.get_segmentation_output(img)

            if (batch_idx >= self.len_block // args.batch_size and args.test_mode == "test_val") or \
                    args.test_mode == "test" or args.test_mode == "val" or args.test_mode == "val_v1":
                loss_seg = self.loss_function_seg(self.seg_output, label, trust_labels=val_batch["trust_label"])
                val_loss = loss_seg
                if args.denoising:
                    loss_denoising, output_denoising = self.get_denoising_loss(val_batch, return_denoising_output=True)
                    val_loss = (val_loss + loss_denoising) / 2
                    
                precision, recall, f1_score, iou = seg_metrics(self.seg_output, label, threshold=args.threshold)
                f4_loss = self.f4_loss(self.seg_output, label, trust_labels=val_batch["trust_label"])

                self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True)
                if args.denoising:
                    self.log('val_loss_seg', loss_seg, on_step=False, on_epoch=True, sync_dist=True)
                    self.log('val_loss_denoising', loss_denoising, on_step=False, on_epoch=True, sync_dist=True)
                self.log('val_precision', precision, on_step=False, on_epoch=True, sync_dist=True)
                self.log('val_recall', recall, on_step=False, on_epoch=True, sync_dist=True)
                self.log('val_f1', f1_score, on_step=False, on_epoch=True, sync_dist=True)
                self.log('val_iou', iou, on_step=False, on_epoch=True, sync_dist=True)
                self.log('val_f4_loss', f4_loss, on_step=False, on_epoch=True, sync_dist=True)
                self.log("epoch_idx", self.current_epoch, on_epoch=True, prog_bar=True, sync_dist=True)
                
                # return loss_seg
                tensorboard = self.logger.experiment

                if (batch_idx == (self.len_block // args.batch_size + 1) and args.test_mode == 'test_val') or \
                        (batch_idx == 0 and args.test_mode == 'test') or \
                        (batch_idx == 0 and args.test_mode == 'val') or \
                        (
                                batch_idx == 0 and args.test_mode == 'val_v1'):  # and True == False  and self.current_epoch % 1 == 0
                    img /= img.abs().max()  # [-1,1]
                    img = img * 0.5 + 0.5  # [0, 1]
                    img_ = img[0, :, 0:(args.block_size - 1):5, :, :].permute(1, 0, 2, 3).repeat(
                        (1, 3, 1, 1))  # sample0: [5, 3, y, x]

                    label_ = label[0, :, 0:(args.block_size - 1):5, :, :]  # sample0 [15, y, x]
                    temp = torch.zeros(
                        (len(np.arange(0, args.block_size - 1, 5)), args.block_size, args.block_size, 3)).float()
                    # print(label.shape, temp.shape)
                    for idx in np.arange(label_.shape[0]):
                        temp[label_[idx] > 0.5] = torch.tensor(
                            COLORS[(idx + 1) if (args.num_classes == 1
                                                 or args.use_paf or
                                                 label_.shape[0] == 1) else idx]).float()
                    label__ = temp.permute(0, 3, 1, 2).contiguous().cuda()  # [15, 3, y, x]

                    seg_output_ = self.seg_output[0, :, 0:(args.block_size - 1):5, :, :]  # sample0 [15, y, x]
                    seg_threshes = [0.5, 0.3, 0.2, 0.15, 0.1, 0.05]
                    seg_preds = []
                    for thresh in seg_threshes:
                        temp = torch.zeros(
                            (len(np.arange(0, args.block_size - 1, 5)), args.block_size, args.block_size, 3)).float()
                        for idx in np.arange(seg_output_.shape[0]):
                            temp[seg_output_[idx] > thresh] = torch.tensor(
                                COLORS[(idx + 1) if (args.num_classes == 1
                                                     or args.use_paf or
                                                     seg_output_.shape[0] == 1) else idx]).float()
                        seg_preds.append(temp.permute(0, 3, 1, 2).contiguous().cuda())  # [15, 3, y, x]

                    seg_preds = torch.cat(seg_preds, dim=0)

                    img_label_seg = torch.cat([img_, label__, seg_preds], dim=0)
                    img_label_seg = make_grid(img_label_seg, (args.block_size - 1) // 5 + 1, padding=2, pad_value=120)

                    tensorboard.add_image('img_label_seg', img_label_seg, self.current_epoch, dataformats="CHW")
                    
                    if args.denoising:
                        img_denoising = output_denoising[0, :, 0:(args.block_size - 1):5, :, :].permute(1, 0, 2, 3).repeat((1, 3, 1, 1))
                        img_denoising = img_denoising * 0.5 + 0.5  # [0, 1]
                        img_denoising = make_grid(img_denoising, (args.block_size - 1) // 5 + 1, padding=2, pad_value=120)
                        tensorboard.add_image('img_denoising', img_denoising, self.current_epoch, dataformats="CHW")
                    
            if args.num_classes > 1:
                out = self._nms_v2(self.seg_output[:, 1:], kernel=args.meanPool_kernel, mp_num=6, positions=index)
                self._val_epoch_output.append(out)
                return self._nms_v2(self.seg_output[:, 1:], kernel=args.meanPool_kernel, mp_num=6, positions=index)
            else:                
                out = self._nms_v2(self.seg_output[:, :], kernel=args.meanPool_kernel, mp_num=6, positions=index)
                self._val_epoch_output.append(out)
                return out

    def validation_step_end(self, outputs):
        args = self.args
        if 'test' in args.test_mode:
            return outputs

    def on_validation_epoch_start(self):
        # will hold per-batch outputs that you previously returned
        self._val_epoch_output = []

    def on_validation_epoch_end(self):
        args = self.args
        epoch_output = getattr(self, "_val_epoch_output", [])
        if len(epoch_output) == 0:
            return
        
        with torch.no_grad():
            if 'test' in args.test_mode:
                if args.meanPool_NMS:
                    if args.num_classes == 1:
                        # coords_out: [N, 5]
                        coords_out = torch.cat(epoch_output, dim=0).detach().cpu().numpy()
                        if coords_out.shape[0] > 50000:
                            loc_p, loc_r, loc_f1, avg_dist = 1e-10, 1e-10, 1e-10, 100
                        else:
                            loc_p, loc_r, loc_f1, avg_dist = \
                                cal_metrics_NMS_OneCls(coords_out,
                                                       self.gt_coords,
                                                       self.occupancy_map,
                                                       self.cfg,
                                                       )
                        print("*" * 100)
                        print(f"Precision:{loc_p}")
                        print(f"Recall:{loc_r}")
                        print(f"F1-score:{loc_f1}")
                        print(f"Avg-dist:{avg_dist}")
                        print("*" * 100)
                        self.log('cls_precision', loc_p, on_step=False, on_epoch=True)
                        self.log('cls_recall', loc_r, on_step=False, on_epoch=True)
                        self.log('cls_f1', loc_f1, on_step=False, on_epoch=True)
                        self.log('cls_dist', avg_dist, on_step=False, on_epoch=True)
                        pr = (loc_p * (loc_r ** args.prf1_alpha)) / (loc_p + (loc_r ** args.prf1_alpha) + 1e-10)
                        self.log(f'cls_pr_alpha{args.prf1_alpha:.1f}', pr, on_step=False, on_epoch=True)
                        time.sleep(0.5)
                    else:
                        coords_out = torch.cat(epoch_output, dim=0).detach().cpu().numpy()
                        loc_p, loc_r, loc_f1, loc_miss, avg_dist, gt_classes, pred_classes, self.num2pdb, cls_f1 = \
                            cal_metrics_MultiCls(coords_out, self.gt_coords, self.occupancy_map, self.cfg, args,
                                                 args.pad_size, self.dir_name, self.partical_volume)
                        self.log('cls_f1', cls_f1, on_step=False, on_epoch=True)
        self._val_epoch_output.clear()


    def train_dataloader(self):
        args = self.args
        train_dataset = Dataset_ClsBased(mode=args.train_mode,
                                         block_size=args.block_size,
                                         num_class=args.num_classes,
                                         random_num=args.random_num,
                                         use_bg=args.use_bg,
                                         data_split=args.data_split,
                                         use_paf=args.use_paf,
                                         cfg=self.train_cfg,
                                         args=args)
        return DataLoader(train_dataset,
                batch_size=args.batch_size,
               num_workers=16,
                shuffle=True,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
        )

    def val_dataloader(self):
        args = self.args
        val_dataset = Dataset_ClsBased(mode=args.test_mode,
                                       block_size=args.val_block_size,
                                       num_class=args.num_classes,
                                       random_num=args.random_num,
                                       use_bg=args.use_bg,
                                       data_split=args.data_split,
                                       test_use_pad=args.test_use_pad,
                                       pad_size=args.pad_size,
                                       use_paf=args.use_paf,
                                       cfg=self.val_cfg,
                                       args=args)

        self.len_block = val_dataset.test_len
        if 'test' in args.test_mode:
            self.data_shape = val_dataset.data_shape
            self.occupancy_map = val_dataset.occupancy_map
            self.gt_coords = val_dataset.gt_coords
            self.dir_name = val_dataset.dir_name

        val_dataloader1 = DataLoader(val_dataset,
                                     batch_size=args.val_batch_size,
                                     num_workers=1,
                                     shuffle=False,
                                     pin_memory=False)
        return val_dataloader1

    def _nms_v2(self, pred, kernel=3, mp_num=5, positions=None):
        args = self.args
        pred = torch.where(pred > 0.5, 1, 0)
        meanPool = nn.AvgPool3d(kernel, 1, kernel // 2).cuda()
        maxPool = nn.MaxPool3d(kernel, 1, kernel // 2).cuda()
        hmax = pred.clone().float()
        for _ in range(mp_num):
            hmax = meanPool(hmax)
        pred = hmax.clone()
        hmax = maxPool(hmax)
        keep = ((hmax == pred).float()) * ((pred > 0.1).float())
        coords = keep.nonzero()  # [N, 5]
        if coords.shape[0] > 2000:
            return torch.zeros([1, 5]).cuda()
        coords = coords[coords[:, 2] >= args.pad_size]
        coords = coords[coords[:, 2] < args.block_size - args.pad_size]
        coords = coords[coords[:, 3] >= args.pad_size]
        coords = coords[coords[:, 3] < args.block_size - args.pad_size]
        coords = coords[coords[:, 4] >= args.pad_size]
        coords = coords[coords[:, 4] < args.block_size - args.pad_size]

        try:
            h_val = torch.cat(
                [hmax[item[0], item[1], item[2], item[3]:item[3] + 1, item[4]:item[4] + 1] for item in
                 coords], dim=0)
            leftTop_coords = positions[coords[:, 0]] - (args.block_size // 2) - args.pad_size
            coords[:, 2:5] = coords[:, 2:5] + leftTop_coords

            pred_final = torch.cat(
                [coords[:, 1:2] + 1, coords[:, 4:5], coords[:, 3:4], coords[:, 2:3], h_val],
                dim=1)

            return pred_final
        except:
            return torch.zeros([0, 5]).cuda()

    def configure_optimizers(self):
        args = self.args
        if args.optim == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=args.learning_rate,
                                        momentum=0.9, weight_decay=0.001
                                        )
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=args.learning_rate,
                                         betas=(0.9, 0.99)
                                         )
        elif args.optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=args.learning_rate,
                                          betas=(0.9, 0.99),
                                          weight_decay=args.weight_decay
                                          )

        if args.scheduler == 'OneCycleLR':
            sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=args.learning_rate,
                                                        total_steps=args.max_epoch,
                                                        pct_start=0.1,
                                                        anneal_strategy='cos',
                                                        div_factor=30,
                                                        final_div_factor=100)
            lr_dict = {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1
            }

        if args.scheduler is None or args.scheduler == 'None':
            return [optimizer]
        else:
            return [optimizer], [lr_dict]


def train_func(args, stdout=None):
    if stdout is not None:
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stdout

    if getattr(args, "train_seed", None) is not None:
        pl.seed_everything(args.train_seed, workers=True)

    args.pad_size = args.pad_size[0]
    if 'test' in args.test_mode:
        val_loss_callback = ModelCheckpoint(save_top_k=1,
                                              monitor=f'cls_pr_alpha{args.prf1_alpha:.1f}' if args.num_classes == 1 else 'cls_f1',
                                              mode='max')
    else:
        val_loss_callback = ModelCheckpoint(save_top_k=5,
                                              monitor='val_loss',
                                              mode='min',
                                              filename='{epoch:04d}_{val_loss:.6f}_{val_f1:.6f}'
                                              )
    
    val_f4_loss_callback = ModelCheckpoint(save_top_k=5,
                                              monitor='val_f4_loss',
                                              mode='min',
                                              filename='{epoch:04d}_{val_f4_loss:.6f}'
                                              )

    model = UNetExperiment(args)
    logger_name = "{}_{}_BlockSize{}_{}Loss_MaxEpoch{}_bs{}_lr{}_IP{}_bg{}_coord{}_Softmax{}_{}_{}_TN{}".format(
        model.train_cfg["dset_name"], args.network, args.block_size, args.loss_func_seg, args.max_epoch,
        args.batch_size,
        args.learning_rate,
        int(args.use_IP), int(args.use_bg), int(args.use_coord),
        int(args.use_softmax), args.norm, args.others, args.sel_train_num)

    os.makedirs(f"{model.train_cfg['base_path']}/runs/{model.train_cfg['dset_name']}", exist_ok=True)
    tb_logger = loggers.TensorBoardLogger(f"{model.train_cfg['base_path']}/runs/{model.train_cfg['dset_name']}",
                                          name=logger_name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    latest_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch_idx",
        mode='max',
        filename='latest-{epoch:04d}',
    )
    
    early_stopping = EarlyStopping(
        monitor=args.early_stop_on if args.early_stop_on is not None else 'val_f4_loss',
        mode="min",
        patience=max(1, (args.max_epoch // 5) // args.check_val_every_n_epoch),
        min_delta=0.0,
        verbose=True,
    )

    # PL1.1 (DeepETPicker default)
    # runner = Trainer(min_epochs=min(50, args.max_epoch),
    #                  max_epochs=args.max_epoch,
    #                  logger=tb_logger,
    #                  gpus=-1,
    #                  checkpoint_callback=checkpoint_callback,
    #                  callbacks=[lr_monitor, latest_checkpoint],
    #                  accelerator='ddp',
    #                  precision=32,
    #                  #profiler=True,
    #                  sync_batchnorm=False,
    #                  resume_from_checkpoint=args.resume_from_checkpoint,
    #                  num_sanity_val_steps=2,
    #                  check_val_every_n_epoch=args.check_val_every_n_epoch,
    #                 )

    # #runner.validate(model)
    # runner.fit(model)
    
    # PL2.x
    # - gpus -> devices
    # - accelerator='ddp' -> strategy='ddp' + accelerator='gpu'
    # - checkpoint_callback=... -> include it in callbacks
    # - resume_from_checkpoint -> pass ckpt_path to fit()
    # - precision can stay (32 / "16-mixed" / "bf16-mixed")

    runner = Trainer(
        min_epochs=min(50, args.max_epoch),
        max_epochs=args.max_epoch,
        logger=tb_logger,

        accelerator="gpu",
        devices="auto",          # uses all visible GPUs; or set devices=2 etc.
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        
        precision=32,
        sync_batchnorm=False,

        callbacks=[lr_monitor, latest_checkpoint, val_loss_callback, val_f4_loss_callback, early_stopping],

        num_sanity_val_steps=0,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accumulate_grad_batches=2 if torch.cuda.device_count() == 1 else 1,
    )

    # resume_from_checkpoint is gone; use ckpt_path in fit()
    runner.fit(model, ckpt_path=args.resume_from_checkpoint or None)
    
    
    print('*' * 100)
    print('Training Finished')
    print(f'Training pid:{os.getpid()}')
    print('*' * 100)
    torch.cuda.empty_cache()
    if stdout is not None:
        sys.stderr = save_stderr
        sys.stdout = save_stdout
    return os.getpid()
        # torch.cuda.empty_cache()
        # if stdout is not None:
        #     stdout.flush()
        #     stdout.write('Training Exception!')
        #     sys.stderr = save_stderr
        #     sys.stdout = save_stdout
        # return os.getpid()
