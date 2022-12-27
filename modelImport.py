from argparse import Namespace
import os

class Model():
    def __init__(self,model_type,weights,device_type):
        self.model_type=model_type
        if model_type=="YOLOv7":
            import os
            if not(str(os.getcwd()).endswith("/YOLOv7")):
                os.chdir("./YOLOv7")
            from utils.torch_utils import select_device as YOLO7_select_device
            self.device_type=device_type
            self.device = YOLO7_select_device(device_type)
            self.weights=weights
            self.best=""
            self.last=""
    
    def predict(self,image,options={}):
        if self.model_type=="YOLOv7":
              
            from detect import detect as YOLO7_detect
            opt = Namespace(agnostic_nms=False, 
                            augment=False, 
                            classes=None, 
                            conf_thres=0.25, 
                            device=self.device_type, 
                            exist_ok=False, 
                            img_size=640, 
                            iou_thres=0.45,
                            weights=self.weights, 
                            name='exp', 
                            no_trace=False, 
                            nosave=False, 
                            project='runs/detect', 
                            save_conf=True, 
                            save_txt=True,
                            return_txt=True, 
                            source=image, 
                            update=False, 
                            view_img=False)
            opt_dict=vars(opt)
            for i in options:
                if i=="weights":
                    if options[i]=="best":
                        opt_dict[i]=self.best
                    elif options[i]=="last":
                        opt_dict[i]=self.last
                    else:
                        opt_dict[i]=self.last
                else:
                    opt_dict[i]=options[i]

            values=YOLO7_detect(opt)
            return values
    
    def train(self,options={}):
        if self.model_type=="YOLOv7":
            from train import train as YOLO7_train
            import argparse
            import os
            import random
            import time
            from pathlib import Path

            import numpy as np
            import torch.distributed as dist
            import torch.utils.data
            import yaml
            from torch.utils.tensorboard import SummaryWriter
            from utils.general import increment_path, fitness, get_latest_run, check_file, set_logging, colorstr
            from utils.torch_utils import select_device
            import logging

            logger = logging.getLogger(__name__)
            print(os.getcwd())
            opt=Namespace(weights='yolo7.pt',
                              cfg='',
                              data='data/coco.yaml',
                              hyp='data/hyp.scratch.p5.yaml',
                              epochs=300,
                              batch_size=16,
                              img_size=[640, 640],
                              rect=False,
                              resume=False,
                              nosave=False,
                              notest=False,
                              noautoanchor=False,
                              evolve=False,
                              bucket='',
                              cache_images=False,
                              image_weights=False,
                              device='0',
                              multi_scale=False,
                              single_cls=False,
                              adam=False,
                              sync_bn=False,
                              local_rank=-1,
                              workers=8,
                              project='runs/train',
                              entity=None,
                              name='exp',
                              exist_ok=False,
                              quad=False,
                              linear_lr=False,
                              label_smoothing=0.0,
                              upload_dataset=False,
                              bbox_interval=-1,
                              save_period=-1,
                              artifact_alias="latest",
                              freeze=[0],
                              v5_metric=False)




            opt_dict=vars(opt)
            for i in options:
                if i=="weights":
                    if options[i]=="best":
                        opt_dict[i]=self.best
                    elif options[i]=="last":
                        opt_dict[i]=self.last
                    else:
                        opt_dict[i]=self.last
                else:
                    opt_dict[i]=options[i]
            

            # Set DDP variables
            opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
            opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
            set_logging(opt.global_rank)

            # Resume
            if opt.resume:  # resume an interrupted run
                ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
                assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
                apriori = opt.global_rank, opt.local_rank
                with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
                    opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
                opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
                logger.info('Resuming training from %s' % ckpt)
            else:
                # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
                opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
                assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
                opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
                opt.name = 'evolve' if opt.evolve else opt.name
                opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

            # DDP mode
            opt.total_batch_size = opt.batch_size
            device = select_device(opt.device, batch_size=opt.batch_size)
            if opt.local_rank != -1:
                assert torch.cuda.device_count() > opt.local_rank
                torch.cuda.set_device(opt.local_rank)
                device = torch.device('cuda', opt.local_rank)
                dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
                assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
                opt.batch_size = opt.total_batch_size // opt.world_size

            # Hyperparameters
            with open(opt.hyp) as f:
                hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

            # Train
            logger.info(opt)
            if not opt.evolve:
                tb_writer = None  # init loggers
                if opt.global_rank in [-1, 0]:
                    prefix = colorstr('tensorboard: ')
                    print("hi")
                    logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
                    b_writer = SummaryWriter(opt.save_dir)  # Tensorboard
                results,self.last,self.best=YOLO7_train(hyp, opt, device, logger, tb_writer)

            # Evolve hyperparameters (optional)
            else:
            # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
                meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                    'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                    'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                    'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                    'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                    'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                    'box': (1, 0.02, 0.2),  # box loss gain
                    'cls': (1, 0.2, 4.0),  # cls loss gain
                    'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                    'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                    'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                    'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                    'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                    'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                    'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                    'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                    'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                    'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                    'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                    'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                    'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                    'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                    'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                    'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                    'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                    'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                    'mixup': (1, 0.0, 1.0),   # image mixup (probability)
                    'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                    'paste_in': (1, 0.0, 1.0)}    # segment copy-paste (probability)
        
                with open(opt.hyp, errors='ignore') as f:
                    hyp = yaml.safe_load(f)  # load hyps dict
                    if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                        hyp['anchors'] = 3
                
                assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
                opt.notest, opt.nosave = True, True  # only test/save final epoch
                # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
                yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
                if opt.bucket:
                    os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

                for _ in range(300):  # generations to evolve
                    if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                        # Select parent(s)
                        parent = 'single'  # parent selection method: 'single' or 'weighted'
                        x = np.loadtxt('evolve.txt', ndmin=2)
                        n = min(5, len(x))  # number of previous results to consider
                        x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                        w = fitness(x) - fitness(x).min()  # weights
                        if parent == 'single' or len(x) == 1:
                            # x = x[random.randint(0, n - 1)]  # random selection
                            x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                        elif parent == 'weighted':
                            x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                        # Mutate
                        mp, s = 0.8, 0.2  # mutation probability, sigma
                        npr = np.random
                        npr.seed(int(time.time()))
                        g = np.array([x[0] for x in meta.values()])  # gains 0-1
                        ng = len(meta)
                        v = np.ones(ng)
                        while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                            v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                        for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                            hyp[k] = float(x[i + 7] * v[i])  # mutate

                    # Constrain to limits
                    for k, v in meta.items():
                        hyp[k] = max(hyp[k], v[1])  # lower limit
                        hyp[k] = min(hyp[k], v[2])  # upper limit
                        hyp[k] = round(hyp[k], 5)  # significant digits

                    # Train mutation
                    results,self.last,self.best = YOLO7_train(hyp.copy(), opt, device, logger)