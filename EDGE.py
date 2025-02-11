import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path


import jittor as jt
jt.flags.use_cuda=1
from jittor import nn
import wandb

from tqdm import tqdm
from jittor.dataset import DataLoader
from dataset.dance_dataset import AISTPPDataset
from dataset.preprocess import increment_path
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class EDGE:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=False,
        learning_rate=4e-5,
        weight_decay=0.02,
    ):
        super(EDGE, self).__init__()
        use_baseline_feats = feature_type == "baseline"

        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, 6dof
        self.repr_dim = repr_dim = pos_dim + rot_dim + 4

        feature_dim = 35 if use_baseline_feats else 4800

        horizon_seconds = 5
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = jt.load(
                checkpoint_path
            )
            self.normalizer = checkpoint["normalizer"]

        model = DanceDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=nn.gelu,
        )

        smpl = SMPLSkeleton()
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )


        print(f"Model has {sum(p.numpy().size for p in model.parameters())} parameters")


        self.model = model

        self.diffusion = diffusion

        optim = jt.optim.Adan(model.parameters(), lr=learning_rate,betas=(0.02, 0.08, 0.01),eps=1e-8,weight_decay=weight_decay)
        self.optim = optim

        if checkpoint_path != "":
            state_dict_key = "ema_state_dict" if EMA else "model_state_dict"
            self.model.load_parameters(checkpoint[state_dict_key])  # 直接加载模型参数


    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()



    def train_loop(self, opt):
        checkpoint_dir = os.path.join("runs", "train", opt.exp_name, "weights")
        checkpoint_path = os.path.join(checkpoint_dir, "train-450.pkl")

        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        print(f"train_tensor_dataset_path: {train_tensor_dataset_path}, exists: {os.path.isfile(train_tensor_dataset_path)}")
        print(f"test_tensor_dataset_path: {test_tensor_dataset_path}, exists: {os.path.isfile(test_tensor_dataset_path)}")
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            num_cpus = multiprocessing.cpu_count()

            train_dataset = AISTPPDataset(  
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            ).set_attrs(
                 batch_size=opt.batch_size, 
                 shuffle=True,
                 endless=False,
                 keep_numpy_array=True,
                 num_workers=0,
                 drop_last=True,
                 )

            test_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            ).set_attrs(
                 batch_size=opt.batch_size,
                 shuffle=True,
                 keep_numpy_array=True,
                 num_workers=0,
                 drop_last=True
                 )
            
            if jt.rank == 0:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        num_batches = len(train_dataset) // opt.batch_size  
        self.normalizer = test_dataset.normalizer
        num_cpus = multiprocessing.cpu_count()

        load_loop = (
            partial(tqdm, position=1, desc="Batch")
        ) if jt.rank == 0 else lambda x: x

        if jt.rank == 0:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(
                project=opt.wandb_pj_name,
                name=opt.exp_name,
            )
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        start_epoch = 0
        if os.path.isfile(checkpoint_path) and opt.continue_train:
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = jt.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.normalizer = checkpoint["normalizer"]
            start_epoch = 450

        else:
            print('No models!')

        for epoch in range(start_epoch+1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            # train
            self.train()
            for step, (x, cond, filename, wavnames, beta, joint_offset, motionclip_features) in enumerate(load_loop(train_dataset)):
                total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(
                    x, cond, beta, joint_offset, motionclip_features, t_override=None
                )

                self.optim.step(total_loss) 
                if jt.rank==0:
                    avg_loss += loss.detach().numpy()
                    avg_vloss += v_loss.detach().numpy()
                    avg_fkloss += fk_loss.detach().numpy()
                    avg_footloss += foot_loss.detach().numpy()

            # Save model
            if (epoch % opt.save_interval) == 0:  
                if jt.rank==0:
                    self.eval()
                    avg_loss /= num_batches   
                    avg_vloss /= num_batches
                    avg_fkloss /= num_batches
                    avg_footloss /= num_batches

                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                    }
                    wandb.log(log_dict)
                    ckpt = {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    jt.save(ckpt, os.path.join(wdir, f"train-{epoch}.pkl"))

                    # generate a sample
                    render_count = 2
                    shape = (render_count, self.horizon, self.repr_dim)

                    print("Generating Sample")
                    (x, cond, filename, wavnames, beta, joint_offset, motionclip_features) = next(iter(test_dataset))
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        beta[:render_count],
                        joint_offset[:render_count],
                        motionclip_features[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=wavnames[:render_count],
                        sound=True,
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        if jt.rank==0:
            wandb.run.finish()

    def render_sample(
        self, data_tuple, beta, motionclip_features, joint_offset, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        
        cond = jt.array(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            beta[:render_count],
            joint_offset[:render_count],
            motionclip_features[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render
        )