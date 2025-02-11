import copy
import os
import pickle
from pathlib import Path
from functools import partial
import jittor as jt
jt.flags.use_cuda=1
from jittor import nn
import numpy as np
from jittor.einops import reduce
from p_tqdm import p_map
from dataset.py3dtojittor import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle)
from tqdm import tqdm

from dataset.quaternion import ax_from_6v, quat_slerp
from vis import skeleton_render

from .utils import extract, make_beta_schedule

def identity(t, *args, **kwargs):
    return t

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def smooth_poses(poses, window_size):
    _, num_frames, num_joints, _ = poses.shape

    smoothed_poses = jt.zeros_like(poses)
    
    for joint in range(num_joints):
        if joint == 0:
            continue
        if joint > 18:
            continue 
        for frame in range(num_frames):
            start_idx = max(0, frame - window_size)
            end_idx = min(num_frames, frame + window_size + 1)
            window_poses = poses[:, start_idx:end_idx, joint, :]
            average_pose = jt.mean(window_poses, dim=1)
            smoothed_poses[:, frame, joint, :] = average_pose
            
    return smoothed_poses

def smooth_root_pos(root_pos, window_size):
    _, num_frames, _ = root_pos.shape
    smoothed_root_pos = jt.zeros_like(root_pos)
    
    for frame in range(num_frames):
        start_idx = max(0, frame - window_size)
        end_idx = min(num_frames, frame + window_size + 1)
        window_pos = root_pos[:, start_idx:end_idx, :]
        average_pos = jt.mean(window_pos, dim=1)
        smoothed_root_pos[:, frame, :] = average_pos
        
    return smoothed_root_pos


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        smpl,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.ema = EMA(0.9999)
        

        self.cond_drop_prob = cond_drop_prob

        # make a SMPL instance for FK module
        self.smpl = smpl


        betas = jt.array( 
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = jt.cumprod(alphas, dim=0)  
        alphas_cumprod_prev = jt.concat([jt.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon


        self.betas=jt.array(betas)
        self.betas.stop_grad()
        self.alphas_cumprod=jt.array(alphas_cumprod)
        self.alphas_cumprod.stop_grad()
        self.alphas_cumprod_prev=jt.array(alphas_cumprod_prev)
        self.alphas_cumprod_prev.stop_grad()

        self.guidance_weight = guidance_weight

        self.sqrt_alphas_cumprod=jt.array(jt.sqrt(alphas_cumprod))
        self.sqrt_alphas_cumprod.stop_grad()


        self.sqrt_one_minus_alphas_cumprod=jt.array(jt.sqrt(1.0 - alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod.stop_grad()
 
        self.log_one_minus_alphas_cumprod=jt.array(jt.log(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod.stop_grad()

        self.sqrt_recip_alphas_cumprod=jt.array(jt.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recip_alphas_cumprod.stop_grad()

        self.sqrt_recipm1_alphas_cumprod=jt.array(jt.sqrt(1.0 / alphas_cumprod - 1))
        self.sqrt_recipm1_alphas_cumprod.stop_grad()

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.posterior_variance=jt.array(posterior_variance)
        self.posterior_variance.stop_grad()

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped=jt.array(jt.log(jt.clamp(posterior_variance, min_v=1e-20)))
        self.posterior_log_variance_clipped.stop_grad()

        self.posterior_mean_coef1=jt.array(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef1.stop_grad()

        self.posterior_mean_coef2=jt.array((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2.stop_grad()

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0

        self.p2_loss_weight=jt.array((self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma)
        self.p2_loss_weight.stop_grad()

        ## get loss coefficients and initialize objective
        self.loss_fn = jt.nn.mse_loss if loss_type == "l2" else jt.nn.L1Loss()


    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, cond, motionclip_features, beta, t, weight=None, clip_x_start = False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond, motionclip_features, beta, t, weight)
        maybe_clip = partial(jt.clamp, min_v=-1., max_v=1.) if clip_x_start else identity
        
        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, motionclip_features, beta, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, motionclip_features, beta, t, weight)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @jt.no_grad()
    def p_sample(self, x, cond, motionclip_features, beta, t):
        b, *_, = x.shape
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, motionclip_features=motionclip_features, beta=beta, t=t
        )
        noise = jt.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        print("model_mean + nonzero_mask * (0.5 * model_log_variance)", model_mean + nonzero_mask * (0.5 * model_log_variance))
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        print("model_mean + nonzero_mask * (0.5 * model_log_variance)exp", model_mean + nonzero_mask * (0.5 * model_log_variance).exp())
        return x_out, x_start

    @jt.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = jt.randn(shape) if noise is None else noise
        cond = cond

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = jt.full((batch_size,), i, dtype=jt.int32)
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        
    #@torch.no_grad()
    @jt.no_grad()
    def ddim_sample(self, shape, cond, motionclip_features, beta, **kwargs):
        batch = shape[0]
        total_timesteps, sampling_timesteps, eta = self.n_timestep, 50, 1

        times = jt.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = jt.randn(shape)
        cond = cond
        beta = beta
        motionclip_features = motionclip_features

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = jt.full((batch,), time, dtype=jt.int32)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, motionclip_features, beta, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = jt.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        return x
    

    @jt.no_grad()
    def long_ddim_sample(self, shape, cond, motionclip_features, beta, **kwargs):
        batch, total_timesteps, sampling_timesteps, eta = shape[0], self.n_timestep, 50, 1

        if batch == 1:
            return self.ddim_sample(shape, cond, motionclip_features, beta)

        times = jt.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]


        x = jt.randn(shape)

        cond = cond
        beta = beta
        motionclip_features = motionclip_features
        
        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        x_start = None

        for time, time_next, weight in tqdm(time_pairs, desc = 'sampling loop time step'):

            time_cond = jt.full((batch,), time, dtype=jt.int32)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, motionclip_features, beta, time_cond, weight=weight, clip_x_start = self.clip_denoised) 

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = jt.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if time > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:]
        return x


    @jt.no_grad()
    def inpaint_loop(
        self,
        shape,
        cond,
        motionclip_features,
        beta,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):

        batch_size = shape[0]
        x = jt.randn(shape) if noise is None else noise
        cond = cond
        motionclip_features = motionclip_features
        beta = beta
        if return_diffusion:
            diffusion = [x]


        mask = constraint["mask"]  
        value = constraint["value"]

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = jt.full((batch_size,), i, dtype=jt.int32)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, motionclip_features, beta, timesteps)
            # enforce constraint between each denoising step
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @jt.no_grad()
    def long_inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        

        batch_size = shape[0]
        x = jt.randn(shape) if noise is None else noise
        cond = cond  
        if return_diffusion:
            diffusion = [x]

        assert x.shape[1] % 2 == 0
        if batch_size == 1:
            # there's no continuation to do, just do normal
            return self.p_sample_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
            )
        assert batch_size > 1
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = jt.full((batch_size,), i, dtype=jt.int32)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            if i > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:] 

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @jt.no_grad()
    def conditional_sample(
        self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
            conditions : [ (time, state), ... ]
        """
        #device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = jt.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, beta, joint_offset, motionclip_features, t):
        noise = jt.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)


        x_recon = self.model(x_noisy, cond, motionclip_features, beta, t, cond_drop_prob=self.cond_drop_prob)
        assert noise.shape == x_recon.shape

        model_out = x_recon  
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        # split off contact from the rest
        model_contact, model_out = jt.split(
            model_out, (4, model_out.shape[2] - 4), dim=2
        )
        target_contact, target = jt.split(target, (4, target.shape[2] - 4), dim=2)

        # velocity loss
        target_v = target[:, 1:] - target[:, :-1]
        model_out_v = (model_out[:, 1:] - model_out[:, :-1])  
        v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        # FK loss
        b, s, c = model_out.shape
        # unnormalize
        # X, Q
        model_x = model_out[:, :, :3]
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))

        # perform FK
        model_xp = self.smpl.execute(model_q, model_x, joint_offset)
        target_xp = self.smpl.execute(target_q, target_x, joint_offset)


        fk_loss = self.loss_fn(model_xp, target_xp,reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)

        # foot skate loss
        foot_idx = [7, 8, 10, 11]

        # find static indices consistent with model's own predictions
        static_idx = model_contact > 0.95  # N x S x 4
        model_feet = model_xp[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        model_foot_v = jt.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        expanded_static_idx = jt.unsqueeze(static_idx,-1)  # [64, 150, 4, 1]
        static_idx = jt.logical_not(static_idx)
        model_foot_v = jt.where(jt.logical_not(expanded_static_idx),jt.zeros_like(model_foot_v),model_foot_v)
        foot_loss = self.loss_fn(
            model_foot_v, jt.zeros_like(model_foot_v),reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")

        losses = (
            0.636 * jt.mean(loss),
            2.964 * jt.mean(v_loss),
            0.646 * jt.mean(fk_loss),
            10.942 * jt.mean(foot_loss),
        )
        return sum(losses), losses

    def loss(self, x, cond, beta, joint_offset, motionclip_features, t_override=None):
        batch_size = len(x)
        if t_override is None:
            t = jt.randint(0, self.n_timestep, (batch_size,), dtype=jt.int32)
        else:
            t = jt.full((batch_size,), t_override, dtype=jt.int32)
        return self.p_losses(x, cond, beta, joint_offset, motionclip_features, t)

    def execute(self, x, cond, beta, joint_offset, motionclip_features, t_override=None):
        return self.loss(x, cond, beta, joint_offset, motionclip_features, t_override)


    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = jt.full((batch_size,), timestep, dtype=jt.int32)
        return self.q_sample(x, t) if timestep > 0 else x

    def render_sample(
        self,
        shape,
        cond,
        beta,
        joint_offset,
        motionclip_features,
        normalizer,
        epoch,
        render_out,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder="ood_sliced",
        start_point=None,
        render=True
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_loop
            elif mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long":
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"

            samples = (
                func_class(
                    shape,
                    cond,
                    motionclip_features,
                    beta,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
                .detach()
                .numpy()
            )
        else:
            samples = shape

        samples = normalizer.unnormalize(samples)

        if samples.shape[2] == 151:
            sample_contact, samples = jt.split(
                samples, (4, samples.shape[2] - 4), dim=2
            )
        else:
            sample_contact = None
        # do the FK all at once
        b, s, c = samples.shape
        pos = samples[:, :, :3]
        q = samples[:, :, 3:].reshape(b, s, 24, 6)
        # go 6d to ax
        q = ax_from_6v(q)

        if mode == "long":
            b, s, c1, c2 = q.shape
            assert s % 2 == 0
            half = s // 2
            if b > 1:
                # if long mode, stitch position using linear interp

                fade_out = jt.ones((1, s, 1))
                fade_in = jt.ones((1, s, 1))
                fade_out[:, half:, :] = jt.linspace(1, 0, half)[None, :, None]
                fade_in[:, :half, :] = jt.linspace(0, 1, half)[None, :, None]

                pos[:-1] *= fade_out
                pos[1:] *= fade_in

                full_pos = jt.zeros((s + half * (b - 1), 3))
                idx = 0
                for pos_slice in pos:
                    full_pos[idx : idx + s] += pos_slice
                    idx += half

                # stitch joint angles with slerp
                slerp_weight = jt.linspace(0, 1, half)[None, :, None]

                left, right = q[:-1, half:], q[1:, :half]
                # convert to quat
                left, right = (
                    axis_angle_to_quaternion(left),
                    axis_angle_to_quaternion(right),
                )
                merged = quat_slerp(left, right, slerp_weight)  # (b-1) x half x ...
                # convert back
                merged = quaternion_to_axis_angle(merged)


                full_q = jt.zeros((s + half * (b - 1), c1, c2))
                full_q[:half] += q[0, :half]
                idx = half
                for q_slice in merged:
                    full_q[idx : idx + half] += q_slice
                    idx += half
                full_q[idx : idx + half] += q[-1, half:]

                # unsqueeze for fk
                full_pos = full_pos.unsqueeze(0)
                full_q = full_q.unsqueeze(0)
            else:
                full_pos = pos
                full_q = q

            full_pose = self.smpl.execute(full_q, full_pos, joint_offset).detach().numpy()
            # squeeze the batch dimension away and render
            skeleton_render(
                full_pose[0],
                epoch=f"{epoch}",
                out=render_out,
                name=name,
                sound=sound,
                stitch=True,
                sound_folder=sound_folder,
                render=render
            )
            if fk_out is not None:
                outname = f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.pkl'
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                pickle.dump(
                    {
                        "smpl_poses": full_q.squeeze(0).reshape((-1, 72)).numpy(),
                        "smpl_trans": full_pos.squeeze(0).numpy(),
                        "full_pose": full_pose[0],
                    },
                    open(os.path.join(fk_out, outname), "wb"),
                )
            return

        poses = self.smpl.execute(q, pos, joint_offset).detach().numpy()
        sample_contact = (
            sample_contact.detach().numpy()
            if sample_contact is not None
            else None
        )

        def inner(xx):
            num, pose = xx
            filename = name[num] if name is not None else None
            contact = sample_contact[num] if sample_contact is not None else None
            skeleton_render(
                pose,
                epoch=f"e{epoch}_b{num}",
                out=render_out,
                name=filename,
                sound=sound,
                contact=contact,
            )

        p_map(inner, enumerate(poses))


        if fk_out is not None and mode != "long":
            Path(fk_out).mkdir(parents=True, exist_ok=True)
            for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                path = os.path.normpath(filename)
                pathparts = path.split(os.sep)
                pathparts[-1] = pathparts[-1].replace("npy", "wav")
                # path is like "data/train/features/name"
                pathparts[2] = "wav_sliced"
                audioname = os.path.join(*pathparts)
                outname = f"{epoch}_{num}_{pathparts[-1][:-4]}.pkl"
                pickle.dump(
                    {
                        "smpl_poses": qq.reshape((-1, 72)).numpy(),
                        "smpl_trans": pos_.numpy(),
                        "full_pose": pose,
                    },
                    open(f"{fk_out}/{outname}", "wb"),
                )