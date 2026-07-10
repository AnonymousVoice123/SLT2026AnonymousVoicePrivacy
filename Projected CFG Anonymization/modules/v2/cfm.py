# This file is adapted from Plachtaa/seed-vc.
# Original project: https://github.com/Plachtaa/seed-vc
# License: GNU General Public License v3.0
# Modifications: projected CFG anonymization and configurable projection modes.

import torch
from tqdm import tqdm
import torch.nn.functional as F
class CFM(torch.nn.Module):
    def __init__(
        self,
        estimator: torch.nn.Module,
    ):
        super().__init__()
        self.sigma_min = 1e-6
        self.estimator = estimator
        self.in_channels = estimator.in_channels
        self.criterion = torch.nn.L1Loss()
    
    @torch.inference_mode()
    def inference(self,
                  mu: torch.Tensor,
                  x_lens: torch.Tensor, 
                  prompt: torch.Tensor,
                  style: torch.Tensor,
                  n_timesteps=10,
                  temperature=1.0,
                  inference_cfg_rate=[0.5, 0.5],
                  random_voice=False,
                  projection_mode="smooth",
                  smooth_kernel=21,
                  ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            x_lens (torch.Tensor): length of each mel-spectrogram
                shape: (batch_size,)
            prompt (torch.Tensor): prompt
                shape: (batch_size, n_feats, prompt_len)
            style (torch.Tensor): style
                shape: (batch_size, style_dim)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            inference_cfg_rate (list or float, optional): projected CFG scales. In this project,
                inference_cfg_rate[0] is scale1 and inference_cfg_rate[1] is scale2.
            projection_mode (str): one of {"global", "token", "smooth"}.
            smooth_kernel (int): smoothing kernel size when projection_mode="smooth".

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(
            z,
            x_lens,
            prompt,
            mu,
            style,
            t_span,
            inference_cfg_rate=inference_cfg_rate,
            random_voice=random_voice,
            projection_mode=projection_mode,
            smooth_kernel=smooth_kernel,
        )
    
    @staticmethod
    def remove_projection(x, direction, eps=1e-6, mode="global", smooth_kernel=21):
        """
        Compute the projection of x onto direction.

        Args:
            x: [B, C, T]
            direction: [B, C, T]
            mode:
                global: one utterance-level direction.
                token: frame/token-level direction.
                smooth: sliding-window smoothed direction.
            smooth_kernel: kernel size for smooth mode.

        Returns:
            Projection component with shape [B, C, T].
        """
        if mode == "token":
            v = direction

        elif mode == "global":
            v = direction.mean(dim=2, keepdim=True)

        elif mode == "smooth":
            smooth_kernel = int(smooth_kernel)
            if smooth_kernel <= 1:
                v = direction
            else:
                # Make kernel odd so that output length stays aligned with input length.
                if smooth_kernel % 2 == 0:
                    smooth_kernel += 1
                pad = smooth_kernel // 2

                # reflect padding requires pad < sequence length; fall back to replicate for very short audio.
                pad_mode = "reflect" if direction.size(-1) > pad else "replicate"
                direction_pad = F.pad(direction, (pad, pad), mode=pad_mode)
                v = F.avg_pool1d(direction_pad, kernel_size=smooth_kernel, stride=1)

        else:
            raise ValueError(
                f"Unknown projection mode: {mode}. "
                "Valid choices are: global, token, smooth."
            )

        dot = (x * v).sum(dim=1, keepdim=True)
        denom = (v * v).sum(dim=1, keepdim=True).clamp_min(eps)
        return dot / denom * v
        
    def solve_euler(
            self,
            x,
            x_lens,
            prompt,
            mu,
            style,
            t_span,
            inference_cfg_rate=[0.5, 0.5],
            random_voice=False,
            projection_mode="smooth",
            smooth_kernel=21,
    ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            x_lens (torch.Tensor): length of each mel-spectrogram
                shape: (batch_size,)
            prompt (torch.Tensor): prompt
                shape: (batch_size, n_feats, prompt_len)
            style (torch.Tensor): style
                shape: (batch_size, style_dim)
            inference_cfg_rate (list or float, optional): projected CFG scales. In this project,
                inference_cfg_rate[0] is scale1 and inference_cfg_rate[1] is scale2.
            projection_mode (str): one of {"global", "token", "smooth"}.
            smooth_kernel (int): smoothing kernel size when projection_mode="smooth".
            sway_sampling (bool, optional): Sway sampling. Defaults to False.
            amo_sampling (bool, optional): AMO sampling. Defaults to False.
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # Normalize CFG rates. This keeps backward compatibility with a scalar CFG value.
        if isinstance(inference_cfg_rate, (float, int)):
            inference_cfg_rate = [float(inference_cfg_rate), float(inference_cfg_rate)]
        elif isinstance(inference_cfg_rate, tuple):
            inference_cfg_rate = list(inference_cfg_rate)
        elif isinstance(inference_cfg_rate, torch.Tensor):
            inference_cfg_rate = inference_cfg_rate.detach().cpu().flatten().tolist()
        if len(inference_cfg_rate) == 1:
            inference_cfg_rate = [float(inference_cfg_rate[0]), float(inference_cfg_rate[0])]
        if len(inference_cfg_rate) < 2:
            raise ValueError("inference_cfg_rate must provide scale1 and scale2")

        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        for step in tqdm(range(1, len(t_span))):
            if random_voice:
                cfg_dphi_dt = self.estimator(
                    torch.cat([x, x], dim=0),
                    torch.cat([torch.zeros_like(prompt_x), torch.zeros_like(prompt_x)], dim=0),
                    torch.cat([x_lens, x_lens], dim=0),
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([torch.zeros_like(style), torch.zeros_like(style)], dim=0),
                    torch.cat([mu, torch.zeros_like(mu)], dim=0),
                )
                cond_txt, uncond = cfg_dphi_dt[0:1], cfg_dphi_dt[1:2]
                dphi_dt = ((1.0 + inference_cfg_rate[0]) * cond_txt - inference_cfg_rate[0] * uncond)
            else:
                cfg_dphi_dt = self.estimator(
                    torch.cat([x, x, x], dim=0),
                    torch.cat([prompt_x, torch.zeros_like(prompt_x), torch.zeros_like(prompt_x)], dim=0),
                    torch.cat([x_lens, x_lens, x_lens], dim=0),
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([style, torch.zeros_like(style), torch.zeros_like(style)], dim=0),
                    torch.cat([mu, mu, torch.zeros_like(mu)], dim=0),
                )

                cond_txt_spk, cond_txt, uncond = (
                    cfg_dphi_dt[0:1],
                    cfg_dphi_dt[1:2],
                    cfg_dphi_dt[2:3],
                )
                scale_1 = inference_cfg_rate[0]  
                scale_2 = inference_cfg_rate[1]
                
                speaker_dir = (cond_txt_spk - cond_txt).detach()
                content_dir = cond_txt - uncond

                speaker_proj = CFM.remove_projection(
                    content_dir,
                    speaker_dir,
                    mode=projection_mode,
                    smooth_kernel=smooth_kernel,
                )
             
                dphi_dt = cond_txt- scale_1 * speaker_proj + scale_2 * content_dir

            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0
        return x
    
    
        
    def forward(self, x1, x_lens, prompt_lens, mu, style):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            # range covered by prompt are set to 0
            y[bib, :, :prompt_lens[bib]] = 0

        estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(), style, mu)
        loss = 0
        for bib in range(b):
            loss += self.criterion(estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]], u[bib, :, prompt_lens[bib]:x_lens[bib]])
        loss /= b

        return loss
