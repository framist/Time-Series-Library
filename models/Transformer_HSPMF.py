"""
HSPMF-Enhanced Transformer

说明：
- 仅在 HSPMF 路线下使用频域输出头
- 预测正频率部分，再显式重建共轭对称频谱
- Forecast 路径可返回后验辅助量，供 End2End-NLL 训练使用
"""

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.HSPMF import HSPMFDecoder, build_conjugate_symmetric_y


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        self.use_hspmf = getattr(configs, "use_hspmf", False)
        if self.use_hspmf:
            self.n_fourier = getattr(configs, "hspmf_n_fourier", 16)
            self.freq_pos_dim = self.n_fourier
            self.freq_full_dim = 2 * self.n_fourier + 1

            x_range = getattr(configs, "hspmf_x_range", None)
            if x_range is None:
                x_lo, x_hi = -6.0, 6.0
            else:
                x_lo, x_hi = x_range

            self.hspmf_decoder = HSPMFDecoder(
                n_fourier=self.n_fourier,
                period=getattr(configs, "hspmf_period", 1.0),
                x_lo=x_lo,
                x_hi=x_hi,
                grid_size=getattr(configs, "hspmf_grid_size", 64),
                beta=getattr(configs, "hspmf_beta", 1.0),
                tau=getattr(configs, "hspmf_tau", 1.0),
                score_mode=getattr(configs, "hspmf_score_mode", "abs2"),
                hier_levels=getattr(configs, "hspmf_hier_levels", None),
                learn_beta=getattr(configs, "hspmf_learn_beta", False),
            )

            self.freq_proj = nn.Linear(
                configs.d_model, configs.c_out * 2 * self.freq_pos_dim
            )
        else:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # Transformer components
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            wv_cfg=configs,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            self.dec_embedding = DataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
                wv_cfg=configs,
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(
                                True,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        AttentionLayer(
                            FullAttention(
                                False,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for _ in range(configs.d_layers)
                ],
                norm_layer=nn.LayerNorm(configs.d_model),
                projection=None,
            )

    def _hspmf_decode(self, hidden):
        """HSPMF 解码（带 AMP 保护），返回点预测与后验辅助量。"""
        B, T, _ = hidden.shape

        # 只预测正频率部分，再显式重建完整的共轭对称频谱。
        freq_real = self.freq_proj(hidden)
        freq_real = freq_real.view(B, T, self.c_out, 2 * self.freq_pos_dim)

        # 复数解码路径统一切回 float32，避免 AMP 下的复数半精度不稳定。
        real = freq_real[..., : self.freq_pos_dim].float()
        imag = freq_real[..., self.freq_pos_dim :].float()
        y_pos = torch.complex(real, imag)
        y_full = build_conjugate_symmetric_y(y_pos, y0=1.0 + 0.0j)
        freq_flat = y_full.reshape(-1, self.freq_full_dim)

        with torch.cuda.amp.autocast(enabled=False):
            x_hat, posterior = self.hspmf_decoder(freq_flat.to(torch.complex64))

        x_hat = x_hat.view(B, T, self.c_out)
        posterior = posterior.view(B, T, self.c_out, -1)
        aux = {
            "posterior": posterior,
            "y_pos": y_pos,
            "y_full": y_full,
        }
        return x_hat, aux

    def _hspmf_forward(self, hidden, return_aux=False):
        x_hat, aux = self._hspmf_decode(hidden)
        if return_aux:
            aux["pred"] = x_hat
            return aux
        return x_hat

    def get_hspmf_beta(self):
        if not self.use_hspmf:
            return None
        return self.hspmf_decoder.beta_value.detach()

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        if self.use_hspmf:
            return self._hspmf_forward(dec_out, return_aux=True)
        return self.projection(dec_out)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        if self.use_hspmf:
            return self._hspmf_forward(enc_out)
        return self.projection(enc_out)

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        if self.use_hspmf:
            return self._hspmf_forward(enc_out)
        return self.projection(enc_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == "imputation":
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        elif self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
