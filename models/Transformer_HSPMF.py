"""
HSPMF-Enhanced Transformer (Fixed for AMP)
"""

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.HSPMF import HSPMFDecoder


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
            self.freq_dim = 2 * self.n_fourier + 1

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
            )

            self.freq_proj = nn.Linear(
                configs.d_model, configs.c_out * 2 * self.freq_dim
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

    def _hspmf_forward(self, hidden):
        """HSPMF 解码（带 AMP 保护）"""
        B, T, _ = hidden.shape

        # 投影到频域
        freq_real = self.freq_proj(hidden)
        freq_real = freq_real.view(B, T, self.c_out, 2 * self.freq_dim)

        real = freq_real[..., : self.freq_dim]
        imag = freq_real[..., self.freq_dim :]
        freq_complex = torch.complex(real, imag)

        freq_flat = freq_complex.reshape(-1, self.freq_dim)

        # 禁用 AMP，强制 float32
        with torch.cuda.amp.autocast(enabled=False):
            x_hat, _ = self.hspmf_decoder(freq_flat.to(torch.complex64))

        return x_hat.view(B, T, self.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        if self.use_hspmf:
            return self._hspmf_forward(dec_out)
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
