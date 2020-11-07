from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from common import UV_LOOKUP_TABLE


class MapDensePoseTex:
    lut: Optional[np.ndarray] = None

    def densepose2tex(
        self, img: np.ndarray, iuv_img: np.ndarray, tex_res: int
    ) -> np.ndarray:
        if MapDensePoseTex.lut is None:
            MapDensePoseTex.lut = np.load(UV_LOOKUP_TABLE.as_posix())

        iuv_raw = iuv_img[iuv_img[:, :, 0] > 0]
        data = img[iuv_img[:, :, 0] > 0]
        i = iuv_raw[:, 0] - 1

        if iuv_raw.dtype == np.uint8:
            u = iuv_raw[:, 1] / 255.0
            v = iuv_raw[:, 2] / 255.0
        else:
            u = iuv_raw[:, 1]
            v = iuv_raw[:, 2]

        u[u > 1] = 1.0
        v[v > 1] = 1.0

        uv_smpl = MapDensePoseTex.lut[
            i.astype(np.int),
            np.round(v * 255.0).astype(np.int),
            np.round(u * 255.0).astype(np.int),
        ]

        tex = np.ones((tex_res, tex_res, img.shape[2])) * 0.5

        u_I = np.round(uv_smpl[:, 0] * (tex.shape[1] - 1)).astype(np.int32)
        v_I = np.round((1 - uv_smpl[:, 1]) * (tex.shape[0] - 1)).astype(
            np.int32
        )

        tex[v_I, u_I] = data

        return tex

    def tex2densepose(
        self, tex: np.ndarray, iuv_img: np.ndarray
    ) -> np.ndarray:
        if MapDensePoseTex.lut is None:
            MapDensePoseTex.lut = np.load(UV_LOOKUP_TABLE.as_posix()).astype(
                np.float32
            )

        iuv_raw = iuv_img[iuv_img[:, :, 0] > 0]
        i = iuv_raw[:, 0] - 1

        if iuv_raw.dtype == np.uint8:
            u = iuv_raw[:, 1].astype(np.float32) / 255.0
            v = iuv_raw[:, 2].astype(np.float32) / 255.0
        else:
            u = iuv_raw[:, 1]
            v = iuv_raw[:, 2]

        u[u > 1] = 1.0
        v[v > 1] = 1.0

        uv_smpl = MapDensePoseTex.lut[
            i.astype(np.int),
            np.round(v * 255.0).astype(np.int),
            np.round(u * 255.0).astype(np.int),
        ]

        u_I = np.round(uv_smpl[:, 0] * (tex.shape[1] - 1)).astype(np.int32)
        v_I = np.round((1 - uv_smpl[:, 1]) * (tex.shape[0] - 1)).astype(
            np.int32
        )

        height, width = iuv_img.shape[:-1]
        output_data = np.zeros((height, width, tex.shape[-1]), dtype=tex.dtype)
        output_data[iuv_img[:, :, 0] > 0] = tex[v_I, u_I]

        return output_data


class MapDensePoseTexModule(nn.Module):
    def __init__(self, tex_res: int) -> None:
        super().__init__()
        self.register_buffer(
            "lut",
            torch.from_numpy(np.load(UV_LOOKUP_TABLE.as_posix())).float(),
        )
        self.tex_res = tex_res

    def forward(
        self, img_or_tex: torch.Tensor, iuv_img: torch.Tensor
    ) -> torch.Tensor:
        return self.tex2densepose(img_or_tex, iuv_img)

    def tex2densepose(
        self, tex_batch: torch.Tensor, iuv_img_batch: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            tex_batch (torch.Tensor): Texture image in the form of
                B x C x H x W.
            iuv_img _batch(torch.Tensor): Byte tensor of the form
                B x 3 x H' x W'. The first channel describes class of the
                part. Next two are UV coordinates in [0, 255] range.

        Returns:
            torch.Tensor: Tensor with mapped pixels from the texture space
                into image space. Dimensions: B x C x H' x W'
        """
        output_img = []
        for tex, iuv_img in zip(tex_batch, iuv_img_batch):
            iuv_raw = (
                iuv_img.masked_select((iuv_img[0] > 0).unsqueeze(dim=0))
                .view((3, -1))
                .t()
            )

            i = iuv_raw[:, 0] - 1
            u = iuv_raw[:, 1].float() / 255.0
            v = iuv_raw[:, 2].float() / 255.0

            u = u.clamp(0.0, 1.0)
            v = v.clamp(0.0, 1.0)

            uv_smpl = self.lut[
                i.long(),
                torch.round(v * 255.0).long(),
                torch.round(u * 255.0).long(),
            ]

            u_I = torch.round(uv_smpl[:, 0] * (self.tex_res - 1)).long()
            v_I = torch.round((1 - uv_smpl[:, 1]) * (self.tex_res - 1)).long()
            output_data = (
                torch.zeros(
                    (tex.shape[0], iuv_img.shape[1], iuv_img.shape[2]),
                    requires_grad=True,
                )
                .float()
                .to(tex)
            )
            output_data = torch.masked_scatter(
                output_data,
                (iuv_img[[0]] > 0),
                tex[:, v_I, u_I],
            )
            output_img.append(output_data)

        return torch.stack(output_img, dim=0)
