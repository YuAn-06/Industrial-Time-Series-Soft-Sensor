import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class TCVAECausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]

        with torch.no_grad():
            mask_enc = torch.zeros(mask_shape, dtype=torch.bool).to(device) 
            mask_dec = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            self._mask = torch.cat([mask_enc, mask_dec], dim=-1)
            
            self._mask_enc = mask_enc
    @property
    def mask(self):
        return self._mask
    @property
    def mask_enc(self):
        return self._mask_enc

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
