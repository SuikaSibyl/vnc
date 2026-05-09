import torch
import numpy as np
import nvdiffrast.torch as dr
from .render import persp_proj, NVDRenderer


def _sh_basis(dirs, L):
    """
    Evaluate real SH basis functions Y_lm up to order L at unit directions.

    Parameters
    ----------
    dirs : torch.Tensor
        Unit directions, shape [..., 3].  xyz convention: x=right, y=up, z=front.
    L : int
        Maximum SH order (inclusive).

    Returns
    -------
    torch.Tensor
        Shape [..., (L+1)^2].  Bands are concatenated in order l=0,1,...,L.
    """
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    out = []

    # l=0
    out.append(torch.full_like(x, 0.282095))

    if L >= 1:
        out.append(0.488603 * y)
        out.append(0.488603 * z)
        out.append(0.488603 * x)

    if L >= 2:
        out.append(1.092548 * x * y)
        out.append(1.092548 * y * z)
        out.append(0.315392 * (2*z*z - x*x - y*y))
        out.append(1.092548 * x * z)
        out.append(0.546274 * (x*x - y*y))

    if L >= 3:
        out.append(0.590044 * y * (3*x*x - y*y))
        out.append(2.890611 * x * y * z)
        out.append(0.457046 * y * (4*z*z - x*x - y*y))
        out.append(0.373176 * z * (2*z*z - 3*x*x - 3*y*y))
        out.append(0.457046 * x * (4*z*z - x*x - y*y))
        out.append(1.445306 * z * (x*x - y*y))
        out.append(0.590044 * x * (x*x - 3*y*y))

    if L >= 4:
        out.append(2.503343 * x * y * (x*x - y*y))
        out.append(1.770131 * y * z * (3*x*x - y*y))
        out.append(0.946175 * x * y * (7*z*z - 1))
        out.append(0.669047 * y * z * (7*z*z - 3))
        out.append(0.105786 * (35*z**4 - 30*z*z + 3))
        out.append(0.669047 * x * z * (7*z*z - 3))
        out.append(0.473087 * (x*x - y*y) * (7*z*z - 1))
        out.append(1.770131 * x * z * (x*x - 3*y*y))
        out.append(0.625836 * (x**4 - 6*x*x*y*y + y**4))

    if L > 4:
        raise NotImplementedError("SH order > 4 not implemented analytically here; use a library.")

    return torch.stack(out, dim=-1)


def _sh_order_for_coeff(i):
    """Return l given flat index i = l^2 + l + m."""
    return int(np.floor(np.sqrt(i)))


def _vmf_sh_weights(roughness, L, device):
    """
    Per-band filter weights for a von Mises-Fisher lobe.

    a_l = exp(-l(l+1) / (2 * kappa^2))

    roughness=0 -> kappa=inf -> delta (mirror)
    roughness=1 -> kappa->0  -> uniform (diffuse)

    Parameters
    ----------
    roughness : torch.Tensor or float
        Scalar roughness in [0, 1].
    L : int
        Maximum SH order.
    device : torch.device

    Returns
    -------
    torch.Tensor
        Shape [(L+1)^2], one weight per SH coefficient.
    """
    if not isinstance(roughness, torch.Tensor):
        roughness = torch.tensor(roughness, dtype=torch.float32, device=device)
    kappa = (1.0 - roughness) / (roughness + 1e-6) * 10.0
    l_idx = torch.tensor([_sh_order_for_coeff(i) for i in range((L+1)**2)],
                         dtype=torch.float32, device=device)
    weights = torch.exp(-l_idx * (l_idx + 1.0) / (2.0 * kappa * kappa))
    return weights  # [(L+1)^2]


class SphericalHarmonicsHO:
    """
    Higher-order spherical harmonics environment map, supporting a differentiable
    roughness parameter via vMF lobe filtering.

    Diffuse (Lambertian) is the special case roughness=1.
    Mirror is the limit roughness=0.
    """

    def __init__(self, envmap, L=4):
        """
        Project the environment map into SH coefficients up to order L.

        Parameters
        ----------
        envmap : torch.Tensor
            HDR environment map, shape [H, W, C] (C=3 or 4).
        L : int
            Maximum SH order.  4 gives 25 coefficients and captures mid-frequency
            glossy.  Increase to 6/8 for sharper highlights (at the cost of
            needing more SH evaluation terms).
        """
        self.L = L
        h, w = envmap.shape[:2]

        theta = torch.linspace(0, np.pi,     h, device='cuda').unsqueeze(1).expand(h, w)
        phi   = torch.linspace(3*np.pi, np.pi, w, device='cuda').unsqueeze(0).expand(h, w)

        sin_theta = torch.sin(theta)
        x =  sin_theta * torch.cos(phi)
        z = -sin_theta * torch.sin(phi)
        y =  torch.cos(theta)

        dirs = torch.stack([x, y, z], dim=-1)           # [H, W, 3]
        Y = _sh_basis(dirs, L)                           # [H, W, (L+1)^2]

        radiance = envmap[..., :3]                       # [H, W, 3]
        dt_dp = 2.0 * np.pi**2 / (h * w)
        weight = sin_theta[..., None] * dt_dp            # [H, W, 1]

        # L_lm: [(L+1)^2, 3]
        self.L_coeffs = (Y * weight).reshape(-1, (L+1)**2).T @ radiance.reshape(-1, 3)

    def eval(self, dirs, roughness=1.0):
        """
        Evaluate glossy irradiance at the given directions.

        Parameters
        ----------
        dirs : torch.Tensor
            Unit directions, shape [..., 3].
        roughness : float or torch.Tensor
            Scalar in [0, 1].  roughness=1 approximates diffuse; roughness=0 is
            mirror.  Differentiable when passed as a tensor.

        Returns
        -------
        torch.Tensor
            Shape [..., 3].
        """
        Y = _sh_basis(dirs, self.L)                      # [..., (L+1)^2]
        a = _vmf_sh_weights(roughness, self.L, dirs.device)  # [(L+1)^2]
        # weighted_coeffs: [(L+1)^2, 3]
        weighted = self.L_coeffs * a.unsqueeze(-1)
        return Y @ weighted                               # [..., 3]


class NVDRendererGlossy(NVDRenderer):
    """
    Drop-in replacement for NVDRenderer that adds a differentiable roughness
    parameter controlling glossy vs diffuse appearance.

    When roughness=1 the output is very close to the original NVDRenderer.
    When roughness=0 the output approaches a mirror reflection.

    The extra cost over NVDRenderer is:
      - higher-order SH evaluation (25 coefficients at order-4 vs 9)
      - per-view reflection direction computation (one matmul per render call)
    """

    def __init__(self, scene_params, shading=True, boost=1.0,
                 black_background=False, roughness=0.5, sh_order=4,
                 diffuse_weight=0.5, specular_weight=0.5):
        """
        Parameters
        ----------
        scene_params : dict
            Same dict used by NVDRenderer.
        roughness : float or torch.nn.Parameter
            Glossy roughness in [0, 1].  Pass a nn.Parameter to optimize it.
        sh_order : int
            SH order for glossy term (max 4 with current _sh_basis).
        diffuse_weight : float
            Weight of the diffuse (original SH) term.
        specular_weight : float
            Weight of the glossy SH term.
        """
        super().__init__(scene_params, shading=shading, boost=boost,
                         black_background=black_background)

        envmap = scene_params['envmap_scale'] * scene_params['envmap']
        self.sh_ho = SphericalHarmonicsHO(envmap, L=sh_order)

        # Camera positions in world space: [V, 3]
        self._cam_pos = torch.linalg.inv(self.view_mats)[:, :3, 3]  # [V, 3]

        if isinstance(roughness, torch.Tensor):
            self.roughness = roughness
        else:
            self.roughness = torch.tensor(roughness, dtype=torch.float32, device='cuda')

        self.diffuse_weight  = diffuse_weight
        self.specular_weight = specular_weight

    def _reflection_dirs(self, v, n):
        """
        Compute per-view, per-vertex reflection directions.

        Parameters
        ----------
        v : torch.Tensor  [N, 3]  vertex positions
        n : torch.Tensor  [N, 3]  vertex normals (unit)

        Returns
        -------
        torch.Tensor  [V, N, 3]
        """
        # view directions from vertex to camera: [V, N, 3]
        view_dirs = self._cam_pos[:, None, :] - v[None, :, :]      # [V, N, 3]
        view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

        n_exp = n[None].expand_as(view_dirs)                        # [V, N, 3]
        dot   = (n_exp * view_dirs).sum(dim=-1, keepdim=True).clamp(0.0)
        refl  = 2.0 * dot * n_exp - view_dirs                      # [V, N, 3]
        return refl / (refl.norm(dim=-1, keepdim=True) + 1e-8)

    def render(self, v, n, f):
        """
        Render with blended diffuse + glossy shading.

        Parameters
        ----------
        v : torch.Tensor  vertex positions  [N, 3]
        n : torch.Tensor  vertex normals    [N, 3]
        f : torch.Tensor  faces             [M, 3]
        """
        v_hom = torch.nn.functional.pad(v, (0, 1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, self.mvps.transpose(1, 2))
        f_int = f.int()
        rast  = dr.rasterize(self.glctx, v_ndc, f_int, self.res)[0]

        if self.shading:
            # --- diffuse term (original, view-independent) ---
            diffuse_verts = self.sh.eval(n).contiguous()            # [N, 3]
            diffuse_px    = dr.interpolate(diffuse_verts[None], rast, f_int)[0]  # [V,H,W,3]

            # --- glossy term (view-dependent) ---
            refl          = self._reflection_dirs(v, n)             # [V, N, 3]
            gloss_verts   = self.sh_ho.eval(refl, self.roughness)   # [V, N, 3]
            gloss_px      = dr.interpolate(gloss_verts.contiguous(), rast, f_int)[0]  # [V,H,W,3]

            combined = (self.diffuse_weight  * diffuse_px / np.pi
                      + self.specular_weight * gloss_px)

            col = torch.cat((combined,
                             torch.ones((*combined.shape[:-1], 1), device='cuda')), dim=-1)
            bg  = torch.zeros_like(self.bgs) if self.black_background else self.bgs
            result = dr.antialias(
                torch.where(rast[..., -1:] != 0, col, bg),
                rast, v_ndc, f_int, pos_gradient_boost=self.boost)
        else:
            v_cols = torch.ones_like(v)
            col    = dr.interpolate(v_cols[None], rast, f_int)[0]
            result = dr.antialias(col, rast, v_ndc, f_int, pos_gradient_boost=self.boost)

        return result
