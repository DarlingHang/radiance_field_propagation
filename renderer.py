import torch,os,imageio,sys
from tqdm.auto import tqdm
from utils import *
from models.tensoRF import  TensorVMSplit, raw2alpha, AlphaGridMask



def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    rgbs, sem_maps = [], []
    fg_maps, bg_maps, error_rgb_maps = [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, sem_map, fg_map, bg_map, error_rgb_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        sem_maps.append(sem_map)
        fg_maps.append(fg_map)
        bg_maps.append(bg_map)
        error_rgb_maps.append(error_rgb_map)
    
    return torch.cat(rgbs), torch.cat(sem_maps), torch.cat(fg_maps), torch.cat(bg_maps), torch.cat(error_rgb_maps)

@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    os.makedirs(savePath, exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, sem_map, fg_map, bg_map, error_egb_map = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)
        fg_map = fg_map.clamp(0.0, 1.0)
        bg_map = bg_map.clamp(0.0, 1.0)
        error_egb_map = error_egb_map.clamp(0.0, 1.0)
        sem_map = sem_map.argmax(-1)
        
        rgb_map, sem_map = rgb_map.reshape(H, W, 3).cpu(), sem_map.reshape(H, W, 1).repeat(1, 1, 3).cpu()
        fg_map, bg_map, error_egb_map = fg_map.reshape(H, W, 3).cpu(), bg_map.reshape(H, W, 3).cpu(), error_egb_map.reshape(H, W, 3).cpu()


        gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        fg_map = (fg_map.numpy() * 255).astype('uint8')
        bg_map = (bg_map.numpy() * 255).astype('uint8')
        sem_map = (sem_map.numpy() * 255).astype('uint8')
        gt_rgb = (gt_rgb.numpy() * 255).astype('uint8')
        error_egb_map = (error_egb_map.numpy() * 255).astype('uint8')
        gt_rgb[sem_map==0] = 0
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}_fg.png', fg_map)
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}_bg.png', bg_map)
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}_mask.png', sem_map)
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}_mask_gt.png', gt_rgb)
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}_err.png', error_egb_map)


