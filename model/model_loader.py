import torch
from model.residual_unet_att import ResidualUNet3D

def get_model(args):

    if args.network == 'ResUNet':
        out_channels = args.num_classes

        
        if args.denoising:
            out_channels += 1
        model = ResidualUNet3D(f_maps=args.f_maps, out_channels=out_channels,
                               args=args, in_channels=args.in_channels, use_att=args.use_att,
                               use_paf=args.use_paf, use_uncert=args.use_uncert, denoising=args.denoising)
        if args.init_model_from_ckpt != '':
            print(f"Initializing model weights from checkpoint: {args.init_model_from_ckpt}")
            state_dict = torch.load(args.init_model_from_ckpt, map_location='cpu')["state_dict"]
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if 'model.' in k}
            model.load_state_dict(state_dict, strict=False)           
    return model
