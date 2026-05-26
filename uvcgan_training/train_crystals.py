import argparse
import os

from uvcgan2               import ROOT_OUTDIR, train
from uvcgan2.presets       import GEN_PRESETS, BH_PRESETS
from uvcgan2.utils.parsers import add_preset_name_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Train UVCGAN2 model on Crystal Dataset'
    )

    add_preset_name_parser(parser, 'gen',  GEN_PRESETS, 'uvcgan2')
    add_preset_name_parser(parser, 'head', BH_PRESETS,  'bsd', 'batch head')

    parser.add_argument(
        '--lambda-gp', dest = 'lambda_gp', type = float,
        default = 0.01, help = 'magnitude of the gradient penalty'
    )

    parser.add_argument(
        '--lambda-cycle', dest = 'lambda_cyc', type = float,
        default = 10.0, help = 'magnitude of the cycle-consistency loss'
    )

    parser.add_argument(
        '--lr-gen', dest = 'lr_gen', type = float,
        default = 1e-4, help = 'learning rate of the generator'
    )

    return parser.parse_args()

cmdargs   = parse_cmdargs()

args_dict = {
    'batch_size' : 1,
    'data' : {
        'datasets' : [
            {
                'dataset' : {
                    'name'   : 'cyclegan',
                    'domain' : 'synth' if domain == 'a' else 'real', 
                    'path'   : 'crystal_dataset',
                },
                # We assume 150x150 resolution based on your data_processing config.
                # If images are 160x160, padding to a multiple of 32 (like 160) or resizing to 256 is common.
                # Here we resize to 160 (if they aren't exactly) and crop.
                'shape'           : (3, 160, 160),
                'transform_train' : [
                    'random-flip-horizontal',
                    { 'name' : 'resize',      'size' : 160, },
                    { 'name' : 'random-crop', 'size' : 160, },
                ],
                'transform_test' : [
                    { 'name' : 'resize',      'size' : 160, },
                    { 'name' : 'center-crop', 'size' : 160, },
                ],
            } for domain in [ 'a', 'b' ]
        ],
        'merge_type' : 'unpaired',
        'workers'    : 1, # Increase this (e.g. 4 or 8) if data loading is a bottleneck
    },
    # Set the number of epochs to match your initial plan
    'epochs'      : 200, 
    'discriminator' : {
        'model'      : 'basic',
        'model_args' : { 'shrink_output' : False, },
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 1e-4,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
        'spectr_norm' : True,
    },
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : cmdargs.lr_gen,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model' : 'uvcgan2',
    'model_args' : {
        'lambda_a'        : cmdargs.lambda_cyc,
        'lambda_b'        : cmdargs.lambda_cyc,
        'lambda_idt'      : 0.5,
        'avg_momentum'    : 0.9999,
        'head_queue_size' : 3,
        'head_config'     : {
            'name'            : BH_PRESETS[cmdargs.head],
            'input_features'  : 512,
            'output_features' : 1,
            'activ'           : 'leakyrelu',
        },
    },
    'gradient_penalty' : {
        'center'    : 0,
        'lambda_gp' : cmdargs.lambda_gp,
        'mix_type'  : 'real-fake',
        'reduction' : 'mean',
    },
    'scheduler'       : None,
    'loss'            : 'lsgan',
    'steps_per_epoch' : 2000,
    
    # CRITICAL: We set transfer to None so it learns from scratch instead of looking for CelebA weights
    'transfer'        : None, 

    # args
    'label'  : (
        f'{cmdargs.gen}-{cmdargs.head}_'
        f'(cyc_{cmdargs.lambda_cyc}_gp_{cmdargs.lambda_gp}_lr_{cmdargs.lr_gen})'
    ),
    'outdir' : os.path.join(ROOT_OUTDIR, 'crystal_experiments', 'experiment_1'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 50,
}

if __name__ == '__main__':
    train(args_dict)