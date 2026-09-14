import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import json

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, output_dir) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    cfg.output_dir = output_dir

    # 记录参数
    cfg.yaml_cfg['seed'] = args.seed
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, 'configs.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg.yaml_cfg, f, ensure_ascii=False, indent=4)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    solver.val_save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=True,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed', default=None,)
    args = parser.parse_args()

    # ======= #
    args.description = 'AI-TOD test 指标计算'

    # 配置参数
    args.config = 'src/zoo/DTW_Det/configs/dtwdet/dtwdet_r50vd_6x_aitod.yml'
    
    # args.resume = 'output/dtwdet_r50vd_6x_aitod_DTW_Det/best_checkpoint.pth'
    # output_dir = 'eval_output/dtwdet_r50vd_6x_aitod_DTW_Det/best_checkpoint'

    # os.makedirs(output_dir, exist_ok=True)

    # main(args, output_dir)

    # 批量测试
    for i in range(0, 20):
        print(f"================= 测试 checkpoint00{71-i} ================= ")
        args.resume = f'output/dtwdet_r50vd_6x_aitod_DTW_Det/checkpoint00{71-i}.pth'
        output_dir = f'eval_output/dtwdet_r50vd_6x_aitod_DTW_Det/checkpoint{71-i}'
        os.makedirs(output_dir, exist_ok=True)
        main(args, output_dir)
        



    
