from easydict import EasyDict
import torch
from functools import partial

from core.envs import SimpleCarlaEnv
from core.policy import CILRSPolicy
from core.eval import CarlaBenchmarkEvaluator
from core.utils.others.tcp_helper import parse_carla_tcp
from ding.utils import set_pkg_seed, deep_merge_dicts
from ding.envs import AsyncSubprocessEnvManager
from demo.cilrs.cilrs_env_wrapper import CILRSEnvWrapper

OBSSCALE=4
cilrs_config = dict(
    env=dict(
        env_num=16,
        visualize=dict(type='obs',
            outputs=['video'],
            show_text=True,
            save_dir='/home/yhxu/qhzhang/video/supp'),
        simulator=dict(
            town='Town02',
            disable_two_wheels=True,
            verbose=False,
            planner=dict(
                type='behavior',
                resolution=1,
            ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[320*OBSSCALE, 180*OBSSCALE],
                    #size=[320, 180],
                    position=[2.0, 0.0, 1.4],
                    #size=[400, 300],
                    #position=[1.3, 0.0, 2.3],
                    rotation=[0,0,0],
                    fov=100,
                ),
                dict(
                    name='obs',
                    type='rgb',
                    size=[320*OBSSCALE, 180*OBSSCALE],
                    #size=[320, 180],
                    position=[-5.5, 0, 2.8],
                    #size=[400, 300],
                    #position=[1.3, 0.0, 2.3],
                    rotation=[-15,0,0],
                    fov=100,
                ),
            ),
        ),
        wrapper=dict(scale=OBSSCALE),
        col_is_failure=True,
        stuck_is_failure=True,
        ignore_light=False,
        manager=dict(
            auto_reset=False,
            shared_memory=False,
            context='spawn',
            max_retry=1,
        ),
    ),
    server=[dict(carla_host='localhost', carla_ports=[9000, 9032, 2])],
    policy=dict(
        #ckpt_path='./checkpoints/cilrs_train/0.000101-best_ckpt.pth',
        ckpt_path='./checkpoints_taco/cilrs_train/0.0001099-00090_ckpt.pth',
        model=dict(
            num_branch=4,
            backbone='resnet34',
            pretrained=False,
            bn=True
        ),
        eval=dict(
            evaluator=dict(
                suite=['FullTown02-v2'],
                transform_obs=True,
                render=True,
                seed=1
            ),
        )
    ),
)

main_config = EasyDict(cilrs_config)


def wrapped_env(env_cfg, host, port, tm_port=None):
    return CILRSEnvWrapper(SimpleCarlaEnv(env_cfg, host, port), env_cfg.wrapper)


def main(cfg, seed=0):
    cfg.env.manager = deep_merge_dicts(AsyncSubprocessEnvManager.default_config(), cfg.env.manager)

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env.env_num
    assert len(tcp_list) >= env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(env_num, len(tcp_list))

    carla_env = AsyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, *tcp_list[i]) for i in range(env_num)],
        cfg=cfg.env.manager,
    )
    carla_env.seed(seed)
    set_pkg_seed(seed)
    cilrs_policy = CILRSPolicy(cfg.policy).eval_mode
    if cfg.policy.ckpt_path is not None:
        print('loading checkpoint')
        state_dict = torch.load(cfg.policy.ckpt_path)
        cilrs_policy.load_state_dict(state_dict)

    carla_env.enable_save_replay('./video')
    evaluator = CarlaBenchmarkEvaluator(cfg.policy.eval.evaluator, carla_env, cilrs_policy)
    success_rate = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    # ckpt_names = ['0.0001022', '5.022e-05', '0.0001011', '5.011e-05']
    # ckpt_names = ['0.000101', '0.000103', '0.000104', '0.0001015']
    # ckpt_names = ['5.022e-05', '5.011e-05', '5.05e-05', '5.02e-05', ]
    # ckpt_names = ['5.05e-05', '5.02e-05', ]
    # ckpt_names = ['0.0001022', '0.0001011', '0.000105', '0.000102', ]
    ckpt_names = ['0.000102', ]
    # ckpt_names = ['0.0005022', '0.0005011', '0.000505', '0.000502', ]
    # ckpt_names = ['0.0001', '0.0005', '1e-05', '5e-05', ]
    # ckpt_names = ['0.0001' ]
    for name in ckpt_names:
        for i in range(6,7):
        # for i in range(6,10):
            # main_config.policy.ckpt_path = f'./checkpoints_taco/cilrs_train/{name}-{i*10:05d}_ckpt.pth'
            main_config.policy.ckpt_path = f'./checkpoints_moco_aug/cilrs_train/{name}-{i*10:05d}_ckpt.pth'
            print(main_config.policy.ckpt_path)
            main_config.policy.eval.evaluator.seed = 0
            main(main_config)
