'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''

from easydict import EasyDict
from ding.utils import set_pkg_seed

from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.eval import SingleCarlaEvaluator
from core.policy import AutoPIDPolicy, ZeroShotPolicy
from core.utils.others.tcp_helper import parse_carla_tcp

autorun_config = dict(
    env=dict(
        ignore_light=True,
        ran_light_is_failure=False,
        simulator=dict(
            town='Town02',
            disable_two_wheels=True,
            # n_vehicles=10,
            # n_pedestrians=10,
            verbose=False,
            planner=dict(type='basic', ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[800, 600],
                    position=[2.0, 0, 1.4],
                    rotation=[0, 0, 0],
                ),
                dict(
                    name='birdview',
                    type='bev',
                    size=[500, 500],
                    pixels_per_meter=8,
                ),
            ),
        ),
        visualize=dict(
            type='rgb',
            outputs=['video'],
            show_text=True,
            save_dir='./'
        ),
        wrapper=dict(),
    ),
    server=[dict(carla_host='localhost', carla_ports=[9000, 9002, 2])],
    policy=dict(
        target_speed=25,
        noise=True,
        eval=dict(
            evaluator=dict(
                suite=['FullTown02-v2'],
                render=True,
                reset_param=dict(
                    start=0,
                    end=2,
                ),
            ),
        ),
    ),
)

main_config = EasyDict(autorun_config)


def main(cfg, seed=0):
    tcp_list = parse_carla_tcp(cfg.server)
    host, port = tcp_list[0]

    carla_env = CarlaEnvWrapper(SimpleCarlaEnv(cfg.env, host, port), cfg.env.wrapper)
    carla_env.seed(seed)
    set_pkg_seed(seed)
    auto_policy = ZeroShotPolicy(cfg.policy)
    evaluator = SingleCarlaEvaluator(None, cfg.policy.eval.evaluator, carla_env, auto_policy.eval_mode)
    evaluator.eval(cfg.policy.eval.evaluator.reset_param)
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
