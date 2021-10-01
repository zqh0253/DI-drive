import os
import numpy as np
from functools import partial
from easydict import EasyDict
import copy
import time
from tensorboardX import SummaryWriter

from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import SerialEvaluator,  SingleCarlaEvaluator, CarlaBenchmarkEvaluator
from ding.envs import SyncSubprocessEnvManager, BaseEnvManager
from ding.policy import PPOPolicy
from ding.worker import BaseLearner, SampleCollector
from ding.utils import set_pkg_seed, DistContext
from demo.simple_rl.model import PPORLModel, CPPORGBRLModel, PPORGBRLModel
from demo.simple_rl.env_wrapper import DiscreteBenchmarkEnvWrapper, ContinuousBenchmarkEnvWrapper, ContinuousRgbBenchmarkEnvWrapper, ContinuousRgbCarlaEnvWrapper
from core.utils.data_utils.bev_utils import unpack_birdview
from core.utils.others.ding_utils import compile_config

train_config = dict(
    exp_name='ppo21_bev32_lr1e4_bs128_ns3000_update5_train_ft',
    env=dict(
        collector_env_num=6,
        evaluator_env_num=1,
        simulator=dict(
            town='Town01',
            disable_two_wheels=True,
            verbose=False,
            waypoint_num=32,
            planner=dict(
                type='behavior',
                resolution=0.1,
                min_distance = 3,
            ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[320, 180],
                    position=[-5.5, 0, 2.8],
                    rotation=[-15, 0, 0],
                ),
            ),
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        off_road_is_failure=True,
        off_route_is_failure=True,
        ignore_light=True,
        finish_reward=300,
        manager=dict(
            collect=dict(
                auto_reset=True,
                shared_memory=False,
                context='spawn',
                max_retry=1,
            ),
            eval=dict(
                shared_memory=False,
                auto_reset=False,
                context='spawn',
                mex_retry=1
                )
        ),
        visualize = dict(
            type='rgb',
            outputs=['video'],
            show_text=True,
            save_dir='/home/qhzhang/code/drive-from-video/video'
        ),
        wrapper=dict(
            # Collect and eval suites for training
            collect=dict(suite='NoCrashTown01-v1', ),
            eval=dict(suite='NoCrashTown01-v1', ),
        ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[8000, 8016, 2]),
    ],
    policy=dict(
        cuda=True,
        nstep_return=False,
        on_policy=True,
        model=dict(
            action_shape=2,
        ),
        learn=dict(
            multi_gpu=False,
            epoch_per_collect=12,
            batch_size=256,
            learning_rate=0.0001,
            weight_decay=0.0001,
            value_weight=0.5,
            adv_norm=False,
            entropy_weight=0.01,
            clip_ratio=0.2,
            target_update_freq=100,
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                ),
            ),
        ),
        collect=dict(
            my_n_sample=1024,
            collector=dict(
                collect_print_freq=1000,
                deepcopy_obs=True,
                transform_obs=True,
            ),
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(
            evaluator=dict(
                suite='NoCrashTown01-v1',
                episodes_per_suite=50,
                eval_freq=5000,
                n_episode=3,
                stop_rate=0.7,
                transform_obs=True,
                render=True,
            ),
        ),
    ),
)

main_config = EasyDict(train_config)


def wrapped_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    return ContinuousRgbBenchmarkEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port), wrapper_cfg)

def single_wrapped_env_for_benchmark(env_cfg, host, port, tm_port=None): 
    return ContinuousRgbCarlaEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port))

def main(cfg, seed=0):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        PPOPolicy,
        BaseLearner,
        SampleCollector,
    )
    tcp_list = parse_carla_tcp(cfg.server)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    assert len(tcp_list) >= collector_env_num + evaluator_env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(
            collector_env_num + evaluator_env_num, len(tcp_list)
    )

    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper.collect, *tcp_list[i]) for i in range(collector_env_num)],
        cfg=cfg.env.manager.collect,
    )
    # evaluate_env = BaseEnvManager(
    #     env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper.eval, *tcp_list[collector_env_num + i]) for i in range(evaluator_env_num)],
    #     cfg=cfg.env.manager.eval,
    # )
    vis_eval_single_env = wrapped_env(cfg.env, cfg.env.wrapper.eval, *tcp_list[collector_env_num])
    eval_single_env = SyncSubprocessEnvManager(
            env_fn=[partial(single_wrapped_env_for_benchmark, cfg.env,  *tcp_list[collector_env_num+1])],
            cfg=cfg.env.manager.eval
        )


    collector_env.seed(seed)
    # evaluate_env.seed(seed)
    eval_single_env.seed(seed)
    vis_eval_single_env.seed(seed)
    set_pkg_seed(seed)

    model = CPPORGBRLModel(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    vis_evaluator = SingleCarlaEvaluator(cfg.policy.eval.evaluator, vis_eval_single_env, policy.eval_mode) #, tb_logger, exp_name=cfg.exp_name)
    evaluator = CarlaBenchmarkEvaluator(cfg.policy.eval.evaluator, eval_single_env, policy.eval_mode)

    learner.call_hook('before_run')
    
    cnt = 0
    while True:
        cnt +=1
        # if evaluator.should_eval(learner.train_iter):
        #     stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
        #     if stop:
        #         break
        vis_evaluator.eval()
        if cnt % 10 == 0:
            evaluator.eval()
        # Sampling data from environments
        new_data = collector.collect(train_iter=learner.train_iter, n_sample=cfg.policy.collect.my_n_sample)
        print('-collect finish!!!')
        unpack_birdview(new_data)
        print('unpack finish!!!')
        learner.train(new_data, collector.envstep)
        print('-train finish!!!!')
    learner.call_hook('after_run')

    collector_env.close()
    eval_single_env.close()
    vis_eval_single_env.close()

    evaluator.close()
    learner.close()


if __name__ == '__main__':
    main(main_config)
