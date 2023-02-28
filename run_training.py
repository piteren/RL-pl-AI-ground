from r4c.envy import RLEnvy
from r4c.actor import TrainableActor
from r4c.trainer import RLTrainer
from r4c.qlearning.qtable.qt_actor import QTableActor
from r4c.qlearning.qtable.qt_trainer import QTableTrainer
from r4c.qlearning.dqn.dqn_actor import DQNActor
from r4c.qlearning.ql_trainer import QLearningTrainer
from r4c.policy_gradients.pg_actor import PGActor
from r4c.policy_gradients.pg_trainer import PGTrainer
from r4c.policy_gradients.actor_critic.ac_critic import ACCritic
from r4c.policy_gradients.actor_critic.ac_trainer import ACTrainer
from r4c.policy_gradients.a2c.a2c_actor import A2CActor
from r4c.policy_gradients.actor_critic_shared.ac_shared_actor import ACSharedActor

from envies import SimpleBoardGame, CartPoleEnvy, AcrobotEnvy



train_configs = {

    'SBG_QTable': {
        'envy_type':        SimpleBoardGame,
        'envy_point':       {'board_size':6},
        'actor_type':       QTableActor,
        'actor_point':      {},
        'trainer_type':     QTableTrainer ,
        'trainer_point':    {
            'batch_size':       10,
            'memsize_batches':  10,
            'exploration':      0.5,
            'train_sampled':    0.1,
            'gamma':            0.5,
            'update_rate':      0.5},
        'num_updates':      100,
        'test_freq':        10},

    'SBG_DQN': {
        'envy_type':        SimpleBoardGame,
        'envy_point':       {'board_size':6},
        'actor_type':       DQNActor,
        'actor_point':      {
            'baseLR':           0.01,
            'device':           None},
        'trainer_type':     QLearningTrainer,
        'trainer_point':    {
            'batch_size':       10,
            'memsize_batches':  10,
            'exploration':      0.5,
            'train_sampled':    0.3,
            'gamma':            0.5},
        'num_updates':      200,
        'test_freq':        10},

    'CP_PG': {
        'envy_type':        CartPoleEnvy,
        'envy_point':       {
            'reward_scale': 0.1,
            'lost_penalty': -10.0},
        'actor_type':       PGActor,
        'actor_point':      {
            'hidden_layers':    (20,20),
            'baseLR':           0.001,# if not episodic else 0.01,
            'lay_norm':         True,
            'use_scaled_ce':    False,  # TODO: check, but with lower LR
            'do_clip':          True},
        'trainer_type':     PGTrainer,
        'trainer_point':    {
            'batch_size':       256,# if not episodic else 555,
            'memsize_batches':  1,
            'exploration':      0.1,
            'train_sampled':    0.3,
            'discount':         0.98,
            'use_mavg':         True,
            'mavg_factor':      0.3,
            'do_zscore':        True},
        'num_updates':      1000,
        'upd_on_episode':   False,#episodic,
        'test_freq':        50,
        'test_episodes':    10},

    'CP_AC': {
        'envy_type':        CartPoleEnvy,
        'envy_point':       {
            'reward_scale': 0.1,
            'lost_penalty': -10.0},
        'actor_type':       PGActor,
        'actor_point':      {
            'hidden_layers':    (20,20),
            'baseLR':           0.0005},
        'trainer_type':     ACTrainer,
        'trainer_point':    {
            'critic_class':     ACCritic,
            'critic_gamma':     0.99,
            'critic_baseLR':    0.0005,
            'batch_size':       256,
            'memsize_batches':  1,
            'exploration':      0.1,
            'train_sampled':    0.0,
            'discount':         0.98,
            'use_mavg':         False,
            'mavg_factor':      0.3,
            'do_zscore':        False},
        'num_updates':      1000,
        'upd_on_episode':   False,
        'test_freq':        50,
        'test_episodes':    10},

    'CP_A2C': {
        'envy_type':        CartPoleEnvy,
        'envy_point':       {
            'reward_scale': 0.1,
            'won_reward':   0.0,
            'lost_penalty': -1.0,
            #'max_steps':    200,
        },
        'actor_type':       A2CActor,
        'actor_point':      {
            #'devices':          None,
            'two_towers':       True,   #False,
            'num_layers':       1,      #2,
            'layer_width':      50,     #20,
            'lay_norm':         True,
            'clamp_advantage':  None,   #0.5,
            'use_scaled_ce':    True,
            'use_huber':        True,
            'baseLR':           0.01,
            'do_clip':          True,
        },
        'trainer_type':     PGTrainer,
        'trainer_point':    {
            'batch_size':       256,    #768,
            'memsize_batches':  1,
            'exploration':      0.2,    #0.0,
            'train_sampled':    0.1,    #0.0,
            'discount':         0.98,
            'use_mavg':         True,
            'mavg_factor':      0.12,
            'do_zscore':        False},
        'num_updates':      2000,
        'upd_on_episode':   False,      #True,
        'test_freq':        50,
        'test_episodes':    10,
        'test_render':      True,
        'inspect':          True,
    },

    'CP_ACShared': {
        'envy_type':        CartPoleEnvy,
        'envy_point':       {
            'reward_scale': 0.1,
            'lost_penalty': -10.0},
        'actor_type':       ACSharedActor,
        'actor_point':      {
            'hidden_layers':    (20,20),
            'baseLR':           0.00005},
        'trainer_type':     PGTrainer,
        'trainer_point':    {
            'batch_size':       256,
            'memsize_batches':  1,
            'exploration':      0.1,
            'train_sampled':    0.3,
            'discount':         0.98,
            'use_mavg':         True,
            'mavg_factor':      0.3,
            'do_zscore':        True},
        'num_updates':      1000,
        'upd_on_episode':   False,
        'test_freq':        50,
        'test_episodes':    10},

    'ACR_AC': {
        'envy_type':        AcrobotEnvy,
        'envy_point':       {},
        'actor_type':       PGActor,
        'actor_point':      {'baseLR':0.01},
        'trainer_type':     ACTrainer,
        'trainer_point':    {
            'critic_class':     ACCritic,
            'critic_gamma':     0.99,
            'critic_baseLR':    0.1,
            'batch_size':       500,
            'memsize_batches':  1,
            'exploration':      0.1,
            'train_sampled':    0.0,
            'discount':         0.98,
            'use_mavg':         False,
            'mavg_factor':      0.3,
            'do_zscore':        False},
        'num_updates':      500,
        'upd_on_episode':   True, # TODO:?
        'test_freq':        20,
        'test_episodes':    10},

}


def run_actor_training(
        envy_type: type(RLEnvy),
        envy_point: dict,                   # for RLEnvy init
        actor_type: type(TrainableActor),
        actor_point: dict,                  # for TrainableActor init
        trainer_type: type(RLTrainer),
        trainer_point: dict,                # for RLTrainer init
        nTS_ep=         100,
        seed=           121,
        loglevel=       20,
        hpmser_mode=    False,
        **train_kwargs,                     # for RLTrainer.train()
) -> dict:

    # early override
    if hpmser_mode:
        nTS_ep = 0
        loglevel = 50

    envy = envy_type(
        seed=       seed,
        loglevel=   loglevel,
        **envy_point)

    actor = actor_type(
        envy=       envy,
        seed=       seed,
        loglevel=   loglevel,
        **actor_point)

    trainer = trainer_type(
        envy=       envy,
        actor=      actor,
        seed=       seed,
        loglevel=   loglevel,
        **trainer_point)

    if nTS_ep:
        ts_res = trainer.test_on_episodes(n_episodes=nTS_ep)
        print(f'Test report: won factor: {int(ts_res[0]*100)}%, avg reward: {ts_res[1]:.1f}')

    tr_res = trainer.train(**train_kwargs)
    if not hpmser_mode:
        print('Training report:')
        print(f'> number of actions performed (n_actions):                    {tr_res["n_actions"]}')
        print(f'> number of terminal states reached (n_terminals):            {tr_res["n_terminals"]}')
        print(f'> number of wins (n_won):                                     {tr_res["n_won"]}')
        print(f'> max number of succeeded tests in a row (succeeded_row_max): {tr_res["succeeded_row_max"]}')

    if nTS_ep:
        ts_res = trainer.test_on_episodes(n_episodes=nTS_ep)
        print(f'Test report: won factor: {int(ts_res[0]*100)}%, avg reward: {ts_res[1]:.1f}')

    return tr_res


if __name__ == "__main__":

    for config_name in [
        #'SBG_QTable',
        #'SBG_DQN',
        #'CP_PG',
        #'CP_AC',
        'CP_A2C',
        #'CP_ACShared',
        #'ACR_AC',
    ]:
        run_actor_training(
            nTS_ep=     10,
            #loglevel=   5,
            **train_configs[config_name])