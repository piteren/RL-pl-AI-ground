from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.printout import stamp
from pypaq.pms.base import POINT
from typing import Dict


from r4c.envy import RLEnvy
from r4c.actor import TrainableActor
from r4c.qlearning.qtable.qt_actor import QTableActor
from r4c.qlearning.dqn.dqn_actor import DQNActor
from r4c.policy_gradients.pg_actor import PGActor
from r4c.policy_gradients.actor_critic.ac_actor import ACActor
from r4c.policy_gradients.actor_critic.ac_critic import ACCritic
from r4c.policy_gradients.a2c.a2c_actor import A2CActor

from envies import SimpleBoardGame, CartPoleEnvy, AcrobotEnvy

# configurations of run_actor_training()
RUN_CONFIGS = {

    ### SimpleBoardGame

    'QTable_SBG': {
        'envy_type':        SimpleBoardGame,
        'envy_point':       {'board_size':6},
        'actor_type':       QTableActor,
        'actor_point':      {
            'exploration':      0.5,
            'sample_TR':        0.1,
            'batch_size':       10,
            'mem_batches':      10,
            'sample_memory':    True,
            'gamma':            0.5,
            'update_rate':      0.5},
        'num_batches':      100,
        'test_freq':        10,
        'test_episodes':    10,
    },

    'DQN_SBG': {
        'envy_type':        SimpleBoardGame,
        'envy_point':       {'board_size':6},
        'actor_type':       DQNActor,
        'actor_point':      {
            'exploration':      0.5,
            'sample_TR':        0.3,
            'batch_size':       10,
            'mem_batches':      10,
            'sample_memory':    True,
            'gamma':            0.5,
            'motorch_point':    {
                'n_hidden':         1,
                'hidden_width':     12,
                'baseLR':           0.01},
        },
        'num_batches':      200,
        'test_freq':        10,
        'test_episodes':    10,
    },

    ### CartPole

    'DQN_CP': {
        'envy_type': CartPoleEnvy,
        'envy_point':       {
            'step_reward':      0.1,
            'won_reward':       0.1,
            'lost_reward':      0.0},
        'actor_type': DQNActor,
        'actor_point': {
            'exploration':      0.3,
            'sample_TR':        0.0,
            'batch_size':       100,
            'mem_batches':      5,
            'sample_memory':    True,
            'gamma':            0.95,
            'motorch_point':    {
                'n_hidden':         2,
                'hidden_width':     20,
                'baseLR':           0.001},
        },
        'num_batches':      1500,
        'test_freq':        50,
        'test_episodes':    10,
    },

    'PG_CP': {
        'envy_type':        CartPoleEnvy,
        'envy_point':       {
            'step_reward':      0.1,
            'won_reward':       0.1,
            'lost_reward':      0.0},
        'actor_type':       PGActor,
        'actor_point':      {
            'exploration':      0.17,#0.84,
            'sample_TR':        0.88,#0.7,
            'batch_size':       64,#128,
            'discount':         0.96,#0.99,
            'do_zscore':        False,
            'motorch_point': {
                'n_hidden':         1,
                'hidden_width':     41,#27,
                'baseLR':           0.03,
                #'lay_norm':         True,
                #'use_scaled_ce':    True,
                #'do_clip':          True,
            },
        },
        'num_batches':      700,
        'test_freq':        50,
        'test_episodes':    10,
    },

    'PG_CP_exp': {
        'envy_type':        CartPoleEnvy,
        'envy_point':       {
            'step_reward':      0.01,
            'won_reward':       0.0,
            'lost_reward':     -1.0},
        'actor_type':       PGActor,
        'actor_point':      {
            'sample_TR':        1.0,
            'batch_size':       128,
            'discount':         0.95,
            'motorch_point': {
                'n_hidden':         1,
                'hidden_width':     30,
                'baseLR':           1e-4,
                #'lay_norm':         True,
            },
        },
        'num_batches':      500,
        'test_freq':        100,
        'test_episodes':    10,
    },

    'AC_CP': {
        'envy_type':        CartPoleEnvy,
        'envy_point':       {
            'step_reward':      0.1,
            'won_reward':       0.1,
            'lost_reward':      0.0},
        'actor_type':       ACActor,
        'actor_point':      {
            'exploration':      0.1,
            'sample_TR':        0.0,
            'batch_size':       256,
            'discount':         0.98,
            'do_zscore':        False,
            'critic_class':     ACCritic,
            'critic_gamma':     0.99,
            'critic_baseLR':    0.0005,
            'motorch_point':    {
                'n_hidden':         2,
                'hidden_width':     20,
                'baseLR':           0.0005,
            },
        },
        'num_batches':      1000,
        'test_freq':        100,
        'test_episodes':    10,
    },

    'A2C_CP': {
        'envy_type':        CartPoleEnvy,
        'envy_point':       {
            'step_reward':      0.1,
            'won_reward':       0.1,
            'lost_reward':      0.0},
        'actor_type':       A2CActor,
        'actor_point':      {
            'exploration':      0.25,#0.4,
            'sample_TR':        0.96,#0.3,
            'batch_size':       64,
            'discount':         0.66,#0.82,
            'do_zscore':        False,
            'motorch_point':    {
                'two_towers':       True,
                'n_hidden':         1,
                'hidden_width':     35,
                #'lay_norm':         True,
                'clamp_advantage':  None,
                'use_scaled_ce':    False,
                'use_huber':        True,#False,
                'baseLR':           0.03,
                #'do_clip':          True,
            },},
        'num_batches':      700,
        'test_freq':        50,
        'test_episodes':    10,
    },

    ### Acrobot

    'AC_ACR': {
        'envy_type':        AcrobotEnvy,
        'envy_point':       {},
        'actor_type':       ACActor,
        'actor_point':      {
            'exploration':      0.1,
            'sample_TR':        0.0,
            'batch_size':       500,
            'discount':         0.98,
            'do_zscore':        False,
            'critic_class':     ACCritic,
            'critic_gamma':     0.99,
            'critic_baseLR':    0.1,
            'motorch_point':    {
                'n_hidden':         2,
                'hidden_width':     20,
                'baseLR':           0.01,
        },},
        'num_batches':      500,
        'test_freq':        20,
        'test_episodes':    10,
    },

}


def run_actor_training(
        envy_type: type(RLEnvy),
        envy_point: POINT,
        actor_type: type(TrainableActor),
        actor_point: POINT,
        num_TS_ep=          100,
        seed=               121,
        loglevel=           20,
        save_topdir=        '_models',
        hpmser_mode=        False,
        picture: bool=      False,
        **train_point,  # for RLRunner.train()
) -> Dict:

    # early override in hpmser_mode
    if hpmser_mode:
        num_TS_ep = 0
        loglevel = 50

    name = f'{actor_type.__name__}_{envy_type.__name__}_{stamp()}'

    logger = get_pylogger(
        name=       name,
        add_stamp=  False,
        folder=     f'{save_topdir}/{name}' if not hpmser_mode else None,
        level=      loglevel)

    envy = envy_type(
        seed=       seed,
        logger=     logger,
        **envy_point)
    logger.debug(envy)

    actor = actor_type(
        name=           name,
        add_stamp=      False,
        envy=           envy,
        seed=           seed,
        logger=         logger,
        hpmser_mode=    hpmser_mode,
        **actor_point)
    logger.debug(actor)

    tr_res = actor.run_train(**train_point, picture=picture)

    if not hpmser_mode:
        actor.save()

    if not hpmser_mode:
        tr_nfo =   'Training report:\n'
        tr_nfo += f'> number of actions performed (n_actions):                    {tr_res["n_actions"]}\n'
        tr_nfo += f'> number of terminal states reached (n_terminals):            {tr_res["n_terminals"]}\n'
        tr_nfo += f'> number of wins (n_won):                                     {tr_res["n_won"]}\n'
        tr_nfo += f'> max number of succeeded tests in a row (succeeded_row_max): {tr_res["succeeded_row_max"]}'
        logger.info(tr_nfo)

    if num_TS_ep:
        ts_res = actor.test_on_episodes(n_episodes=num_TS_ep, max_steps=train_point.get('test_max_steps', None))
        logger.info(f'Test report: won factor: {int(ts_res[0]*100)}%, avg reward: {ts_res[1]:.1f}')

    return tr_res


if __name__ == "__main__":

    for run_config_name in [
        #'QTable_SBG',
        #'DQN_SBG',

        #'DQN_CP',
        #'PG_CP',
        'PG_CP_exp',
        #'AC_CP',
        #'A2C_CP',

        #'AC_ACR',
    ]:
        run_actor_training(
            num_TS_ep=  10,
            #loglevel=   10,
            #picture=    True,
            **RUN_CONFIGS[run_config_name])