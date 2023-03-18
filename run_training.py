from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.printout import stamp


from r4c.envy import RLEnvy
from r4c.actor import TrainableActor
from r4c.runner import RLRunner
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
            'gamma':            0.5,
            'update_rate':      0.5},
        'num_updates':      100,
        'batch_size':       10,
        'mem_batches':      10,
        'sample_memory':    True,
        'exploration':      0.5,
        'sampled_TR':       0.1,
        'test_freq':        10},

    'DQN_SBG': {
        'envy_type':        SimpleBoardGame,
        'envy_point':       {'board_size':6},
        'actor_type':       DQNActor,
        'actor_point':      {
            'gamma':            0.5,
            'motorch_point':    {
                'n_hidden':         1,
                'hidden_width':     12,
                'baseLR':           0.01},
        },
        'num_updates':      200,
        'batch_size':       10,
        'mem_batches':      10,
        'sample_memory':    True,
        'exploration':      0.5,
        'sampled_TR':       0.3,
        'test_freq':        10},

    ### CartPole

    'DQN_CP': {
        'envy_type': CartPoleEnvy,
        'envy_point':       {
            'step_reward':      0.1,
            'won_reward':       0.1,
            'lost_reward':      0.0},
        'actor_type': DQNActor,
        'actor_point': {
            'gamma': 0.95,
            'motorch_point': {
                'n_hidden':         2,
                'hidden_width':     20,
                'baseLR':           0.001},
        },
        # RLRunner.train()
        'num_updates':      1500,
        'batch_size':       100,
        'mem_batches':      5,
        'sample_memory':    True,
        'exploration':      0.3,
        'sampled_TR':       0.0,
        'upd_on_episode':   False,
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
            'discount':         0.92,#0.95,
            'use_mavg':         False,
            'mavg_factor':      0.3,
            'do_zscore':        False,
            'motorch_point': {
                'n_hidden':         1,#2,
                'hidden_width':     36,#20,
                'baseLR':           0.001,
                #'lay_norm':         True,
                #'use_scaled_ce':    True,  # TODO: check, but with lower LR
                #'do_clip':          True,
            },
        },
        'num_updates':      700,
        'batch_size':       64,#100,
        'exploration':      0.1,#0.2,#0.5,
        'sampled_TR':       0.97,#1.0,#0.5,
        'upd_on_episode':   False,
        'test_freq':        50,
        'test_episodes':    10,
    },

    'AC_CP': {
        'envy_type':        CartPoleEnvy,
        'envy_point':       {
            'reward_scale': 0.1,
            'lost_penalty': -10.0},
        'actor_type':       ACActor,
        'actor_point':      {
            'discount':         0.98,
            'use_mavg':         False,
            'mavg_factor':      0.3,
            'do_zscore':        False,
            'critic_class':     ACCritic,
            'critic_gamma':     0.99,
            'critic_baseLR':    0.0005,
            'hidden_layers':    (20,20),
            'baseLR':           0.0005},
        'num_updates':      1000,
        'batch_size':       256,
        'exploration':      0.1,
        'sampled_TR':       0.0,
        'upd_on_episode':   False,
        'test_freq':        50,
        'test_episodes':    10},

    'A2C_CP': {
        'envy_type':        CartPoleEnvy,
        'envy_point':       {
            'reward_scale': 0.1,
            'won_reward':   0.0,
            'lost_penalty': -1.0,
            #'max_steps':    200,
        },
        'actor_type':       A2CActor,
        'actor_point':      {
            'discount':         0.98,
            'use_mavg':         True,
            'mavg_factor':      0.12,
            'do_zscore':        False,
            'two_towers':       True,   #False,
            'num_layers':       1,      #2,
            'layer_width':      50,     #20,
            'lay_norm':         True,
            'clamp_advantage':  None,   #0.5,
            'use_scaled_ce':    True,
            'use_huber':        True,
            'baseLR':           0.01,
            'do_clip':          True},
        'num_updates':      2000,
        'batch_size':       256,  # 768,
        'exploration':      0.2,  # 0.0,
        'sampled_TR':       0.1,  # 0.0,
        'upd_on_episode':   False,      #True,
        'test_freq':        50,
        'test_episodes':    10,
    },

    ### Acrobot

    'AC_ACR': {
        'envy_type':        AcrobotEnvy,
        'envy_point':       {},
        'actor_type':       PGActor,
        'actor_point':      {'baseLR':0.01},
        'trainer_point':    {
            'critic_class':     ACCritic,
            'critic_gamma':     0.99,
            'critic_baseLR':    0.1,
            'batch_size':       500,
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
        envy_point: dict,       # for Envy init
        actor_type: type(TrainableActor),
        actor_point: dict,      # for Actor init
        nTS_ep=         100,
        seed=           121,
        loglevel=       20,
        save_topdir=    '_models',
        hpmser_mode=    False,
        inspect=        False,
        **train_point,          # for RLRunner.train()
) -> dict:

    # early override
    if hpmser_mode:
        nTS_ep = 0
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
    logger.info(envy)

    actor = actor_type(
        name=           name,
        add_stamp=      False,
        envy=           envy,
        seed=           seed,
        logger=         logger,
        hpmser_mode=    hpmser_mode,
        **actor_point)
    logger.info(actor)

    runner = RLRunner(
        envy=       envy,
        actor=      actor,
        seed=       seed,
        logger=     logger)

    if inspect:
        max_steps = train_point['test_max_steps'] if 'test_max_steps' in train_point else None
        runner.play_episode(max_steps=max_steps, inspect=True)

    if nTS_ep:
        ts_res = runner.test_on_episodes(n_episodes=nTS_ep)
        logger.info(f'Test report: won factor: {int(ts_res[0]*100)}%, avg reward: {ts_res[1]:.1f}')

    tr_res = runner.train(**train_point, inspect=inspect)

    if not hpmser_mode:
        actor.save()

    if not hpmser_mode:
        tr_nfo =   'Training report:\n'
        tr_nfo += f'> number of actions performed (n_actions):                    {tr_res["n_actions"]}\n'
        tr_nfo += f'> number of terminal states reached (n_terminals):            {tr_res["n_terminals"]}\n'
        tr_nfo += f'> number of wins (n_won):                                     {tr_res["n_won"]}\n'
        tr_nfo += f'> max number of succeeded tests in a row (succeeded_row_max): {tr_res["succeeded_row_max"]}'
        logger.info(tr_nfo)

    if inspect:
        max_steps = train_point['test_max_steps'] if 'test_max_steps' in train_point else None
        runner.play_episode(max_steps=max_steps, inspect=True)

    if nTS_ep:
        ts_res = runner.test_on_episodes(n_episodes=nTS_ep)
        logger.info(f'Test report: won factor: {int(ts_res[0]*100)}%, avg reward: {ts_res[1]:.1f}')

    return tr_res


if __name__ == "__main__":

    for run_config_name in [
        #'QTable_SBG',
        #'DQN_SBG',

        #'DQN_CP',
        'PG_CP',
        #'AC_CP',
        #'A2C_CP',

        #'AC_ACR',
    ]:
        run_actor_training(
            nTS_ep=     10,
            #loglevel=   5,
            #inspect=    True,
            **RUN_CONFIGS[run_config_name])