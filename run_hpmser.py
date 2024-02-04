from copy import deepcopy
from hpmser.search import HPMSer
from pypaq.mpython.devices import DevicesPypaq

from run_training import RUN_CONFIGS, run_actor_training


# updates nested dict (run_configs[run_config_name]) with kwargs, returns score <0;1>
def run_actor_training_wrap(
        run_config_name: str,
        device: DevicesPypaq,
        max_batch_size: int,
        hpmser_mode=        True,
        **kwargs                # kwargs starting with specific prefix go to one of mdicts
) -> float:

    pd = deepcopy(RUN_CONFIGS[run_config_name])

    envy_point = {}
    actor_point = {}
    motorch_point = {'device':device}
    point = {}
    for param in kwargs:

        param_from_points = False
        sd = {'env_':envy_point, 'act_':actor_point, 'mot_':motorch_point}
        for k in sd:
            if param.startswith(k):
                sd[k][param[4:]] = kwargs[param]
                param_from_points = True

        if not param_from_points:
            point[param] = kwargs[param]

    pd['hpmser_mode'] = hpmser_mode
    pd.update(point)
    pd['envy_point'].update(envy_point)
    pd['actor_point'].update(actor_point)
    if 'motorch_point' in pd['actor_point']:
        pd['actor_point']['motorch_point'].update(motorch_point)

    tr_res = run_actor_training(**pd)

    if tr_res['n_updates_done'] == pd['num_batches']:
        return 0.0

    num_actions_done = tr_res['n_updates_done'] * actor_point['batch_size']
    num_actions_max = pd['num_batches'] * max_batch_size

    return (num_actions_max - num_actions_done) / num_actions_max



if __name__ == "__main__":

    hpmser_configs = {

        'DQN_CP': {
            'psdd': {
                'act_exploration':      [0.0,0.9],
                'act_sample_TR':        [0.0,0.9],
                'act_batch_size':       (8,16,32,64,128,256),
                'act_mem_batches':      [1,20],
                'act_gamma':            [0.5, 0.99],
                #'act_lay_norm':         (True, False),
                #'act_use_huber':        (True, False),
                #'act_do_clip':          (True, False),
                'mot_n_hidden':         [1,3],
                'mot_hidden_width':     [10,50],
                'mot_baseLR':           (0.03, 0.01, 0.005, 0.001, 0.0005, 0.0001),
            },
        },

        'PG_CP': {
            'psdd': {
                'act_exploration':      [0.0,1.0],
                'act_sample_TR':        [0.0,1.0],
                'act_batch_size':       (8,16,32,64,128,256),
                'act_discount':         [0.5,1.0],
                'mot_n_hidden':         [1,3],
                'mot_hidden_width':     [10,50],
                'mot_baseLR':           (0.03, 0.01, 0.005, 0.001, 0.0005, 0.0001),
            }
        },

        'AC_CP': {
            'psdd': {
                'act_exploration':          [0.0,1.0],
                'act_sample_TR':            [0.0,1.0],
                'act_batch_size':           (8,16,32,64,128,256),
                'act_discount':             [0.5,1.0],
                'act_critic_gamma':         [0.0,1.0],
                'act_critic_baseLR':        (0.03, 0.01, 0.005, 0.001, 0.0005, 0.0001),
                'act_critic_n_hidden':      [1,3],
                'act_critic_hidden_width':  [10,50],
                'mot_n_hidden':             [1,3],
                'mot_hidden_width':         [10,50],
                'mot_baseLR':               (0.03, 0.01, 0.005, 0.001, 0.0005, 0.0001),
            }
        },

        'A2C_CP': {
            'psdd': {
                'act_exploration':      [0.0,1.0],
                'act_sample_TR':        [0.0,1.0],
                'act_batch_size':       (8,16,32,64,128,256),
                'act_discount':         [0.0,1.0],
                'mot_two_towers':       (True, False),
                'mot_n_hidden':         [1,3],
                'mot_hidden_width':     [10,50],
                'mot_use_huber':        (True, False),
                'mot_baseLR':           (0.03, 0.01, 0.005, 0.001, 0.0005, 0.0001),
            }
        },

    }

    for rc_name in [
        #'DQN_CP',
        #'PG_CP',
        'AC_CP',
        #'A2C_CP',
    ]:

        func_const = {
            'run_config_name':  rc_name,
            'num_batches':      1000,
            'test_freq':        20,
            'test_episodes':    10,
            'inspect':          False,
            'break_ntests':     1,
            'max_batch_size':   max(hpmser_configs[rc_name]['psdd']['act_batch_size']),
        }
        if 'const' in hpmser_configs[rc_name]:
            func_const.update(hpmser_configs[rc_name]['const'])

        HPMSer(
            func=       run_actor_training_wrap,
            func_psdd=  hpmser_configs[rc_name]['psdd'],
            func_const= func_const,
            devices=    [None]*10,
            n_loops=    1000,
            plot_axes=  ['mot_hidden_width','act_exploration'],
            #loglevel=   10,
            #do_TB=      False,
        )