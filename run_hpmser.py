from hpmser.search_function import hpmser
from pypaq.mpython.devices import DevicesPypaq

from applied_RL.run_training import run_actor_training, train_configs


global_const = {
    # train kwargs
    'num_updates':      500,
    'test_freq':        20,
    'test_episodes':    10,
    'test_render':      False,
    'inspect':          False,
    'break_ntests':     2,
    # run_actor_training kwargs
    # ..not needed any, hpmser_mode sets rest
}

# parameters of run_wrap
hpmser_configs = {

    'CP_A2C_PT': {
        'train_psdd': {
            'a_two_towers':         (True, False),
            'a_num_layers':         [1, 3],
            'a_layer_width':        (10, 20, 50, 100, 150),
            'a_lay_norm':           (True, False),
            'a_clamp_advantage':    (0.3, 0.5, 0.7, 1.0, None),
            'a_use_scaled_ce':      (True, False),
            'a_use_huber':          (True, False),
            'a_baseLR':             (0.03, 0.01, 0.005, 0.001, 0.0005, 0.0001),
            'a_do_clip':            (True, False),
            't_batch_size':         (32, 64, 128, 256, 512, 768),
            't_exploration':        (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5),
            't_train_sampled':      [0.0, 1.0],
            't_discount':           [0.6, 0.99],
            't_use_mavg':           (True, False),
            't_mavg_factor':        [0.01, 0.4],
            't_do_zscore':          (True, False),
        },
        'func_const': {
            'e_max_steps':          200,
        }
    },

}

# updates nested dict (train_configs[tr_preset]) with flat kwargs, prepares and returns proper score
def run_wrap(
        tr_preset: str,
        devices: DevicesPypaq,
        hpmser_mode=    True,
        **kwargs # kwargs that start with specific prefix go to one of mdicts
) -> float:

    pd = train_configs[tr_preset]

    envy_mdict = {}
    actor_mdict = {
        'devices':      devices,
        'hpmser_mode':  hpmser_mode}
    trainer_mdict = {
        'hpmser_mode':  hpmser_mode}
    other_kwargs = {}
    for param in kwargs:

        param_from_mdicts = False
        if param.startswith('e_'):
            envy_mdict[param[2:]] = kwargs[param]
            param_from_mdicts = True
        if param.startswith('a_'):
            actor_mdict[param[2:]] = kwargs[param]
            param_from_mdicts = True
        if param.startswith('t_'):
            trainer_mdict[param[2:]] = kwargs[param]
            param_from_mdicts = True

        if not param_from_mdicts:  other_kwargs[param] = kwargs[param]

    pd.update(global_const)
    pd['envy_mdict'].update(envy_mdict)
    pd['actor_mdict'].update(actor_mdict)
    pd['trainer_mdict'].update(trainer_mdict)
    pd.update(other_kwargs)
    pd['hpmser_mode'] = hpmser_mode

    tr_res = run_actor_training(**pd)
    n_actions = tr_res["n_actions"]
    succeeded_row_max = tr_res["succeeded_row_max"]
    if succeeded_row_max < pd['break_ntests']: return 0

    return 1 / n_actions



if __name__ == "__main__":


    for config_name in [
        #'SBG_QTable',
        #'SBG_DQN_TF',
        #'SBG_DQN_PT',
        #'CP_PG_TF',
        #'CP_PG_PT',
        #'CP_AC_TF',
        #'CP_A2C_TF',
        'CP_A2C_PT',
        #'CP_ACShared_TF',
        #'ACR_AC_TF',
    ]:

        func_const = hpmser_configs[config_name]['func_const']
        func_const['tr_preset'] = config_name

        hpmser(
            func=       run_wrap,
            func_psdd=  hpmser_configs[config_name]['train_psdd'],
            func_const= func_const,
            devices=    [None]*10,
            #do_TB=      False,
        )