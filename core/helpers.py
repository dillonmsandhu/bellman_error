# helpers.py
# This file contains technical helpers used for the RL loop, including GAE and trace computation, PPO loss, and environment initialization.
from core.imports import *
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper
from envs.log_wrapper import LogWrapper
from envs.wrappers import (NormalizeObservationWrapper, NormalizeRewardWrapper, 
AddChannelWrapper, ClipAction, NormalizeRewardEnvState, NormalizeObsEnvState, 
TerminalInfoWrapper, MountainCarNormalizeWrapper, MountainCarSparseRewardWrapper)
from envs.boyan_chain import MatrixMockEnv, BoyanParams
from envs.whirlpool import WhirlpoolExactValue
from envs.whirlpool_env import Whirlpool
from gymnax.environments import spaces
from flax.core import unfreeze, freeze

def create_evaluator(config, env=None, env_params=None):
    from envs.fourrooms import (
        FourRoomsExactValue,
        FourRoomsDenseExactValue,
        FourRoomsMinesExactValue,
        FourRoomsMinesDenseExactValue,
    )
    from envs.fourrooms_continuing import ContinuingFourRooms
    from envs.eightrooms import (
        EightRoomsExactValue,
        ContinuingEightRooms,
        EightRoomsDenseExactValue,
        ContinuingEightRoomsDense,
    )
    from envs.boyan_chain import ContinuingBoyanRing
    from envs.whirlpool import WhirlpoolExactValue, ContinuingWhirlpool
    from envs.mountaincar_exact import MountainCarExactValue

    env_name = config['ENV_NAME'].lower()

    if env_name in ['fourrooms', 'fourrooms-misc']:
        return FourRoomsExactValue(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (11, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['fourrooms-dense', 'fourrooms_dense', 'fourroomsdense', 'fourrooms-misc-dense']:
        return FourRoomsDenseExactValue(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (11, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name in ['fourrooms-mines', 'fourrooms_mines', 'fourroom-mines', 'fourrooms-misc-mines']:
        return FourRoomsMinesExactValue(
            start_pos=getattr(env, 'pos_fixed', config.get('START_POS', (3, 1))),
            goal_pos=getattr(env, 'goal_fixed', config.get('GOAL_POS', (11, 11))),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            mine_locations=config.get('MINE_LOCATIONS', None),
            mine_reward=config.get('MINE_REWARD', 0.0),
        )
    elif env_name in ['fourrooms-mines-dense', 'fourrooms_mines_dense', 'fourroom-mines-dense']:
        return FourRoomsMinesDenseExactValue(
            start_pos=getattr(env, 'pos_fixed', config.get('START_POS', (3, 1))),
            goal_pos=getattr(env, 'goal_fixed', config.get('GOAL_POS', (11, 11))),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            mine_locations=config.get('MINE_LOCATIONS', None),
            mine_reward=config.get('MINE_REWARD', 0.0),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name in ['continuous-fourrooms', 'continuous_fourrooms', 'continuousfourrooms', 'continuous-fourrooms-misc', 'fourrooms-continuous']:
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        return FourRoomsExactValue(
            start_pos=(3, 1),
            goal_pos=(11, 11),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.0)),
            gamma=config['GAMMA'],
            use_visual_obs=use_visual,
        )
    elif env_name in ['continuous-fourrooms-dense', 'continuous_fourrooms_dense', 'continuousfourroomsdense', 'continuous-fourrooms-misc-dense', 'fourrooms-continuous-dense']:
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        return FourRoomsDenseExactValue(
            start_pos=(3, 1),
            goal_pos=(11, 11),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.0)),
            gamma=config['GAMMA'],
            use_visual_obs=use_visual,
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name == 'fourrooms-cont':
        return ContinuingFourRooms(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (11, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['eightrooms', 'eightrooms-misc']:
        return EightRoomsExactValue(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (23, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['eightrooms-cont']:
        return ContinuingEightRooms(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (23, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['eightrooms-dense', 'eightrooms_dense', 'eightroomsdense', 'eightrooms-misc-dense']:
        return EightRoomsDenseExactValue(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (23, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name in ['eightrooms-dense-cont', 'eightrooms_dense_cont', 'eightroomsdense-cont']:
        return ContinuingEightRoomsDense(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (23, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name in ['continuous-eightrooms', 'continuous_eightrooms', 'continuouseightrooms', 'continuous-eightrooms-misc', 'eightrooms-continuous']:
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        return EightRoomsExactValue(
            start_pos=(3, 1),
            goal_pos=(23, 11),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.0)),
            gamma=config['GAMMA'],
            use_visual_obs=use_visual,
        )
    elif env_name in ['continuous-eightrooms-dense', 'continuous_eightrooms_dense', 'continuouseightroomsdense', 'continuous-eightrooms-misc-dense', 'eightrooms-continuous-dense']:
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        return EightRoomsDenseExactValue(
            start_pos=(3, 1),
            goal_pos=(23, 11),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.0)),
            gamma=config['GAMMA'],
            use_visual_obs=use_visual,
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name == 'boyan':
        return ContinuingBoyanRing(
            size=config.get('ENV_SIZE', 21),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['whirlpool', 'whirlpool-misc']:
        return WhirlpoolExactValue(
            size=config.get('ENV_SIZE', 21),
            gamma=config['GAMMA'],
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.5)),
            start_pos=getattr(env, 'pos_fixed', None),
            goal_pos=getattr(env, 'goal_fixed', None),
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['whirlpool-cont']:
        return ContinuingWhirlpool(
            size=config.get('ENV_SIZE', 20),
            gamma=config['GAMMA'],
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.5)),
            start_pos=getattr(env, 'pos_fixed', None),
            goal_pos=getattr(env, 'goal_fixed', None),
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['spaceinvadersexactvalue', 'spaceinvaders', 'spaceinvaders-exact']:
        from envs.space_invaders import SpaceInvadersExactValue
        return SpaceInvadersExactValue(
            width=config.get('SPACE_INVADERS_WIDTH', 7),
            height=config.get('SPACE_INVADERS_HEIGHT', 6),
            num_aliens=config.get('SPACE_INVADERS_NUM_ALIENS', 3),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            endless=config.get('SPACE_INVADERS_ENDLESS', True),
        )
    elif env_name in ['mountaincar', 'mountaincar-v0']:
        from envs.mountaincar_exact import MountainCarExactValue
        return MountainCarExactValue(
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', False),
            goal_reward=config.get('GOAL_REWARD', 1.0),
        )
    elif env_name in ['mountaincar-dense', 'mountaincardense']:
        from envs.mountaincar_exact import MountainCarDenseExactValue
        return MountainCarDenseExactValue(
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', False),
            potential_scale=config.get('POTENTIAL_SCALE', 20.0/100.0),
            goal_reward=config.get('GOAL_REWARD', 1.0),
        )
    return None

def initialize_evaluator(config, env, env_params):
    if not config.get("CALC_TRUE_VALUES", False):
        return None
    if hasattr(env, "evaluator") and env.evaluator is not None:
        return env.evaluator
    return create_evaluator(config, env, env_params)

def make_env(config):
    env_name = config['ENV_NAME'].lower()
    use_tabular = config.get("USE_TABULAR_SIMULATOR", True)

    tabular_env_names = [
        'whirlpool', 'whirlpool-misc', 'whirlpool-cont',
        'fourrooms', 'fourrooms-misc', 'fourrooms-cont', 'fourrooms-dense',
        'fourrooms-mines', 'fourrooms_mines', 'fourroom-mines', 'fourrooms-misc-mines',
        'fourrooms-mines-dense', 'fourrooms_mines_dense', 'fourroom-mines-dense',
        'eightrooms', 'eightrooms-misc', 'eightrooms-cont',
        'eightrooms-dense', 'eightrooms_dense', 'eightroomsdense', 'eightrooms-misc-dense',
        'eightrooms-dense-cont', 'eightrooms_dense_cont',
        'boyan',
        'spaceinvadersexactvalue', 'spaceinvaders', 'spaceinvaders-exact',
        'mountaincar', 'mountaincar-v0', 'mountaincar-dense', 'mountaincardense',
    ]

    if use_tabular and env_name in tabular_env_names:
        from envs.tabular_matrix_env import TabularMatrixEnv, TabularParams
        evaluator = create_evaluator(config)
        env = TabularMatrixEnv(evaluator, name=config['ENV_NAME'])
        env_params = TabularParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1e6)),
            fail_prob=getattr(evaluator, 'fail_prob', 0.0),
        )
        env = TerminalInfoWrapper(env)
        if '-cont' in env_name:
            from envs.wrappers import ContinuingWrapper
            env = ContinuingWrapper(env)

    elif env_name in ['mountaincar-v0', 'mountaincar', 'mountaincar-dense']:
        env, env_params = gymnax.make('MountainCar-v0')
        env_params = env_params.replace(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1e6))
        )
        env = TerminalInfoWrapper(env)
        env = MountainCarNormalizeWrapper(env)
        if 'dense' in env_name:
            from envs.wrappers import MountainCarDenseRewardWrapper
            env = MountainCarDenseRewardWrapper(
                env,
                potential_scale=config.get('POTENTIAL_SCALE', 20.0),
                goal_reward=config.get('GOAL_REWARD', 100.0),
                gamma=config['GAMMA'],
            )
        else:
            env = MountainCarSparseRewardWrapper(env, goal_reward=config.get('GOAL_REWARD', 100.0))

    elif config['ENV_NAME'] == 'FourRooms-misc':
        env, env_params = gymnax.make(config["ENV_NAME"], use_visual_obs=True, goal_fixed=(11,11), pos_fixed = (3,1))
        env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE'], 
            fail_prob=config.get('FAIL_PROB', 0.1)
        )
        env = TerminalInfoWrapper(env)
        
    elif config['ENV_NAME'] == 'FourRooms-cont':
        from envs.wrappers import ContinuingWrapper
        env, env_params = gymnax.make('FourRooms-misc', use_visual_obs=True, goal_fixed=(11,11), pos_fixed = (3,1))
        env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE'], 
            fail_prob=config.get('FAIL_PROB', 0.1)
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    elif config['ENV_NAME'].lower() in ['eightrooms', 'eightrooms-misc']:
        from envs.eightrooms import EightRooms, EightRoomsParams
        env = EightRooms(use_visual_obs=True, goal_fixed=(23, 11), pos_fixed=(3, 1))
        env_params = EightRoomsParams(
            max_steps_in_episode=config.get('MAX_STEPS_IN_EPISODE', 1e6),
            fail_prob=config.get('FAIL_PROB', 0.01)
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower() in ['eightrooms-cont']:
        from envs.eightrooms import EightRooms, EightRoomsParams
        from envs.wrappers import ContinuingWrapper
        env = EightRooms(use_visual_obs=True, goal_fixed=(23, 11), pos_fixed=(3, 1))
        env_params = EightRoomsParams(
            max_steps_in_episode=config.get('MAX_STEPS_IN_EPISODE', 1e6),
            fail_prob=config.get('FAIL_PROB', 0.01)
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    elif config['ENV_NAME'].lower() in ['eightrooms-dense', 'eightrooms_dense', 'eightroomsdense', 'eightrooms-misc-dense']:
        from envs.eightrooms import EightRoomsDense, EightRoomsDenseParams
        env = EightRoomsDense(
            use_visual_obs=True,
            goal_fixed=(23, 11),
            pos_fixed=(3, 1),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 1.0),
        )
        env_params = EightRoomsDenseParams(
            max_steps_in_episode=config.get('MAX_STEPS_IN_EPISODE', 1e6),
            fail_prob=config.get('FAIL_PROB', 0.01),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 1.0),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower() in ['eightrooms-dense-cont', 'eightrooms_dense_cont', 'eightroomsdense-cont']:
        from envs.eightrooms import EightRoomsDense, EightRoomsDenseParams
        from envs.wrappers import ContinuingWrapper
        env = EightRoomsDense(
            use_visual_obs=True,
            goal_fixed=(23, 11),
            pos_fixed=(3, 1),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 1.0),
        )
        env_params = EightRoomsDenseParams(
            max_steps_in_episode=config.get('MAX_STEPS_IN_EPISODE', 1e6),
            fail_prob=config.get('FAIL_PROB', 0.01),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 1.0),
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuouseightrooms',
        'continuouseightroomsmisc',
        'eightroomscontinuous',
        'eightroomscontcontrol',
    ]:
        from envs.continuous_eightrooms import ContinuousEightRooms, ContinuousEightRoomsParams
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousEightRooms(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (23.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
        )
        env_params = ContinuousEightRoomsParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuouseightroomscont',
        'eightroomscontinuouscont',
    ]:
        from envs.continuous_eightrooms import ContinuousEightRooms, ContinuousEightRoomsParams
        from envs.wrappers import ContinuingWrapper
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousEightRooms(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (23.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
        )
        env_params = ContinuousEightRoomsParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuouseightroomsdense',
        'continuouseightroomsmiscdense',
        'eightroomscontinuousdense',
    ]:
        from envs.continuous_eightrooms import ContinuousEightRoomsDense, ContinuousEightRoomsDenseParams
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousEightRoomsDense(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (23.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env_params = ContinuousEightRoomsDenseParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuouseightroomsdensecont',
        'eightroomscontinuousdensecont',
    ]:
        from envs.continuous_eightrooms import ContinuousEightRoomsDense, ContinuousEightRoomsDenseParams
        from envs.wrappers import ContinuingWrapper
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousEightRoomsDense(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (23.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env_params = ContinuousEightRoomsDenseParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuousfourrooms',
        'continuousfourroomsmisc',
        'fourroomscontinuous',
        'fourroomscontcontrol',
    ]:
        from envs.continuous_fourrooms import ContinuousFourRooms, ContinuousFourRoomsParams
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousFourRooms(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (11.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
        )
        env_params = ContinuousFourRoomsParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuousfourroomsdense',
        'continuousfourroomsmiscdense',
        'fourroomscontinuousdense',
    ]:
        from envs.continuous_fourrooms import ContinuousFourRoomsDense, ContinuousFourRoomsDenseParams
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousFourRoomsDense(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (11.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env_params = ContinuousFourRoomsDenseParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'] == 'boyan':
        # Create our lightweight mock primitives right here
        env = MatrixMockEnv(size=20, use_visual_obs=config.get("USE_VISUAL_OBS", True))
        env_params = BoyanParams(
            fail_prob=0.0, 
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE']
        )
    elif config['ENV_NAME'].lower() in ['whirlpool', 'whirlpool-misc']:
        from envs.whirlpool_env import Whirlpool, EnvParams
        env = Whirlpool(size=config.get('ENV_SIZE', 21), use_visual_obs=True)
        env_params = EnvParams(
            fail_prob=config.get('FAIL_PROB', 0.5),
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1e6)),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower() in ['whirlpool-cont']:
        from envs.whirlpool_env import Whirlpool, EnvParams
        from envs.wrappers import ContinuingWrapper
        env = Whirlpool(size=config.get('ENV_SIZE', 21), use_visual_obs=True)
        env_params = EnvParams(
            fail_prob=config.get('FAIL_PROB', 0.5),
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1e6)),
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
    
    print('Env:', config['ENV_NAME'])
    print('Default Obs Shape:', env.observation_space(env_params).shape)
    
    env = LogWrapper(env)
    
    if isinstance(env.action_space(env_params), spaces.Box):
        env = ClipAction(env)
    
    if config.get("NETWORK_TYPE") == "mlp":
        if len(env.observation_space(env_params).shape) > 1:
            env = FlattenObservationWrapper(env)
    if config.get("NETWORK_TYPE") == "cnn":
        if len(env.observation_space(env_params).shape) < 3:
            env = AddChannelWrapper(env)
    if config.get("NORMALIZE_OBS", False):
        env = NormalizeObservationWrapper(env) 
    
    print('Obs Shape:', env.observation_space(env_params).shape)
    print('Action Shape:', env.action_space(env_params).shape)
    return env, env_params
    
def _loss_fn(params, network, traj_batch, gae, targets, config):
    # Critic loss
    value_loss = v_loss_fn(params, network, traj_batch, gae, targets, config)

    # Actor loss
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        config.get('POLICY_COEFF', 1.0) * loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)

def _loss_fn_no_w(params, network, traj_batch, gae, targets, config):
    # Critic loss
    value_loss = no_w_v_loss_fn(params, network, traj_batch, gae, targets, config)

    # Actor loss
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)    

def post_process_advantage(advantages, config, weights=None):
    """
    Standardizes and clips advantages for PPO policy optimization.
    
    If weights are provided (e.g. w = mu[:-1, None] * old_pi for exact methods),
    computes the weighted mean and weighted standard deviation over state-action visitation.
    Otherwise (e.g. sampled rollouts in standard PPO and hybrid scripts),
    computes the unweighted sample mean and standard deviation.
    
    Supports soft floor (ADV_STD_FLOOR) to prevent noise explosion near initialization,
    and outlier clipping (ADV_CLIP).
    """
    std_floor = config.get("ADV_STD_FLOOR", 0.1)
    adv_clip = config.get("ADV_CLIP", 3.0)

    if weights is not None:
        if weights.ndim == 1 and advantages.ndim == 2:
            weights = weights[:, None]
        w = weights / jnp.sum(weights)
        mean_adv = jnp.sum(w * advantages)
        var_adv = jnp.sum(w * (advantages - mean_adv) ** 2)
        std_adv = jnp.sqrt(var_adv + 1e-8)
    else:
        mean_adv = jnp.mean(advantages)
        var_adv = jnp.var(advantages)
        std_adv = jnp.sqrt(var_adv + 1e-8)

    if std_floor is not None and std_floor > 0.0:
        denom = jnp.maximum(std_adv, std_floor)
    else:
        denom = std_adv + 1e-8

    adv_norm = (advantages - mean_adv) / denom

    if adv_clip is not None and adv_clip > 0.0:
        adv_norm = jnp.clip(adv_norm, -adv_clip, adv_clip)

    return jax.lax.stop_gradient(adv_norm)


def compute_exact_advantage(P, R, P_pi, R_pi, v, γ, λ):
    """Computes exact GAE advantages across all non-terminal states and actions."""
    # δ_gae = (I - γ * λ * P_pi)^(-1) δ is the discounted sum of TD errors from step 1 onward.
    # At step 0, action a has immediate TD error δ(s, a) = R(s, a) + γ * v(s') - v(s).
    # Future TD errors from step 1 onward are discounted by γ * λ:
    # A^GAE(s, a) = δ + γ * λ * E_{s'}[δ_gae(s')]
    #             = R(s, a) + γ * E_{s'}[v(s') + λ * δ_gae(s')] - v(s)
    I = jnp.eye(P_pi.shape[0])
    L_pi = jnp.linalg.inv(I - γ * λ * P_pi)
    δ = R_pi + γ * (P_pi @ v) - v
    δ_gae = L_pi @ δ

    R_sa = jnp.einsum("sam,sam->sa", P[:-1], R[:-1])
    Q_sa = R_sa + γ * jnp.einsum("sam,m->sa", P[:-1], v + λ * δ_gae)
    return Q_sa - v[:-1, None]


def pi_loss_fn(params, network, traj_batch, gae, config):
    pi = network.apply(params, traj_batch.obs, method=network.policy)
    log_prob = pi.log_prob(traj_batch.action)

    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = post_process_advantage(gae, config)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - config["CLIP_EPS"],
            1.0 + config["CLIP_EPS"],
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    return loss_actor, entropy


def ppo_clipped_v_loss(traj_batch, value_pred, targets, config):
    e = config.get("VF_CLIP", None)
    if e is None or e is False or (isinstance(e, (int, float)) and e <= 0):
        return 0.5 * jnp.mean(jnp.square(value_pred - targets))
    value_pred_clipped = traj_batch.value + (
        value_pred - traj_batch.value
    ).clip(-e, e)
    value_losses = jnp.square(value_pred - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    
def v_loss_fn(params, network, traj_batch, gae, targets, config):
    # VALUE LOSS
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    value_loss = ppo_clipped_v_loss(traj_batch, value_pred, targets, config)
    total_loss = config["VF_COEF"] * value_loss
    return total_loss

def v_loss_fn_laplacian_smoothing(params, network, traj_batch, targets, config):
    gamma = config["GAMMA"]
    c = config["VF_CLIP"]
    # 1. Current State Predictions & Errors (e_i)
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    value_pred_clipped = traj_batch.value + (
        value_pred - traj_batch.value).clip(-c,c)
    e_i = targets - value_pred_clipped 
    base_ve_loss = 0.5 * jnp.mean(e_i ** 2)
    # 2. Next State Predictions & Errors (e_j)
    # Requires traj_batch to contain the adjacent (s, G) pairs
    next_value_pred = network.apply(params, traj_batch.next_obs, method=network.value)
    next_v_fixed = jax.lax.stop_gradient(next_value_pred)
    next_value_pred_clipped = next_v_fixed + (
        next_value_pred - next_v_fixed).clip(-c,c)
    e_j = traj_batch.next_target - next_value_pred_clipped
    valid_mask = 1.0 - traj_batch.done
    n_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)
    laplacian_loss = 0.5 * jnp.sum(valid_mask * (e_i - e_j) ** 2) / n_valid
    # Combine using the exact Dirichlet expansion weights
    weight_laplacian = gamma * config['LAPLACE_SMOOTHING_COEFF']
    dirichlet_value_loss = (1 - weight_laplacian) * base_ve_loss + weight_laplacian * laplacian_loss
    total_loss = config["VF_COEF"] * dirichlet_value_loss
    return total_loss, {"base_ve_loss": base_ve_loss, "laplacian_loss": laplacian_loss, "total_loss": total_loss}

def no_w_v_loss_fn(params, network, traj_batch, gae, targets, config):
    # ---------------------------------------------------------
    # Firewalled Parameters
    # ---------------------------------------------------------
    def freeze_w_map(path, val):
        is_w = any(getattr(p, 'key', None) in ('w_layer', 'critic_head') or 
                   'w_layer' in str(p) or 'critic_head' in str(p) for p in path)
        return jax.lax.stop_gradient(val) if is_w else val
    
    params_w_frozen = jax.tree_util.tree_map_with_path(freeze_w_map, params)
    value_for_phi = network.apply(params_w_frozen, traj_batch.obs, method = network.value)
    loss_phi = ppo_clipped_v_loss(traj_batch, value_for_phi, targets, config)
    return config["VF_COEF"] * loss_phi

def v_loss_fn_no_grad(params, network, traj_batch, gae, targets, config):
    "No update to phi."
    # 1. Forward pass through the CNN to get the features
    phi = network.apply(params, traj_batch.obs, method=network.value_features)
    
    # 2. SEVER THE GRAPH: Gradients from the value loss cannot pass this point.
    # The CNN weights will receive zero gradient from this loss function.
    phi_freeze = jax.lax.stop_gradient(phi)
    
    # 3. Forward pass through ONLY the linear head using the frozen features
    value_pred = network.apply(params, phi_freeze, method=network.value_from_features)
    value_loss = ppo_clipped_v_loss(traj_batch, value_pred, targets, config)
    total_loss = config["VF_COEF"] * value_loss
    return total_loss

def shuffle_and_batch(rng, transitions, n_minibatches):
    def preprocess_transition(x, rng):
        x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs (batch_size), ...
        x = jax.random.permutation(rng, x)  # shuffle the transitions
        x = x.reshape(n_minibatches, -1, *x.shape[1:])  # num_mini_updates, batch_size/num_mini_updates, ...
        return x
    minibatches = jax.tree.map(lambda x: preprocess_transition(x, rng), transitions)  # num_actors*num_envs (batch_size), ...
    return minibatches


def add_values_to_metric(config, metric, evaluator, network, train_state, traj_batch, compute_true_vals = True):
    """Uses evaluator to compute the per-state quantities and append them to metric."""
    if evaluator:
        pi, v_pred = network.apply(train_state.params, evaluator.obs_stack)
        pi = jnp.vstack([pi, jnp.zeros((1, pi.shape[-1]))]) # assumes terminal state.
        
        Φ = network.apply(train_state.params, evaluator.obs_stack, method=network.value_features)
        Φ = jnp.vstack([Φ, jnp.zeros((1, Φ.shape[-1]))])  # assumes terminal state.
        
        v_pred = network.apply(train_state.params, Φ, method=network.value_from_features)
    
    # True value
    if compute_true_vals:
        # The evaluator dictates the exact ground truth shapes here
        v = evaluator.compute_true_values(pi)

    # 4. Visitation Logic
    obs = jnp.asarray(traj_batch.obs)
    next_obs = jnp.asarray(traj_batch.next_obs)
    env_name = config.get("ENV_NAME", "")
    
    if env_name in {"FourRooms-misc", "FourRoomsCustom-v0"} or "SparseMaze" in env_name:
        if obs.ndim >= 5:
            metric['visitation_count'] = next_obs[..., 1].sum(axis=(0, 1))
        elif obs.ndim >= 3 and obs.shape[-1] >= 2:
            size = traj_batch.reward.shape[0] 
            pos = next_obs[..., :2].astype(jnp.int32)
            y = jnp.clip(pos[..., 0], 0, size - 1).reshape(-1)
            x = jnp.clip(pos[..., 1], 0, size - 1).reshape(-1)
            counts = jnp.zeros((size, size), dtype=jnp.float32)
            metric['visitation_count'] = counts.at[y, x].add(1.0)
    
    # 5. Error Metrics (Perfect shape alignment guaranteed by the evaluator)    
    metric.update({
        "v": v,
        "v_pred": v_pred,
        "pi": pi,
        "Empirical MSVE": jnp.mean((v - v_pred)**2) ,
    })
    
    return metric


# def calculate_gae(traj_batch, γ, λ,):

#     def _get_advantages(gae, transition):
#         done = transition.done

#         delta = transition.reward + γ * transition.next_value * (1 - done) - transition.value
#         gae = delta + (γ * λ * (1 - done) * gae)
        
#         return gae, gae

#     initial_accs = jnp.zeros_like(traj_batch.value[0])
#     _, advantages = jax.lax.scan(
#         _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
#     )
    
#     return (advantages, advantages + traj_batch.value)

def calculate_gae(traj_batch, γ, λ):
    def _get_advantages(gae, transition):
        done = transition.done
        is_timeout = transition.info["is_timeout"]

        # MASK 1: Value Bootstrapping
        true_terminal = done & ~is_timeout
        bootstrap_mask = 1.0 - true_terminal

        # MASK 2: GAE Accumulation (Trajectory Boundary)
        # Sever the GAE chain if the environment reset for ANY reason (terminal or timeout).
        # The 'gae' variable coming from the future belongs to a different episode.
        boundary_mask = 1.0 - done

        # 1. Compute TD Error (Safely bootstraps through timeouts)
        delta = transition.reward + γ * transition.next_value * bootstrap_mask - transition.value
        
        # 2. Accumulate GAE (Safely breaks at episode resets)
        gae = delta + (γ * λ * boundary_mask * gae)
        
        return gae, gae

    initial_accs = jnp.zeros_like(traj_batch.value[0])
    _, advantages = jax.lax.scan(
        _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
    )
    
    return (advantages, advantages + traj_batch.value)


def find_closest_divisor(total, requested):
    for n in range(requested, 0, -1):
        if total % n == 0:
            return n
    return 1

def inject_weights(train_state, w):
    """Overwrites the critic_head weights, preserving the original PyTree type."""
    # 1. Slice the weights (last dim is bias)
    kernel_weights = jnp.expand_dims(w[:-1], axis=-1)
    bias_weight = w[-1:]
    
    # 2. Define the new layer dictionary
    new_head = {
        'kernel': kernel_weights,
        'bias': bias_weight
    }
    
    # 3. Inject it while preserving the container type (dict vs frozendict)
    params = train_state.params
    if not isinstance(params, dict):
        params = unfreeze(params)
    
    new_params = dict(params)
    new_params['params'] = dict(new_params['params'])
    if 'critic_head' in new_params['params']:
        new_params['params']['critic_head'] = new_head
    if 'w_layer' in new_params['params']:
        new_params['params']['w_layer'] = new_head

    if isinstance(train_state.params, dict):
        return train_state.replace(params=new_params)
    else:
        return train_state.replace(params=freeze(new_params))

def get_evaluation_policies(base_config, evaluator):
    """Gets a target policy for evaluation. Returns (policy_fn, policy_matrix):
        - policy_fn: a function from obs to action distribution
        - policy_matrix: a matrix of size |S| x |A| of the policy
        This function constructs these functions for either an epsilon-optimal policy or a trained actor network.
    """
    if base_config.get("USE_GREEDY_POLICY", False):
        import core.bellman_error as bellman_error
        import distrax
        if hasattr(evaluator, "get_optimal_value_function"):
            V_star = evaluator.get_optimal_value_function()
        else:
            V_star = jnp.zeros(evaluator.num_total_states)
        
        greedy_actions = bellman_error.compute_greedy_policy(evaluator.P, evaluator.R, evaluator.gamma, V_star)
        pi_greedy = jax.nn.one_hot(greedy_actions, evaluator.num_actions)
        epsilon = base_config.get("POLICY_EPSILON", 0.0)
        pi_eps = (1 - epsilon) * pi_greedy + (epsilon / evaluator.num_actions) * jnp.ones_like(pi_greedy)
        
        def policy_fn(obs):
            obs_flat = obs.reshape((obs.shape[0], -1)) if obs.ndim > 1 else obs.flatten()[None, :]
            stack_flat = evaluator.obs_stack.reshape((evaluator.obs_stack.shape[0], -1))
            diffs = jnp.sum((obs_flat[:, None, :] - stack_flat[None, :, :])**2, axis=-1)
            state_indices = jnp.argmin(diffs, axis=-1)
            probs = pi_eps[state_indices]
            return distrax.Categorical(probs=probs)
            
        # pi_eps already has shape (num_total_states, A), we just need to ensure the terminal state is uniform
        pi_eps = pi_eps.at[-1, :].set(jnp.ones(evaluator.num_actions) / evaluator.num_actions)

        is_continuous_env = base_config.get("ENV_NAME", "").lower().replace('_', '').replace('-', '') in [
            'continuouseightrooms',
            'continuouseightroomsdense',
            'continuouseightroomsmisc',
            'continuouseightroomsmiscdense',
            'eightroomscontinuous',
            'eightroomscontinuousdense',
            'continuouseightroomscont',
            'eightroomscontinuouscont',
        ]
        if is_continuous_env:
            from flax import struct
            action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)

            @struct.dataclass
            class CardinalContinuousPolicy:
                cat_dist: Any
                basis: jax.Array

                def sample(self, seed=None):
                    idx = self.cat_dist.sample(seed=seed)
                    return self.basis[idx]

                def log_prob(self, value):
                    diffs = jnp.sum((value[..., None, :] - self.basis)**2, axis=-1)
                    idx = jnp.argmin(diffs, axis=-1)
                    return self.cat_dist.log_prob(idx)

                def mode(self):
                    return self.basis[self.cat_dist.mode()]

            orig_policy_fn = policy_fn
            policy_fn = lambda obs: CardinalContinuousPolicy(cat_dist=orig_policy_fn(obs), basis=action_basis)

        return policy_fn, pi_eps
    else:
        import core.utils as utils
        from scripts.sweep_pipeline import resolve_model_load_dir
        resolved_dir = resolve_model_load_dir(base_config.get('MODEL_LOAD_DIR'), base_config['ENV_NAME'], 'results')
        model_dir = 'ppo/' + resolved_dir if not resolved_dir.startswith('ppo/') else resolved_dir
        print(model_dir)
        _, out = utils.load_run_data(model_dir, base_config['ENV_NAME'], 'results') 
        policy_train_state = out['runner_state'][0]
        policy_params = jax.tree_util.tree_map(lambda x: x[0], policy_train_state.params)
        
        def policy_fn(obs):
            pi, _ = policy_train_state.apply_fn(policy_params, obs)
            # handle cases where apply_fn returns a tuple (pi, value) or just pi
            if isinstance(pi, tuple):
                pi = pi[0]
            return pi
            
        # build the matrix
        pi_dist = policy_fn(evaluator.obs_stack)
        if hasattr(pi_dist, "probs"):
            pi_probs = pi_dist.probs
        else:
            action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)
            log_p = jax.vmap(lambda a: pi_dist.log_prob(a), in_axes=0, out_axes=-1)(action_basis)
            pi_probs = jax.nn.softmax(log_p, axis=-1)
        terminal_policy = jnp.ones([1, evaluator.num_actions], dtype=pi_probs.dtype) / evaluator.num_actions
        policy_matrix = jnp.vstack([pi_probs, terminal_policy])
        
        return policy_fn, policy_matrix


# ==============================================================================
# Sampled E and E(lambda) Critic Losses & Helpers
# ==============================================================================

def e_lambda_fixed_loss_fn(params, network, traj_batch, gae, targets, config):
    """
    Unclipped critic MSE loss + clipped PPO actor loss for E_lambda_fixed.
    Targets are precomputed symmetrized E(lambda) targets.
    """
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    value_loss = 0.5 * jnp.mean(jnp.square(value_pred - targets))
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        config.get("POLICY_COEFF", 1.0) * loss_actor
        + config.get("VF_COEF", 0.5) * value_loss
        - config.get("ENT_COEF", 0.01) * entropy
    )
    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "actor_loss": loss_actor,
        "entropy": entropy,
    }
    return total_loss, losses


def e_critic_loss(v_i, targets_i, v_j, targets_j, true_terminal, gamma):
    """
    Computes sampled E-loss (magnitude anchor + Dirichlet/Laplacian smoothness + boundary correction).
    - For ongoing transitions and timeouts: smooths e_i against e_j.
    - For true terminal transitions (true_terminal=True): absorbing state error is 0,
      so (e_i - e_j)^2 = (e_i - 0)^2 = e_i^2.
    - Boundary correction: 0.5 * gamma * (mean(e_i^2) - mean(e_j^2)) to match the exact matrix E objective.
    """
    e_i = targets_i - v_i
    e_j = jnp.where(true_terminal, 0.0, targets_j - v_j)

    magnitude_loss = (1.0 - gamma) * jnp.mean(e_i ** 2)
    laplacian_loss = 0.5 * gamma * jnp.mean((e_i - e_j) ** 2)
    corr_loss = 0.5 * gamma * (jnp.mean(e_i ** 2) - jnp.mean(e_j ** 2))

    value_loss = magnitude_loss + laplacian_loss + corr_loss
    return value_loss, magnitude_loss, laplacian_loss


def ppo_actor_loss(params, network, obs_mb, action_mb, log_prob_mb, advantages_mb, config):
    """
    Standard PPO actor clipped surrogate objective with entropy bonus.
    """
    pi = network.apply(params, obs_mb, method=network.policy)
    log_prob = pi.log_prob(action_mb)
    entropy = pi.entropy().mean()
    ratio = jnp.exp(log_prob - log_prob_mb)

    adv_norm = post_process_advantage(advantages_mb, config)

    surr1 = ratio * adv_norm
    surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    total_loss = actor_loss - entropy * config.get("ENT_COEF", 0.01)
    return total_loss, {
        "actor_loss": actor_loss,
        "entropy": entropy,
    }


def critic_td_lambda_loss(params, network, obs_mb, targets_mb, value_mb, config):
    """
    TD(lambda) critic loss with optional value function clipping.
    Requires precomputed lambda targets (e.g. from helpers.calculate_gae).
    """
    value_pred = network.apply(params, obs_mb, method=network.value)
    if config.get("VF_CLIP", 0.0) > 0:
        e = config["VF_CLIP"]
        value_pred_clipped = value_mb + (value_pred - value_mb).clip(-e, e)
        value_losses = jnp.square(value_pred - targets_mb)
        value_losses_clipped = jnp.square(value_pred_clipped - targets_mb)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    else:
        value_loss = 0.5 * jnp.mean((value_pred - targets_mb) ** 2)
    return value_loss, {"value_loss": value_loss}


def critic_td_zero_loss(params, network, obs_mb, next_obs_mb, reward_mb, mask_mb, value_mb, config):
    """
    TD(0) critic loss computed from scratch each minibatch using inference on v(s') and stop_gradient.
    Requires next_obs, reward, and bootstrap_mask (1 - (done & ~is_timeout)).
    """
    value_pred = network.apply(params, obs_mb, method=network.value)
    next_value_pred = network.apply(params, next_obs_mb, method=network.value)
    td_target = reward_mb + config["GAMMA"] * mask_mb * next_value_pred
    td_target = jax.lax.stop_gradient(td_target)

    if config.get("VF_CLIP", 0.0) > 0:
        e = config["VF_CLIP"]
        value_pred_clipped = value_mb + (value_pred - value_mb).clip(-e, e)
        value_losses = jnp.square(value_pred - td_target)
        value_losses_clipped = jnp.square(value_pred_clipped - td_target)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    else:
        value_loss = 0.5 * jnp.mean((value_pred - td_target) ** 2)
    return value_loss, {"value_loss": value_loss}


def critic_sampled_e_loss(params, network, obs_mb, next_obs_mb, targets_mb, next_target_mb, true_terminal_mb, config):
    """
    Sampled E-loss for critic (magnitude anchor + Dirichlet/Laplacian smoothness + boundary correction).
    Requires targets and aligned next_targets.
    """
    gamma = config["GAMMA"]
    v_i = network.apply(params, obs_mb, method=network.value)
    v_j = network.apply(params, next_obs_mb, method=network.value)
    value_loss, magnitude_loss, laplacian_loss = e_critic_loss(
        v_i, targets_mb, v_j, next_target_mb, true_terminal_mb, gamma
    )
    return value_loss, {
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "laplacian_loss": laplacian_loss,
    }


def e_loss_fn(
    params,
    network,
    obs,
    action,
    log_prob_old,
    next_obs,
    true_terminal,
    next_target,
    advantages,
    targets,
    config,
):
    """
    Combined loss for PPO with Sampled E critic.
    """
    # 1. Actor Loss (PPO clipped surrogate)
    pi = network.apply(params, obs, method=network.policy)
    log_prob = pi.log_prob(action)
    entropy = pi.entropy().mean()
    ratio = jnp.exp(log_prob - log_prob_old)

    adv_norm = post_process_advantage(advantages, config)
    surr1 = ratio * adv_norm
    surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    # 2. Critic Loss (Sampled E-loss)
    gamma = config["GAMMA"]
    v_i = network.apply(params, obs, method=network.value)
    v_j = network.apply(params, next_obs, method=network.value)
    # Terminal absorbing state has value 0
    v_j = jnp.where(true_terminal, 0.0, v_j)

    value_loss, magnitude_loss, laplacian_loss = e_critic_loss(
        v_i, targets, v_j, next_target, true_terminal, gamma
    )

    total_loss = (
        config["POLICY_COEFF"] * actor_loss
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "laplacian_loss": laplacian_loss,
        "actor_loss": actor_loss,
        "entropy": entropy,
    }
    return total_loss, losses


def calculate_e_lambda_targets(
    traj_batch,
    gamma: float,
    lmbda: float,
    return_lambda: float = 1.0,
    next_value_T: jnp.ndarray = None,
):
    """
    Computes scalar value regression targets for the symmetrized E(lambda) objective
    using forward and backward error traces:
        y_t = G_t - sg[ 0.5 * gamma * (1 - lambda) * (e_{>t} + e_{<t}) ]
    """
    # 1. Baseline returns G_t and errors e_t = G_t - v_t
    _, returns = calculate_gae(traj_batch, gamma, return_lambda)
    errors = returns - traj_batch.value
    gl = gamma * lmbda

    is_timeout = traj_batch.info.get("is_timeout", jnp.zeros_like(traj_batch.done, dtype=bool))
    true_terminal = traj_batch.done & ~is_timeout
    done_f = traj_batch.done.astype(jnp.float32)
    true_term_f = true_terminal.astype(jnp.float32)

    # Continuation error at rollout buffer boundary step T
    if next_value_T is None:
        next_value_T = traj_batch.next_value[-1]
    e_T = jnp.where(true_terminal[-1], 0.0, traj_batch.next_value[-1] - next_value_T)
    errors_ext = jnp.concatenate([errors, e_T[None]], axis=0)
    e_next = errors_ext[1:]

    # 2. Forward Trace: Accumulate future errors backwards in time
    def _backward_pass(forward_trace, transition):
        done_t, true_term_t, e_nxt = transition
        valid_future = 1.0 - done_t
        trace_t = jnp.where(
            true_term_t > 0.5,
            0.0,
            e_nxt + gl * valid_future * forward_trace,
        )
        return trace_t, trace_t

    init_forward = jnp.zeros_like(errors[0])
    _, forward_traces = jax.lax.scan(
        _backward_pass,
        init_forward,
        (done_f, true_term_f, e_next),
        reverse=True,
    )

    # 3. Backward Trace: Accumulate past errors forwards in time
    prev_done = jnp.roll(done_f, shift=1, axis=0).at[0].set(1.0)
    e_prev = jnp.roll(errors, shift=1, axis=0).at[0].set(0.0)

    def _forward_pass(backward_trace, transition):
        p_done, e_prv = transition
        trace_t = (1.0 - p_done) * (e_prv + gl * backward_trace)
        return trace_t, trace_t

    init_backward = jnp.zeros_like(errors[0])
    _, backward_traces = jax.lax.scan(
        _forward_pass,
        init_backward,
        (prev_done, e_prev),
        reverse=False,
    )

    # 4. Construct Symmetrized E(lambda) scalar regression targets
    coeff = 0.5 * gamma * (1.0 - lmbda)
    smoothing_correction = coeff * (forward_traces + backward_traces)
    targets = returns - jax.lax.stop_gradient(smoothing_correction)

    diagnostics = {
        "returns": returns,
        "errors": errors,
        "forward_traces": forward_traces,
        "backward_traces": backward_traces,
        "correction": smoothing_correction,
    }
    return targets, diagnostics


def e_lambda_differentiable_critic_loss(
    values,
    targets,
    dones,
    gamma: float,
    lmbda: float,
    true_terminals=None,
    next_value_T=None,
    next_target_T=None,
):
    """
    Computes differentiable E(lambda) loss using backward moment traces (Method 2).
    Backpropagates through both values v(s_t) and future values v(s_{t+k+1}).
    """
    if true_terminals is None:
        true_terminals = dones
    if next_value_T is None:
        next_value_T = jnp.zeros_like(values[0])
    if next_target_T is None:
        next_target_T = jnp.zeros_like(targets[0])

    errors = targets - values
    gl = gamma * lmbda
    done_f = dones.astype(jnp.float32)
    true_term_f = true_terminals.astype(jnp.float32)

    # Continuation error at rollout buffer boundary step T
    e_T = jnp.where(true_terminals[-1], 0.0, next_target_T - next_value_T)
    errors_ext = jnp.concatenate([errors, e_T[None]], axis=0)
    e_next = errors_ext[1:]

    def _moment_step(traces, transition):
        w0_future, w1_future, w2_future = traces
        done_t, true_term_t, e_curr, e_nxt = transition

        valid_future = 1.0 - done_t

        w0_t = jnp.where(true_term_t > 0.5, 1.0, 1.0 + gl * valid_future * w0_future)
        w1_t = jnp.where(true_term_t > 0.5, 0.0, e_nxt + gl * valid_future * w1_future)
        w2_t = jnp.where(true_term_t > 0.5, 0.0, (e_nxt ** 2) + gl * valid_future * w2_future)

        # Dirichlet quadratic expansion: w0 * e_t^2 - 2 * w1 * e_t + w2
        dirichlet_t = w0_t * (e_curr ** 2) - 2.0 * w1_t * e_curr + w2_t

        # Traces passed backward to step t-1 are severed if episode ended (done_t == 1)
        valid_to_prev = 1.0 - done_t
        w0_to_prev = valid_to_prev * w0_t
        w1_to_prev = valid_to_prev * w1_t
        w2_to_prev = valid_to_prev * w2_t

        return (w0_to_prev, w1_to_prev, w2_to_prev), dirichlet_t

    init_traces = (
        jnp.zeros_like(errors[0]),
        jnp.zeros_like(errors[0]),
        jnp.zeros_like(errors[0]),
    )
    _, dirichlet_terms = jax.lax.scan(
        _moment_step,
        init_traces,
        (done_f, true_term_f, errors, e_next),
        reverse=True,
    )

    magnitude_weight = (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * gamma * (1.0 - lmbda)

    magnitude_loss = magnitude_weight * jnp.mean(errors ** 2)
    dirichlet_loss = dirichlet_weight * jnp.mean(dirichlet_terms)
    value_loss = magnitude_loss + dirichlet_loss

    return value_loss, magnitude_loss, dirichlet_loss


def shuffle_and_batch_envs(rng, batch, n_minibatches):
    """
    Shuffles environments (columns) and splits into minibatches,
    preserving full trajectory sequences along the time axis (axis 0).
    """
    sample_leaf = jax.tree.leaves(batch)[0]
    num_envs = sample_leaf.shape[1]
    n_minibatches = max(1, min(n_minibatches, num_envs))
    env_per_mb = num_envs // n_minibatches

    perm = jax.random.permutation(rng, num_envs)

    def _split_leaf(x):
        x_shuffled = jnp.take(x, perm, axis=1)
        x_trimmed = x_shuffled[:, : env_per_mb * n_minibatches]
        reshaped = x_trimmed.reshape(x.shape[0], n_minibatches, env_per_mb, *x.shape[2:])
        return jnp.swapaxes(reshaped, 0, 1)

    return jax.tree.map(_split_leaf, batch)


def e_lambda_differentiable_loss_fn(
    params,
    network,
    traj_batch,
    advantages,
    returns,
    config,
):
    """
    Combined PPO loss with Method 2 differentiable E(lambda) critic loss.
    """
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, advantages, config)

    values = network.apply(params, traj_batch.obs, method=network.value)
    gamma = config["GAMMA"]
    e_lambda = config["VALUE_LAMBDA"]

    # Boundary continuation at step T
    next_value_T = network.apply(params, traj_batch.next_obs[-1], method=network.value)
    next_target_T = traj_batch.next_value[-1]

    # Terminals vs Timeouts
    is_timeout = traj_batch.info.get("is_timeout", jnp.zeros_like(traj_batch.done, dtype=bool))
    true_terminal = traj_batch.done & ~is_timeout

    value_loss, magnitude_loss, dirichlet_loss = e_lambda_differentiable_critic_loss(
        values=values,
        targets=returns,
        dones=traj_batch.done,
        gamma=gamma,
        lmbda=e_lambda,
        true_terminals=true_terminal,
        next_value_T=next_value_T,
        next_target_T=next_target_T,
    )

    total_loss = (
        config["POLICY_COEFF"] * loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )

    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "actor_loss": loss_actor,
        "entropy": entropy,
    }
    return total_loss, losses


def e_lambda_geometric_critic_loss(
    values,
    targets,
    dones,
    gamma: float,
    lmbda: float,
    rng: jax.Array,
    true_terminals=None,
    next_value_T=None,
    next_target_T=None,
):
    """
    Computes sampled E(lambda) critic loss by sampling lookahead horizon skips
    K ~ Geometric(1 - gamma * lambda) directly from the compound transition matrix
    P_lambda = (1 - gamma * lambda) * sum_{k=0}^infty (gamma * lambda)^k P^{k+1} (Method 3).
    """
    if true_terminals is None:
        true_terminals = dones
    if next_value_T is None:
        next_value_T = jnp.zeros_like(values[0])
    if next_target_T is None:
        next_target_T = jnp.zeros_like(targets[0])

    T, B = targets.shape[:2]
    errors = targets - values
    gl = gamma * lmbda

    # Continuation error at rollout buffer boundary step T
    e_T = jnp.where(true_terminals[-1], 0.0, next_target_T - next_value_T)
    errors_ext = jnp.concatenate([errors, e_T[None]], axis=0)

    # 1. Sample jump lengths K ~ Geometric(1 - gl) with K >= 0
    # For gl < 1e-6 (e.g. lambda = 0), K = 0 deterministically
    u = jax.random.uniform(rng, shape=(T, B))
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    jumps = jnp.maximum(jumps, 0)

    # 2. Target lookahead index and episode boundary masking
    t_arr = jnp.arange(T)[:, None]
    target_idx = t_arr + jumps + 1
    within_chunk = target_idx <= T
    t_clamped = jnp.minimum(target_idx, T)

    t_prev = jnp.minimum(t_arr + jumps, T - 1)
    done_cumsum = jnp.cumsum(dones.astype(jnp.int32), axis=0)
    has_reset_between = (
        jnp.take_along_axis(done_cumsum, t_prev, axis=0) - done_cumsum
    ) > 0

    valid_jump = within_chunk & (~has_reset_between)
    e_jump = jnp.take_along_axis(errors_ext, t_clamped, axis=0)

    is_timeout = dones & (~true_terminals)
    diff_sq = jnp.where(
        true_terminals,
        errors ** 2,
        jnp.where(
            is_timeout,
            jnp.where(jumps == 0, (errors - e_jump) ** 2, 0.0),
            jnp.where(valid_jump, (errors - e_jump) ** 2, 0.0),
        ),
    )

    # 4. Weighting
    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    magnitude_weight = (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * tilde_gamma

    magnitude_loss = magnitude_weight * jnp.mean(errors ** 2)
    dirichlet_loss = dirichlet_weight * jnp.mean(diff_sq)
    value_loss = magnitude_loss + dirichlet_loss

    return value_loss, magnitude_loss, dirichlet_loss


def e_lambda_geometric_loss_fn(
    params,
    network,
    traj_batch,
    advantages,
    returns,
    config,
    rng,
):
    """
    Combined PPO loss with Method 3 sampled geometric E(lambda) critic loss.
    """
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, advantages, config)

    values = network.apply(params, traj_batch.obs, method=network.value)
    gamma = config["GAMMA"]
    e_lambda = config["VALUE_LAMBDA"]

    # Boundary continuation at step T
    next_value_T = network.apply(params, traj_batch.next_obs[-1], method=network.value)
    next_target_T = traj_batch.next_value[-1]

    # Terminals vs Timeouts
    is_timeout = traj_batch.info.get("is_timeout", jnp.zeros_like(traj_batch.done, dtype=bool))
    true_terminal = traj_batch.done & ~is_timeout

    value_loss, magnitude_loss, dirichlet_loss = e_lambda_geometric_critic_loss(
        values=values,
        targets=returns,
        dones=traj_batch.done,
        gamma=gamma,
        lmbda=e_lambda,
        rng=rng,
        true_terminals=true_terminal,
        next_value_T=next_value_T,
        next_target_T=next_target_T,
    )

    total_loss = (
        config["POLICY_COEFF"] * loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )

    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "actor_loss": loss_actor,
        "entropy": entropy,
    }
    return total_loss, losses


# Re-export compute_runtime_metrics for convenience
try:
    from core.runtime_metrics import compute_runtime_metrics
except ImportError:
    pass

