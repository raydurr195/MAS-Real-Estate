from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
import pandas as pd
from ToyMarket import ToyMarket
from Callbacks import customcall

#from ray.rllib.core.rl_module.rl_module import RLModuleSpec
#from ray.rllib.models.catalog import ModelCatalog

env_name = "toymarket-v1"




# Creates the Werewolf environment
def env_creator(config):
    env = ToyMarket(num_buyer=config["num_buyer"], num_seller = config['num_seller'], money = config['money'], t = config['t'])
    env.reset()  # Ensure agents are properly initialized
    para_env = ParallelPettingZooEnv(env)
    para_env.agents = para_env.par_env.agents
    para_env.possible_agents = para_env.par_env.possible_agents
    return para_env

register_env(env_name, lambda config: env_creator(config))

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "buyer_policy" if "buyer" in agent_id else "seller_policy"

# Initialize environment
env = ToyMarket()
obs_spaces = {agent: env.observation_space(agent) for agent in env.possible_agents}
act_spaces = {agent: env.action_space(agent) for agent in env.possible_agents}

# rl_module_spec = RLModuleSpec(
#     observation_space=obs_spaces["player_0"],
#     action_space=act_spaces["player_0"],
#     #catalog_class=ModelCatalog,
#     model_config={
#         "fcnet_hiddens": [256, 256], 
#         "fcnet_activation": "relu"
#     }
# )

# Update policy specification
config = (
    PPOConfig()
    .environment(env=env_name, env_config={"num_buyer": 2, "num_seller": 2, "money": [1000,1000,1000,1000], "t": 5})
    .framework("torch")  # Use PyTorch backend
    .env_runners(num_env_runners=2)  # Number of workers
    .training( 
        train_batch_size=1024,
        minibatch_size=128,  
        num_epochs=10  
    )
    .multi_agent( 
        policies={
            "buyer_policy": (None, obs_spaces["buyer_0"], act_spaces["buyer_0"], {}),
            "seller_policy": (None, obs_spaces["seller_0"], act_spaces["seller_0"], {})
        },
        policy_mapping_fn=policy_mapping_fn
    )
   # .rl_module(
    #    rl_module_spec=rl_module_spec
    #)
    .api_stack(
        enable_rl_module_and_learner=False, 
        enable_env_runner_and_connector_v2=False
    )
    .callbacks(
        customcall
    )
)

# Training
analysis = tune.run(
    "PPO",
    name="toymarket_training",
    stop={"training_iteration": 10},
    config=config.to_dict(),
    checkpoint_at_end=True,
    checkpoint_freq=1,
    keep_checkpoints_num = 10,
)

print(analysis.results_df.head())
print(analysis.results_df.columns)