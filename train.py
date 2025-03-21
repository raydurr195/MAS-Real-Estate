from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
import ray
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ToyMarket import ToyMarket
from Callbacks import ToyMarketCall
ray.init()

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



# Update policy specification
config = (
    PPOConfig()
    .environment(
        env=env_name, 
        env_config={
            "num_buyer": 2, 
            "num_seller": 2, 
            "money": [1500,1500,1000,1000], 
            "t": 5
            }
    )
    .callbacks(ToyMarketCall)
    .framework("torch")
    .training(
        train_batch_size=4096,
        minibatch_size=256,
        num_epochs=20,
        lr=1e-4,
        lambda_=0.95,
        vf_loss_coeff=1.0,
        entropy_coeff=0.01,
        clip_param=0.2
    )
     .resources(
         #num_gpus_per_learner=1,
         #num_cpus_per_worker = 1
     )
    .multi_agent(
        policies={
            "buyer_policy": (None, env.observation_space('buyer'), env.action_space('buyer'), {}),
            "seller_policy": (None, env.observation_space('seller'), env.action_space('seller'), {})
        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["buyer_policy","seller_policy"]
    )
    .api_stack(
        enable_rl_module_and_learner=False, 
        enable_env_runner_and_connector_v2=False
    )
    .env_runners(
        num_env_runners = 6,
        num_cpus_per_env_runner = 1
    )
)

# Training with more iterations
tune.run(
    "PPO",
    name="ToyMarket_Training",
    stop={"training_iteration": 2000},
    config=config.to_dict(),
    storage_path="C:/Users/Owner/Desktop/ToyMarket Training",  # Local path in workspace
    checkpoint_freq=100,
    checkpoint_at_end=True,
)

ray.shutdown()