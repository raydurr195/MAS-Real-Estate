from ray.rllib.algorithms.callbacks import DefaultCallbacks

class ToyMarketCall(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        wrapped_env = base_env.get_sub_environments()[0]  # This is ParallelPettingZooEnv
        real_env = wrapped_env.par_env
        episode.custom_metrics['Bids'] = real_env.bid
        episode.custom_metrics['Prices'] = real_env.price
   

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
       
        wrapped_env = base_env.get_sub_environments()[0]  # This is ParallelPettingZooEnv
        real_env = wrapped_env.par_env                    # We access its underlying "par_env", which should be a `ToyMarket` instance.

        episode.custom_metrics["day_end"] = real_env.day