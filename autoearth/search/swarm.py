import copy
from typing import Optional

from fragile.optimize.swarm import FunctionMapper
import numpy as np


class EarthSwarm(FunctionMapper):

    def __init__(self,  best_reward_found: float = -1e10, best_found: Optional[np.ndarray] = None,
                  *args, **kwargs):
        super(EarthSwarm, self).__init__(
            true_best=best_reward_found, true_best_reward=best_found,
            true_best_end=False, *args, **kwargs
        )

    def __repr__(self):
        msg = "Epoch {}\n".format(self.walkers.n_iters)
        msg += "Best reward:{:.3f} Best is valid: {}\n".format(
            self.walkers.states.best_reward_found, self.walkers.states.best_is_alive)

        values = self.env.X.loc[:, self.best_found.astype(bool)].columns
        msg += "Features: {}\n".format(list(values))
        msg += "Features counts {}\n".format(self.walkers.env_states.observs.sum(axis=0))
        msg += "Non zero feature counts {}\n".format(self.walkers.env_states.observs.sum(axis=1))
        rs = self.walkers.states.cum_rewards
        msg += "Rewards: max {:.3f} mean {:.3f} min {:.3f} std {:.3f}\n".format(rs.max(),
                                                                                rs.mean(),
                                                                                rs.min(),
                                                                                rs.std())
        rs = self.walkers.states.will_clone
        ds = self.walkers.env_states.ends
        msg += "Cloned pct {:.3f} dead pct {:.3f} \n".format(rs.sum()/len(rs)*100,
                                                             ds.sum()/len(ds)*100,)
        msg += "current best {}".format(np.array(self.best_found).__repr__())

        return msg

    def calculate_end_condition(self):
        self.walkers.n_iters += 1
        return self.epoch > self.walkers.max_iters

    def run_step(self):
        self.walkers.fix_best()
        self.step_walkers()
        old_ids, new_ids = self.walkers.balance()
        self.prune_tree(old_ids=set(old_ids.tolist()), new_ids=set(new_ids.tolist()))

    def _get_real_best(self):
        best = self.walkers.env_states.observs[-1]
        reward = self.walkers.states.cum_rewards[-1]
        best_end = self.walkers.env_states.ends
        self.walkers.states.update(true_best_reward=reward, true_best=best,
                                   true_best_end=best_end)

    def step_walkers(self):
        """
        Make the walkers undergo a random perturbation process in the swarm \
        Environment.
        """
        model_states = self.walkers.model_states
        env_states = self.walkers.env_states

        states_ids = (
            copy.deepcopy(self.walkers.states.id_walkers).astype(int).flatten().tolist()
            if self._use_tree
            else None
        )

        model_states = self.model.predict(env_states=env_states, model_states=model_states,
                                          walkers_states=self.walkers.states)
        env_states = self.env.step(model_states=model_states, env_states=env_states)
        ends = env_states.ends # if not env_states.ends.all() else np.zeros_like(env_states.ends)
        self.walkers.update_states(
            env_states=env_states, model_states=model_states, end_condition=ends
        )
        self.update_tree(states_ids)