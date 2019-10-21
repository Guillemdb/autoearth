from fragile.core.walkers import Walkers


class EarthWalkers(Walkers):

    def __init__(self, *args, **kwargs):
        super(EarthWalkers, self).__init__(
            best_is_alive=False, *args, **kwargs
        )

    def __repr__(self):
        msg = "Best_reward:{:.3f} Best is valid: {}".format(self.states.best_reward_found,
                                                            self.states.best_is_alive)
        return msg

    def update_best(self):
        ix = self._get_best_index()
        best = self.env_states.observs[ix].copy()
        best_reward = float(self.states.cum_rewards[ix])
        best_is_alive = not bool(self.env_states.ends[ix])
        has_improved = (self.states.best_reward_found > best_reward if self.minimize else
                        self.states.best_reward_found < best_reward)
        improves_and_alive = has_improved and best_is_alive
        improves_over_dead = has_improved and not best_is_alive and not self.states.best_is_alive
        alive_over_dead = best_is_alive and not self.states.best_is_alive

        if improves_and_alive:# or improves_over_dead or alive_over_dead:
            self.states.update(best_reward_found=best_reward)
            self.states.update(best_is_alive=best_is_alive)
            self.states.update(best_found=best)