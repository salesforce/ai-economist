import numpy as np

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)

@component_registry.add
class UBI(BaseComponent):
    """Gives each mobile agent a constant amount of coin.

    Args:
        amt_min (float): Minimum UBI payment amount. Must be >= 0 (default).
        amt_max (float): Maximum UBI payment amount. Default is 50 (arbitrary).
        amt_disc (float): The interval separating discrete UBI payment amounts
            that the planner can select. Default is 5 (arbitrary).
        period (int): Length of a period in environment timesteps. UBI is distributed
            at the end of each period. Must be > 0. Default is 100 timesteps.
    """

    name = "UBI"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]


    def __init__(
        self,
        *base_component_args,
        amt_min=0,
        amt_max=50, # TODO: make this non-arbitrary
        amt_disc=5, # TODO: make this non-arbitrary
        period=100,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # Min and max UBI payment amounts
        self.amt_min = amt_min
        assert self.amt_min >= 0
        self.amt_max = amt_max

        # Size of each discrete step
        self.amt_disc = amt_disc

        # Discrete options for the amount of each UBI payment 
        self.disc_amts = np.arange(
            self.amt_min, self.amt_max + self.amt_disc, self.amt_disc
        )
        self.disc_amts = self.disc_amts[self.disc_amts <= self.amt_max]
        assert len(self.disc_amts) > 1 
        
        # Number of discrete options for the amount of each UBI payment
        self.n_disc_amts = len(self.disc_amts)
        

        # Current discrete option for the size of each UBI payment
        self.curr_amt_index = 0
        
        # Amount of coin to give to each agent
        self.basic_income_level = self.disc_amts[self.curr_amt_index]

        # Record of the amount of UBI payment in the previous period
        self.last_basic_income_level = None

        # Record of UBI payments at each time step (for dense log)
        self.ubi_payments = []

        # How many timesteps a "year" lasts (period over which UBI is paid once)
        self.period = int(period)
        assert self.period > 0

        # Track the UBI payment cycle
        self.ubi_cycle_pos = 1

        # Cache planner masks
        self._planner_masks = None

    def set_new_period_amt_model(self):
        """Update UBI payment amount using actions from the planner agent."""

        # AI version
        planner_action = self.world.planner.get_component_action(
            self.name, "UBI_payment_amount"
        )
        # When NO-OP, set UBI as 0 (cannot do nothing because agent may 
        # not be able to afford the same UBI rate as last time)
        if planner_action == 0:
            self.curr_amt_index = 0
        elif planner_action <= self.n_disc_amts:
            self.curr_amt_index = int(planner_action - 1)
        else:
            raise ValueError

    def pay_ubi(self):
        """Pay UBI from the planner agent to each mobile agent"""
        world = self.world
        
        # pay UBI to mobile agents
        for agent in world.agents:
            agent.state["inventory"]["Coin"] += self.basic_income_level
        # remove UBI costs from planner agent
        world.planner.state['inventory']['Coin'] -= self.basic_income_level * len(world.agents)

        # record UBI level
        self.last_basic_income_level = self.basic_income_level
        self.ubi_payments.append(dict(ubi_amt=self.basic_income_level))
    """
    Required methods for implementing components
    --------------------------------------------
    """
    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.
        The planner's action space includes an action subspace for UBI. 
        The action space has as many actions as there are discretized tax rates.
        """
        # Only the planner takes actions through this component
        if agent_cls_name == "BasicPlanner":
            # The planner can select one of the discretized UBI amounts.
            return [
                ("UBI_payment_amount", self.n_disc_amts)
            ]

        # Return 0 (no added actions) if the other conditions aren't met
        return 0

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any state fields."""
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        Gives all agents a set amount of coin.
        """
        
        # 1. On the first day of a new period: Set up the UBI for this period.
        if self.ubi_cycle_pos == 1:
            self.set_new_period_amt_model()
            self.basic_income_level = self.disc_amts[self.curr_amt_index]

        # Provide UBI every period
        if self.ubi_cycle_pos == self.period:      
            self.pay_ubi()
            self.ubi_cycle_pos = 0
        else:
            self.ubi_payments.append([])
        self.ubi_cycle_pos += 1

    def generate_observations(self):
        """
        See base_component.py for detailed description.
        Agents observe where in the UBI period cycle they are, the
        last period's UBI amount, and the current period's UBI amount.
        The planner observes the same information.
        """
        is_ubi_day = float(self.ubi_cycle_pos >= self.period)
        is_first_day = float(self.ubi_cycle_pos == 1)
        ubi_phase = self.ubi_cycle_pos / self.period

        obs = dict()

        obs[self.world.planner.idx] = dict(
            is_ubi_day=is_ubi_day,
            is_first_day=is_first_day,
            ubi_phase=ubi_phase,
            curr_amt=self.basic_income_level,
            last_amt=self.last_basic_income_level,
        )

        for agent in self.world.agents:
            i = agent.idx
            k = str(i)

            obs[k] = dict(
                is_ubi_day=is_ubi_day,
                is_first_day=is_first_day,
                ubi_phase=ubi_phase,
                curr_amt=self.basic_income_level,
                last_amt=self.last_basic_income_level,
            )

        return obs


    def generate_masks(self, completions=0): # TODO: SHL
        """
        See base_component.py for detailed description.
        Masks only apply to the planner.
        All UBI actions are masked (so, only NO-OPs can be sampled) on all timesteps
        except when self.ubi_cycle_pos==1 (meaning a new UBI period is starting).
        """

        if self._planner_masks is None:
            masks = super().generate_masks(completions=completions)
            self._planner_masks = dict(
                zeros={
                    k: np.zeros_like(v)
                    for k, v in masks[self.world.planner.idx].items()
                },
            )
        
        # No need to recompute the zeros mask. Use the cached masks.
        masks = dict()
        if self.ubi_cycle_pos != 1:
            # Apply zero masks for any timestep where UBI
            # is not going to be updated.
            masks[self.world.planner.idx] = self._planner_masks["zeros"]
        else:
            # Mask possible UBI amounts when setting new UBI 
            planner_coin = self.world.planner.state['inventory']['Coin']
            masks[self.world.planner.idx] = {'UBI_payment_amount': (self.disc_amts*len(self.world.agents) <= planner_coin).astype(int)}

        return masks

    # For non-required customization
    # ------------------------------

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.
        Reset trackers.
        """
        self.curr_amt_index = 0
        self.basic_income_level = 0
        self.last_basic_income_level = 0
        self.ubi_cycle_pos = 1
        self.ubi_payments = []
        self._planner_masks = None

    def get_dense_log(self):
        """
        Log UBI payments.
        Returns:
            ubi_payments (list): A list of UBI payments. Each entry corresponds to a single
                timestep. Entries are empty except for timesteps where a UBI period
                ended and UBI was distributed. For those timesteps, each entry
                contains the UBI payment amount.
                Returns None if taxes are disabled.
        """
        return self.ubi_payments