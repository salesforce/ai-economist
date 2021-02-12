import numpy as np

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)

@component_registry.add
class UBI(BaseComponent):
    """Gives each mobile agent a constant amount of coin.
    """

    name = "UBI"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]


    def __init__(
        self,
        *base_component_args,
        basic_income_level=1, # TODO: update these value
        period=20,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # Amount of coin to give to each agent
        self.basic_income_level = basic_income_level
        assert self.basic_income_level > 0

        # How many timesteps a "year" lasts (period over which UBI is paid once)
        self.period = int(period)
        assert self.period > 0

        # Track the UBI payment cycle
        self.ubi_cycle_pos = 1

    def pay_ubi(self):
        """Pay UBI from the planner agent to each mobile agent"""
        world = self.world

        # pay UBI to mobile agents
        for agent in world.agents:
            agent.state["inventory"]["Coin"] += self.basic_income_level
        # remove UBI costs from planner agent
        world.planner.state['inventory']['Coin'] -= self.basic_income_level * len(world.agents)
        
    """
    Required methods for implementing components
    --------------------------------------------
    """

    def get_n_actions(self, agent_cls_name):
        """This component is passive: it does not add any actions."""
        return

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any state fields."""
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.
        Gives all agents a set amount of coin.
        """
       
        # Provide UBI every period
        if self.ubi_cycle_pos == self.period:      
            self.pay_ubi()
            self.ubi_cycle_pos = 0
        self.ubi_cycle_pos += 1

    def generate_observations(self):
        """This component does not add any observations."""
        obs = {}
        return obs

    def generate_masks(self, completions=0):
        """Passive component. Masks are empty."""
        masks = {}
        return masks