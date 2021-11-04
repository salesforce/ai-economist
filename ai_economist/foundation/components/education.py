import numpy as np
from ai_economist.foundation.base.base_component import BaseComponent, component_registry
@component_registry.add

class GetEducated(BaseComponent):
  name = "GetEducated"
  required_entities = ["Coin", "Labor", "Skill"]  # <--- We can now look up "Widget" in the resource registry
  agent_subclasses = ["BasicMobileAgent"]

  def __init__(
      self,
      *base_component_args,
      tuition=100, # same tuition cost as building 10 houses <- tweak later
      education_labor=100.0,
      skill_gain = 10
      **base_component_kwargs
  ):
      super().__init__(*base_component_args, **base_component_kwargs)
      self.tuition = int(tuition)
      self.skill_gain = float(skill_gain)
      assert self.tuition >= 0
      self.education_labor = float(education_labor)
      assert self.education_labor >= 0
      # self.skill = int(skill)
      self.educates = []

  def agent_can_get_educated(self, agent):
    """Return True if agent can actually get educated."""
    # See if the agent has the resources necessary to complete the action
    
    if agent.state["inventory"]["Coin"] < self.tuition:
        return False

    # Do nothing if skill is already max
    if True: # TODO see how to get skill
        return False

    # If we made it here, the agent can go to college.
    return True

  def get_additional_state_fields(self, agent_cls_name):
    if agent_cls_name not in self.agent_subclasses:
        return {}
    if agent_cls_name == "BasicMobileAgent":
        return {"tuition_payment": float(self.tuition)} # check
    raise NotImplementedError

  def additional_reset_steps(self):
      self.available_wood_units = 0

  def get_n_actions(self, agent_cls_name):
      if agent_cls_name == "BasicMobileAgent":
          return 1
      return None

  def generate_masks(self, completions=0):
      masks = {}
      for agent in self.world.agents:
          masks[agent.idx] = np.array([
              agent.state["inventory"]["Coin"] >= self.widget_price and self.available_widget_units > 0
          ])

      return masks

  def component_step(self):
      """
      See base_component.py for detailed description.
      Convert coin to skill for agents that choose to go to school and can.
      """
      
      world = self.world
      build = []
      # Apply any go_to_school actions taken by the mobile agents
      for agent in world.get_random_order_agents():

          action = agent.get_component_action(self.name)

          # This component doesn't apply to this agent!
          if action is None:
              continue

          # NO-OP!
          if action == 0:
              pass

          # Learn! (If you can.)
          elif action == 1:
              if self.agent_can_get_educated(agent):
                  # Remove the resources
                  agent.state["inventory"]["Coin"] -= self.tuition

                  # Receive skills for going to school
                  # agent.state["inventory"]["Coin"] += agent.state["build_payment"]
                  self.payment_max_skill_multiplier += self.skill_gain

                  # Incur the labor cost for going to school
                  agent.state["endogenous"]["Labor"] += self.education_labor

                #   build.append(
                #       {
                #           "student": agent.idx,
                #           "loc": np.array(agent.loc),
                #           "cost": float(agent.state["build_payment"]),
                #       }
                #   )
          else:
              raise ValueError

    #   self.builds.append(build)

  def generate_observations(self):
      obs_dict = dict()
      for agent in self.world.agents:
          obs_dict[agent.idx] = {
              "education_reward": self.skill_gain,
              "education_price": self.tuition
          }

      return obs_dict