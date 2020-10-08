# 깃헙 주소 : https://github.com/deepmind/pysc2
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import random
from absl import app


class TerranAgent(base_agent.BaseAgent):

  def step(self, obs):
    super(TerranAgent, self).step(obs)

    # --use_feature_units 추가
    # scvs = [unit for unit in obs.observation.feature_units
    #         if unit.unit_type == units.Terran.SCV]

    print("일꾼 수 :", obs.observation.player.food_workers)
    print("놀고 있는 일꾼 수 :", obs.observation.player.idle_worker_count)
    print("인구 수 제한 :", obs.observation.player.food_army)
    print("인구 수 제한 :", obs.observation.player.food_cap)
    print(obs.observation.player)

    # 현재 화면에만 존재하는 유닛들만
    scvs = []
    commandcenters = []
    supplydepot = []
    barracks = []
    marines = []

    free_population = obs.observation.player.food_cap - obs.observation.player.food_workers
    if free_population != self.free_population:
      self.on_building_SupplyDepot = False
      self.free_population = free_population

    # --use_feature_units 추가
    for unit in obs.observation.feature_units:
      if unit.unit_type == units.Terran.SCV:
        scvs.append(unit)
      elif unit.unit_type == units.Terran.Marine:
        marines.append(unit)
      elif unit.unit_type == units.Terran.Barracks:
        barracks.append(unit)
      elif unit.unit_type == units.Terran.CommandCenter:
        commandcenters.append(unit)
      elif unit.unit_type == units.Terran.SupplyDepot:
        supplydepot.append(unit)

    if self.unit_type_is_selected(obs, units.Terran.CommandCenter):
      if (actions.FUNCTIONS.Train_SCV_quick.id in obs.observation.available_actions):
        x = random.randint(0, 83)
        y = random.randint(0, 83)
        return actions.FUNCTIONS.Train_SCV_quick("now", (x, y))

    EXPLORE_POPULATION = 20
    if self.unit_type_is_selected(obs, units.Terran.SCV):
      if (actions.FUNCTIONS.Build_SupplyDepot_screen.id in obs.observation.available_actions) and free_population < 10 and not self.on_building_SupplyDepot:
        x = random.randint(0, 83)
        y = random.randint(0, 83)
        self.on_building_SupplyDepot = True
        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", (x, y))

      if not self.on_explore and obs.observation.player.food_workers > EXPLORE_POPULATION:
        pos = self.explore()
        self.on_explore = True
        return actions.FUNCTIONS.Attack_screen("now", (pos[0], pos[1]))

    if len(obs.observation.single_select) == 0 or len(obs.observation.multi_select) == 0:
      if scvs and commandcenters:
        selected_unit = random.choice(scvs+commandcenters)
        if 0 <= selected_unit.x <= 83 and 0 <= selected_unit.y <= 83:
          return actions.FUNCTIONS.select_point("select_all_type", (selected_unit.x, selected_unit.y))

    # if not self.unit_type_is_selected(obs, units.Terran.CommandCenter):
    #   if len(scvs) > 0:
    #     scv_or_command = random.choice(commandcenters)
    #     return actions.FUNCTIONS.select_point("select_all_type", (scv_or_command.x, scv_or_command.y))

    return actions.FUNCTIONS.no_op()

  def explore(self):
    return (random.randint(0, 83), random.randint(0, 83))

  def unit_type_is_selected(self, obs, unit_type):
    if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
      return True
    if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
      return True

    return False