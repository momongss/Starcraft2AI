#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run SC2 to play a game or a replay."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import platform
import sys
import time

from absl import app
from absl import flags
from pysc2 import maps
from pysc2 import run_configs
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import renderer_human
from pysc2.lib import replay
from pysc2.lib import stopwatch

from pysc2.lib import features

from s2clientprotocol import sc2api_pb2 as sc_pb

from pysc2.agents.model_agent import ModelAgent
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_bool("realtime", False, "Whether to run in realtime mode.")
flags.DEFINE_bool("full_screen", False, "Whether to run full screen.")

flags.DEFINE_float("fps", 22.4, "Frames per second to run the game.")
flags.DEFINE_integer("step_mul", 1, "Game steps per observation.")
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
point_flag.DEFINE_point("feature_screen_size", "128",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "128",
                        "Resolution for minimap feature layers.")
flags.DEFINE_integer("feature_camera_width", 24,
                     "Width of the feature layer camera.")
point_flag.DEFINE_point("rgb_screen_size", "256,192",
                        "Resolution for rendered screen.")
flags.DEFINE_bool("use_feature_units", True,
                  "Whether to include feature units.")
point_flag.DEFINE_point("rgb_minimap_size", "128",
                        "Resolution for rendered minimap.")
point_flag.DEFINE_point("window_size", "640,480",
                        "Screen size if not full screen.")
flags.DEFINE_string("video", None, "Path to render a video of observations.")
flags.DEFINE_integer("max_game_steps", 0, "Total game steps to run.")
flags.DEFINE_integer("max_episode_steps", 0, "Total game steps per episode.")

flags.DEFINE_string("user_name", getpass.getuser(),
                    "Name of the human player for replays.")
flags.DEFINE_enum("user_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "User's race.")
flags.DEFINE_enum("bot_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "AI race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "Bot's strength.")
flags.DEFINE_enum("bot_build", "random", sc2_env.BotBuild._member_names_,  # pylint: disable=protected-access
                  "Bot's build strategy.")
flags.DEFINE_bool("disable_fog", False, "Disable fog of war.")
flags.DEFINE_integer("observed_player", 1, "Which player to observe.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use to play.")
flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")

flags.DEFINE_string("map_path", None, "Override the map for this replay.")
flags.DEFINE_string("replay", "C:\\Users\\rkrp1\\Desktop\\종합설계\\# 리플레이 만들기\\Simple64_2020-07-03-04-16-13.SC2Replay", "Name of a replay to show.")


def main(unused_argv):
  """Run SC2 to play a game or a replay."""
  if FLAGS.trace:
    stopwatch.sw.trace()
  elif FLAGS.profile:
    stopwatch.sw.enable()

  if (FLAGS.map and FLAGS.replay) or (not FLAGS.map and not FLAGS.replay):
    sys.exit("Must supply either a map or replay.")

  if FLAGS.replay and not FLAGS.replay.lower().endswith("sc2replay"):
    sys.exit("Replay must end in .SC2Replay.")

  if FLAGS.realtime and FLAGS.replay:
    # TODO(tewalds): Support realtime in replays once the game supports it.
    sys.exit("realtime isn't possible for replays yet.")

  if FLAGS.render and (FLAGS.realtime or FLAGS.full_screen):
    sys.exit("disable pygame rendering if you want realtime or full_screen.")

  if platform.system() == "Linux" and (FLAGS.realtime or FLAGS.full_screen):
    sys.exit("realtime and full_screen only make sense on Windows/MacOS.")

  if not FLAGS.render and FLAGS.render_sync:
    sys.exit("render_sync only makes sense with pygame rendering on.")

  run_config = run_configs.get()

  interface = sc_pb.InterfaceOptions()
  interface.raw = FLAGS.render
  interface.raw_affects_selection = True
  interface.raw_crop_to_playable_area = True
  interface.score = True
  interface.show_cloaked = True
  interface.show_burrowed_shadows = True
  interface.show_placeholders = True
  if FLAGS.feature_screen_size and FLAGS.feature_minimap_size:
    interface.feature_layer.width = FLAGS.feature_camera_width
    FLAGS.feature_screen_size.assign_to(interface.feature_layer.resolution)
    FLAGS.feature_minimap_size.assign_to(
        interface.feature_layer.minimap_resolution)
    interface.feature_layer.crop_to_playable_area = True
    interface.feature_layer.allow_cheating_layers = True
  if FLAGS.render and FLAGS.rgb_screen_size and FLAGS.rgb_minimap_size:
    FLAGS.rgb_screen_size.assign_to(interface.render.resolution)
    FLAGS.rgb_minimap_size.assign_to(interface.render.minimap_resolution)

  max_episode_steps = FLAGS.max_episode_steps

  if FLAGS.map:
    create = sc_pb.RequestCreateGame(
        realtime=FLAGS.realtime,
        disable_fog=FLAGS.disable_fog)
    try:
      map_inst = maps.get(FLAGS.map)
    except maps.lib.NoMapError:
      if FLAGS.battle_net_map:
        create.battlenet_map_name = FLAGS.map
      else:
        raise
    else:
      if map_inst.game_steps_per_episode:
        max_episode_steps = map_inst.game_steps_per_episode
      if FLAGS.battle_net_map:
        create.battlenet_map_name = map_inst.battle_net
      else:
        create.local_map.map_path = map_inst.path
        create.local_map.map_data = map_inst.data(run_config)

    create.player_setup.add(type=sc_pb.Participant)
    create.player_setup.add(type=sc_pb.Computer,
                            race=sc2_env.Race[FLAGS.bot_race],
                            difficulty=sc2_env.Difficulty[FLAGS.difficulty],
                            ai_build=sc2_env.BotBuild[FLAGS.bot_build])
    join = sc_pb.RequestJoinGame(
        options=interface, race=sc2_env.Race[FLAGS.user_race],
        player_name=FLAGS.user_name)
    version = None
  else:
    replay_data = run_config.replay_data(FLAGS.replay)
    start_replay = sc_pb.RequestStartReplay(
        replay_data=replay_data,
        options=interface,
        disable_fog=FLAGS.disable_fog,
        observed_player_id=FLAGS.observed_player)
    version = replay.get_replay_version(replay_data)
    run_config = run_configs.get(version=version)  # Replace the run config.

  with run_config.start(
      full_screen=FLAGS.full_screen,
      window_size=FLAGS.window_size,
      want_rgb=interface.HasField("render")) as controller:
    if FLAGS.map:
      controller.create_game(create)
      controller.join_game(join)
    else:
      info = controller.replay_info(replay_data)
      print(" Replay info ".center(60, "-"))
      print(info)
      print("-" * 60)
      map_path = FLAGS.map_path or info.local_map_path
      if map_path:
        start_replay.map_data = run_config.map_data(map_path,
                                                    len(info.player_info))
      controller.start_replay(start_replay)

    if FLAGS.render:
      renderer = renderer_human.RendererHuman(
          fps=FLAGS.fps, step_mul=FLAGS.step_mul,
          render_sync=FLAGS.render_sync, video=FLAGS.video)
      renderer.run(
          run_config, controller, max_game_steps=FLAGS.max_game_steps,
          game_steps_per_episode=max_episode_steps,
          save_replay=FLAGS.save_replay)
    else:  # Still step forward so the Mac/Windows renderer works.
      try:
        agent = ModelAgent()
        f = features.features_from_game_info(game_info=controller.game_info(),
                                             agent_interface_format=sc2_env.parse_agent_interface_format(
                                                 feature_screen=128,
                                                 feature_minimap=128,
                                                 rgb_screen=None,
                                                 rgb_minimap=None,
                                                 action_space=None,
                                                 use_feature_units=True,
                                                 use_raw_units=False),
                                             useRGB=False)
        replay_save = {}
        while True:
          if not FLAGS.realtime:
            controller.step(FLAGS.step_mul)
          obs = controller.observe()
          agent_obs = f.transform_obs(obs)
          func = 0
          args = []
          if obs.actions:
              func_call = f.reverse_action(obs.actions[0])
              func, args = func_call.function.real, func_call.arguments
              for y in range(len(args)):
                  for x in range(len(args[y])):
                      if type(args[y][x]) != int:
                          args[y][x] = args[y][x].real
          if func != 0:
              print(agent_obs['game_loop'], func, args)
              print(agent_obs['feature_units'])
              selected_obs = agent.select_obs(agent_obs)
              replay_save[agent_obs['game_loop'][0][0]] = [func, args, selected_obs]
          if obs.player_result:
            with open('good_replay.txt', 'wb') as fk:
                pickle.dump(replay_save, fk)
            break
      except KeyboardInterrupt:
        pass
      print("Score: ", obs.observation.score.score)
      print("Result: ", obs.player_result)
      if FLAGS.map and FLAGS.save_replay:
        replay_save_loc = run_config.save_replay(
            controller.save_replay(), "local", FLAGS.map)
        print("Replay saved to:", replay_save_loc)
        # Save scores so we know how the human player did.
        with open(replay_save_loc.replace("SC2Replay", "txt"), "w") as f:
          f.write("{}\n".format(obs.observation.score.score))

  if FLAGS.profile:
    print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
  app.run(main)


if __name__ == "__main__":
  app.run(main)
