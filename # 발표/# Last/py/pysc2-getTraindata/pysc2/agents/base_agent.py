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
"""A base agent to write custom scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions


class BaseAgent(object):
    """A base agent to write custom scripted agents.

  It can also act as a passive agent that does nothing but no-ops.
  """

    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        self.supply_count = 0
        self.supply_point = [
            [124, 37],
            [102, 48],
            [113, 48],
            [124, 48],
            [91, 37],
            [91, 48],
            [102, 15],
            [113, 15],
        ]

        self.attack_point = [0, 0]

        self.DELAY_TIME = 22
        self.action_delay = 0

        self.ACTION_TRAIN_MARIN = 0
        self.ACTION_BUILD_SUPPLY = 1
        self.ACTION_ATTACK_MINIMAP = 2

        self.special_action = 0
        self.small_step_count = 0
        self.last_used_scv = 3

        self.copystep = 3740
        # self.copystep = 100
        self.copylist = {
            24: [2, [[0], [73, 61]]],
            44: [4, [[1], [5]]],
            94: [2, [[2], [54, 43]]],
            116: [4, [[1], [0]]],
            129: [4, [[0], [5]]],
            202: [490, [[0]]],
            228: [4, [[0], [0]]],
            247: [2, [[0], [60, 49]]],
            254: [4, [[1], [1]]],
            267: [4, [[0], [0]]],
            283: [2, [[0], [45, 44]]],
            290: [4, [[1], [2]]],
            306: [4, [[0], [0]]],
            323: [2, [[0], [53, 33]]],
            332: [4, [[1], [3]]],
            342: [4, [[0], [1]]],
            402: [91, [[0], [102, 26]]],
            448: [451, [[1], [65, 29]]],
            468: [4, [[0], [5]]],
            523: [490, [[0]]],
            804: [490, [[0]]],
            892: [4, [[0], [1]]],
            992: [42, [[0], [105, 66]]],
            1058: [4, [[0], [5]]],
            1077: [490, [[0]]],
            1093: [4, [[0], [1]]],
            1175: [4, [[0], [2]]],
            1283: [42, [[0], [121, 66]]],
            1303: [4, [[0], [5]]],
            1368: [490, [[0]]],
            1427: [4, [[0], [3]]],
            1470: [4, [[0], [5]]],
            1492: [4, [[0], [3]]],
            1604: [42, [[0], [105, 82]]],
            1621: [4, [[0], [5]]],
            1700: [490, [[0]]],
            1747: [4, [[0], [1]]],
            1758: [4, [[0], [2]]],
            1762: [4, [[0], [3]]],
            1876: [4, [[0], [5]]],
            1939: [2, [[2], [105, 66]]],
            1974: [4, [[1], [8]]],
            2007: [4, [[0], [5]]],
            2027: [490, [[0]]],
            2078: [4, [[0], [1]]],
            2114: [91, [[0], [113, 26]]],
            2167: [4, [[0], [3]]],
            2191: [4, [[0], [8]]],
            2206: [477, [[0]]],
            2230: [477, [[0]]],
            2245: [4, [[0], [5]]],
            2318: [309, [[0]]],
            2344: [4, [[0], [3]]],
            2389: [4, [[0], [8]]],
            2407: [477, [[0]]],
            2412: [477, [[0]]],
            2422: [4, [[0], [2]]],
            2465: [4, [[0], [8]]],
            2481: [477, [[0]]],
            2521: [4, [[0], [2]]],
            2698: [42, [[0], [121, 82]]],
            2806: [4, [[0], [8]]],
            2818: [477, [[0]]],
            2823: [477, [[0]]],
            2868: [2, [[2], [105, 66]]],
            2897: [4, [[1], [8]]],
            2921: [477, [[0]]],
            2943: [4, [[0], [1]]],
            3005: [4, [[0], [2]]],
            3049: [91, [[0], [102, 37]]],
            3069: [4, [[0], [1]]],
            3108: [4, [[0], [8]]],
            3146: [477, [[0]]],
            3193: [4, [[0], [1]]],
            3265: [91, [[0], [113, 37]]],
            3324: [4, [[0], [2]]],
            3472: [91, [[0], [124, 26]]],
            3590: [4, [[0], [8]]],
            3606: [477, [[0]]],
            3611: [477, [[0]]],
            3616: [477, [[0]]],
            3622: [477, [[0]]],
            3628: [477, [[0]]],
            3700: [477, [[0]]],
            3732: [7, [[0]]],
            3750: [13, [[0], [112, 99]]],
            3814: [4, [[0], [1]]],
            3855: [91, [[0], [102, 37]]],
            3880: [4, [[0], [8]]],
            3894: [477, [[0]]],
            3946: [477, [[0]]],
            3991: [7, [[0]]],
            4011: [13, [[0], [116, 99]]],
            4073: [4, [[0], [3]]],
            4089: [4, [[0], [8]]],
            4099: [477, [[0]]],
            4109: [477, [[0]]],
            4158: [4, [[0], [5]]],
            4192: [183, [[0], [44, 34]]],
            4235: [4, [[0], [8]]],
            4244: [477, [[0]]],
            4250: [477, [[0]]],
            4268: [7, [[0]]],
            4283: [13, [[0], [113, 96]]],
            4306: [4, [[0], [8]]],
            4322: [477, [[0]]],
            4348: [4, [[0], [5]]],
            4372: [183, [[0], [65, 29]]],
            4430: [4, [[0], [1]]],
            4458: [91, [[0], [124, 37]]],
            4493: [4, [[0], [8]]],
            4502: [477, [[0]]],
            4543: [7, [[0]]],
            4549: [1, [[10, 16]]],
            4551: [1, [[10, 17]]],
            4574: [13, [[0], [116, 97]]],
            4636: [4, [[0], [8]]],
            4644: [477, [[0]]],
            4650: [477, [[0]]],
            4654: [477, [[0]]],
            4669: [7, [[0]]],
            4687: [13, [[0], [112, 96]]],
            4705: [13, [[0], [115, 98]]],
            4731: [7, [[0]]],
            4742: [13, [[0], [115, 98]]],
            4748: [4, [[0], [2]]],
            4771: [4, [[0], [1]]],
            4782: [4, [[0], [3]]],
            4805: [1, [[13, 16]]],
            4807: [1, [[13, 16]]],
            4841: [91, [[0], [115, 16]]],
            4851: [4, [[0], [1]]],
            4873: [91, [[0], [115, 27]]],
            4896: [4, [[0], [8]]],
            4905: [477, [[0]]],
            4916: [477, [[0]]],
            4932: [7, [[0]]],
            4948: [13, [[0], [111, 95]]],
            4955: [1, [[110, 97]]],
            4998: [13, [[0], [122, 102]]],
            5005: [1, [[109, 97]]],
            5007: [1, [[108, 97]]],
            5008: [1, [[107, 97]]],
            5027: [4, [[0], [8]]],
            5033: [1, [[106, 97]]],
            5034: [477, [[0]]],
            5046: [477, [[0]]],
            5049: [477, [[0]]],
            5076: [2, [[2], [46, 55]]],
            5085: [2, [[2], [79, 51]]],
            5088: [451, [[0], [66, 49]]],
            5093: [451, [[0], [63, 47]]],
            5098: [451, [[0], [63, 47]]],
            5102: [451, [[0], [59, 48]]],
            5111: [12, [[0], [62, 72]]],
            5118: [451, [[0], [43, 67]]],
            5122: [451, [[0], [29, 64]]],
            5126: [451, [[0], [33, 67]]],
            5141: [451, [[0], [45, 73]]],
            5145: [451, [[0], [43, 77]]],
            5151: [451, [[0], [41, 80]]],
            5165: [12, [[0], [66, 65]]],
            5188: [12, [[0], [71, 67]]],
            5207: [2, [[0], [45, 75]]],
            5211: [451, [[0], [41, 64]]],
            5216: [451, [[0], [38, 62]]],
            5220: [451, [[0], [36, 62]]],
            5226: [451, [[0], [32, 59]]],
            5236: [12, [[0], [60, 71]]],
            5249: [2, [[0], [49, 59]]],
            5256: [2, [[0], [42, 78]]],
            5272: [2, [[0], [49, 54]]],
            5280: [2, [[0], [49, 59]]],
            5306: [4, [[0], [8]]],
            5314: [477, [[0]]],
            5324: [477, [[0]]],
            5325: [477, [[0]]],
            5326: [477, [[0]]],
            5329: [451, [[0], [67, 55]]],
            5350: [7, [[0]]],
            5368: [13, [[0], [117, 98]]],
            5377: [13, [[0], [117, 98]]],
            5400: [4, [[0], [8]]],
            5406: [477, [[0]]],
            5419: [477, [[0]]]
        }

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1

    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
