# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from .ant import Ant
from .anymal import Anymal
from .anymal_terrain import AnymalTerrain
from .ball_balance import BallBalance
from .cartpole import Cartpole
from .factory.factory_task_gears import FactoryTaskGears
from .factory.factory_task_insertion import FactoryTaskInsertion
from .factory.factory_task_nut_bolt_pick import FactoryTaskNutBoltPick
from .factory.factory_task_nut_bolt_place import FactoryTaskNutBoltPlace
from .factory.factory_task_nut_bolt_screw import FactoryTaskNutBoltScrew
from .franka_cabinet import FrankaCabinet
from .franka_cube_stack import FrankaCubeStack
from .humanoid import Humanoid
from .humanoid_amp import HumanoidAMP
from .ingenuity import Ingenuity
from .quadcopter import Quadcopter
from .shadow_hand import ShadowHand
from .allegro_hand import AllegroHand
from .trifinger import Trifinger
from .a1 import A1
from .anymal_interactive import AnymalInteractive
from .anymal_line import AnymalLine
from .a1_interactive import A1Interactive
from .a1_custom import A1Custom
from .a1_walk_forward_4_legs import A1WalkForward4Legs
from .a1_walk_forward_2_legs import A1WalkForward2Legs
from .a1_walk_forward_4_legs_to_2_legs import A1WalkForward4LegsTo2Legs
from .a1_4legs_with_spring import A14legsWithSpring
from .a1_throwing import A1Throwing
from .a1_walking_and_putting_in import A1Locomotion
from .a1_approaching import A1Approaching
from .a1_throwing_with_camera import A1ThrowingWithCamera
from .a1_approaching_extreme_parkour_version import A1ApproachingExtremeParkourVersion
from .a1_run import A1Run
from .a1_test import A1Test
from .a1_meta_controller_helper import A1MetaControllerHelper
from .a1_helper import A1Helper

# Mappings from strings to environments
isaacgym_task_map = {
    "AllegroHand": AllegroHand,
    "Ant": Ant,
    "Anymal": Anymal,
    "AnymalTerrain": AnymalTerrain,
    "BallBalance": BallBalance,
    "Cartpole": Cartpole,
    "FactoryTaskGears": FactoryTaskGears,
    "FactoryTaskInsertion": FactoryTaskInsertion,
    "FactoryTaskNutBoltPick": FactoryTaskNutBoltPick,
    "FactoryTaskNutBoltPlace": FactoryTaskNutBoltPlace,
    "FactoryTaskNutBoltScrew": FactoryTaskNutBoltScrew,
    "FrankaCabinet": FrankaCabinet,
    "FrankaCubeStack": FrankaCubeStack,
    "Humanoid": Humanoid,
    "HumanoidAMP": HumanoidAMP,
    "Ingenuity": Ingenuity,
    "Quadcopter": Quadcopter,
    "ShadowHand": ShadowHand,
    "Trifinger": Trifinger,
    "A1": A1,
    "AnymalInteractive": AnymalInteractive,
    "AnymalLine" : AnymalLine,
    "A1Interactive": A1Interactive,
    "A1Custom": A1Custom,
    "A1WalkForward4Legs": A1WalkForward4Legs,
    "A1WalkForward2Legs": A1WalkForward2Legs,
    "A1WalkForward4LegsTo2Legs": A1WalkForward4LegsTo2Legs,
    "A14legsWithSpring": A14legsWithSpring,
    "A1Throwing": A1Throwing,
    "A1Locomotion": A1Locomotion,
    "A1Approaching": A1Approaching,
    "A1ThrowingWithCamera": A1ThrowingWithCamera,
    "A1ApproachingExtremeParkourVersion": A1ApproachingExtremeParkourVersion,
    "A1Run": A1Run,
    "A1Test": A1Test,
    "A1MetaControllerHelper": A1MetaControllerHelper,
    "A1Helper": A1Helper,
}
