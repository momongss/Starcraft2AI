all_actions = [
    1,      # move_camera
    2,      # select_point
    3,      # select_rect
    4,      # select_control_group
    5,      # select_unit
    6,      # select_idle_worker
    7,      # select_army
    10,     # unload            ?
    11,     # build_queue
    12,     # Attack_screen
    13,     # Attack_minimap
    14,     # Attack_AttackBuilding_screen
    15,     # Attack_AttackBuilding_minimap
    16,     # Attack_Battlecruiser_screen
    17,     # Attack_Battlecruiser_minimap
    18,     # Attack_Redirect_screen
    21,     # Behavior_BuildingAttackOff_quick
    22,     # Behavior_BuildingAttackOn_quick
    31,     # Behavior_HoldFireOff_quick
    34,     # Behavior_HoldFireOn_quick
    140,    # Cancel_quick
    168,    # Cancel_Last_quick
    264,    # Harvest_Gather_screen
    269,    # Harvest_Return_quick
    274,    # HoldPosition_quick
    559,    # HoldPosition_Hold_quick

    287,    # Load_screen
    294,    # LoadAll_quick

    331,    # Move_screen
    332,    # Move_minimap
    563,    # Move_Move_screen
    564,    # Move_Move_minimap

    333,    # Patrol_screen
    334,    # Patrol_minimap
    567,    # Patrol_Patrol_screen
    568,    # Patrol_Patrol_minimap

    335,    # Rally_Units_screen
    336,    # Rally_Units_minimap
    337,    # Rally_Building_screen
    338,    # Rally_Building_minimap
    341,    # Rally_Morphing_Unit_screen
    342,    # Rally_Morphing_Unit_minimap
    343,    # Rally_Workers_screen
    344,    # Rally_Workers_minimap

    451,    # Smart_screen
    452,    # Smart_minimap
    453,    # Stop_quick
    454,    # Stop_Building_quick
    455,    # Stop_Redirect_quick
    456,    # Stop_Stop_quick

    511,    # UnloadAll_quick
    516,    # UnloadAllAt_screen
    517,    # UnloadAllAt_minimap
]

terran_actions = [
    23,     # Behavior_CloakOff_quick           ?
    24,     # Behavior_CloakOff_Banshee_quick   ?
    25,     # Behavior_CloakOff_Ghost_quick     ?
    26,     # Behavior_CloakOn_quick            ?
    27,     # Behavior_CloakOn_Banshee_quick    ?
    28,     # Behavior_CloakOn_Ghost_quick      ?

    32,     # Behavior_HoldFireOff_Ghost_quick  ?
    35,     # Behavior_HoldFireOn_Ghost_quick   ?

    39,     # Build_Armory_screen
    42,     # Build_Barracks_screen
    43,     # Build_Bunker_screen
    44,     # Build_CommandCenter_screen
    50,     # Build_EngineeringBay_screen
    53,     # Build_Factory_screen
    56,     # Build_FusionCore_screen
    58,     # GhostAcademy
    64,     # Build_MissileTurret_screen
    66,     # Build_Nuke_quick

    71,     # Build_Reactor_quick               ??
    72,     # Build_Reactor_screen
    73,     # Build_Reactor_Barracks_quick
    74,     # Build_Reactor_Barracks_screen
    75,     # Build_Reactor_Factory_quick
    76,     # Build_Reactor_Factory_screen

    77,     # Build_Reactor_Starport_quick
    78,     # Build_Reactor_Starport_screen

    79,     # Build_Refinery_screen
    83,     # Build_SensorTower_screen
    89,     # Build_Starport_screen
    91,     # Build_SupplyDepot_screen
    92,     # Build_TechLab_quick
    93,     # Build_TechLab_screen
    94,     # Build_TechLab_Barracks_quick
    95,     # Build_TechLab_Barracks_screen
    96,     # Build_TechLab_Factory_quick
    97,     # Build_TechLab_Factory_screen
    98,     # Build_TechLab_Starport_quick
    99,     # Build_TechLab_Starport_screen

    143,    # Cancel_BarracksAddOn_quick
    146,    # Cancel_FactoryAddOn_quick

    163,    # Cancel_Nuke_quick
    166,    # Cancel_StarportAddOn_quick

    178,    # Effect_AutoTurret_screen
    183,    # Effect_CalldownMULE_screen
    190,    # Effect_EMP_screen
    185,    # Effect_GhostSnipe_screen          ??
    198,    # Effect_Heal_screen
    199,    # Effect_Heal_autocast
    528,    # Effect_InterferenceMatrix_screen  ??
    205,    # Effect_KD8Charge_screen
    213,    # Effect_NukeCalldown_screen        ??
    217,    # Effect_PointDefenseDrone_screen

    220,    # Effect_Repair_screen
    221,    # Effect_Repair_autocast
    222,    # Effect_Repair_Mule_screen
    223,    # Effect_Repair_Mule_autocast
    530,    # Effect_Repair_RepairDrone_screen
    531,    # Effect_Repair_RepairDrone_autocast
    224,    # Effect_Repair_SCV_screen
    225,    # Effect_Repair_SCV_autocast
    532,    # Effect_RepairDrone_screen
    226,    # Effect_Salvage_quick
    227,    # Effect_Scan_screen
    542,    # Effect_Scan_minimap
    234,    # Effect_Stim_quick
    235,    # Effect_Stim_Marauder_quick    ?
    236,    # Effect_Stim_Marauder_Redirect_quick
    237,    # Effect_Stim_Marine_quick
    238,    # Effect_Stim_Marine_Redirect_quick
    239,    # Effect_SupplyDrop_screen      ?
    240,    # Effect_TacticalJump_screen
    553,    # Effect_TacticalJump_minimap
    247,    # Effect_YamatoGun_screen

    266,    # Harvest_Gather_Mule_screen
    268,    # Harvest_Gather_SCV_screen

    271,    # Harvest_Return_Mule_quick
    273,    # Harvest_Return_SCV_quick

    558,    # HoldPosition_Battlecruiser_quick

    275,    # Land_screen
    276,    # Land_Barracks_screen
    277,    # Land_CommandCenter_screen
    278,    # Land_Factory_screen
    279,    # Land_OrbitalCommand_screen
    280,    # Land_Starport_screen
    281,    # Lift_quick
    282,    # Lift_Barracks_quick
    283,    # Lift_CommandCenter_quick
    284,    # Lift_Factory_quick
    285,    # Lift_OrbitalCommand_quick
    286,    # Lift_Starport_quick

    288,    # Load_Bunker_screen
    289,    # Load_Medivac_screen
    295,    # LoadAll_CommandCenter_quick
    300,    # Morph_Hellbat_quick
    301,    # Morph_Hellion_quick

    304,    # Morph_LiberatorAAMode_quick
    305,    # Morph_LiberatorAGMode_screen
    554,    # Morph_LiberatorAGMode_minimap
    312,    # Morph_PlanetaryFortress_quick
    317,    # Morph_SiegeMode_quick
    318,    # Morph_SupplyDepot_Lower_quick
    319,    # Morph_SupplyDepot_Raise_quick

    320,    # Morph_ThorExplosiveMode_quick
    321,    # Morph_ThorHighImpactMode_quick
    322,    # Morph_Unsiege_quick

    326,    # Morph_VikingAssaultMode_quick
    327,    # Morph_VikingFighterMode_quick
    561,    # Move_Battlecruiser_screen
    562,    # Move_Battlecruiser_minimap
    565,    # Patrol_Battlecruiser_screen
    566,    # Patrol_Battlecruiser_minimap
    345,    # Rally_CommandCenter_screen
    346,    # Rally_CommandCenter_minimap

    352,    # Research_AdvancedBallistics_quick
    353,    # Research_BansheeCloakingField_quick
    354,    # Research_BansheeHyperflightRotors_quick
    355,    # Research_BattlecruiserWeaponRefit_quick
    361,    # Research_CombatShield_quick
    362,    # Research_ConcussiveShells_quick
    570,    # Research_CycloneLockOnDamage_quick
    540,    # Research_CycloneRapidFireLaunchers_quick
    363,    # Research_DrillingClaws_quick
    572,    # Research_EnhancedShockwaves_quick
    369,    # Research_HiSecAutoTracking_quick
    370,    # Research_HighCapacityFuelTanks_quick
    371,    # Research_InfernalPreigniter_quick
    375,    # Research_NeosteelFrame_quick
    377,    # Research_PathogenGlands_quick
    378,    # Research_PersonalCloaking_quick
    402,    # Research_RavenCorvidReactor_quick             ?
    403,    # Research_RavenRecalibratedExplosives_quick    ?
    373,    # Research_SmartServos_quick
    405,    # Research_Stimpack_quick
    406,    # Research_TerranInfantryArmor_quick
    407,
    408,
    409,
    410,
    411,
    412,
    413,
    414,
    415,
    416,
    417,
    418,
    419,
    420,
    421,
    422,
    423,
    424,
    425,
    426,

    571,    # Stop_Battlecruiser_quick
    459,    # Train_Banshee_quick
    460,    # Train_Battlecruiser_quick
    464,    # Train_Cyclone_quick
    468,    # Train_Ghost_quick
    469,    # Train_Hellbat_quick
    470,    # Train_Hellion_quick
    475,    # Train_Liberator_quick
    476,    # Train_Marauder_quick
    477,    # Train_Marine_quick
    478,    # Train_Medivac_quick
    487,    # Train_Raven_quick
    488,    # Train_Reaper_quick
    490,    # Train_SCV_quick
    492,    # Train_SiegeTank_quick
    496,    # Train_Thor_quick
    498,    # Train_VikingFighter_quick
    502,    # Train_WidowMine_quick

    512,    # UnloadAll_Bunker_quick
    513,    # UnloadAll_CommandCenter_quick
    518,    # UnloadAllAt_Medivac_screen
    519,    # UnloadAllAt_Medivac_minimap

]

#     Function.ability(406, "Research_TerranInfantryArmor_quick", cmd_quick, 3697),
#     Function.ability(407, "Research_TerranInfantryArmorLevel1_quick", cmd_quick, 656, 3697),
#     Function.ability(408, "Research_TerranInfantryArmorLevel2_quick", cmd_quick, 657, 3697),
#     Function.ability(409, "Research_TerranInfantryArmorLevel3_quick", cmd_quick, 658, 3697),
#     Function.ability(410, "Research_TerranInfantryWeapons_quick", cmd_quick, 3698),
#     Function.ability(411, "Research_TerranInfantryWeaponsLevel1_quick", cmd_quick, 652, 3698),
#     Function.ability(412, "Research_TerranInfantryWeaponsLevel2_quick", cmd_quick, 653, 3698),
#     Function.ability(413, "Research_TerranInfantryWeaponsLevel3_quick", cmd_quick, 654, 3698),
#     Function.ability(414, "Research_TerranShipWeapons_quick", cmd_quick, 3699),
#     Function.ability(415, "Research_TerranShipWeaponsLevel1_quick", cmd_quick, 861, 3699),
#     Function.ability(416, "Research_TerranShipWeaponsLevel2_quick", cmd_quick, 862, 3699),
#     Function.ability(417, "Research_TerranShipWeaponsLevel3_quick", cmd_quick, 863, 3699),
#     Function.ability(418, "Research_TerranStructureArmorUpgrade_quick", cmd_quick, 651),
#     Function.ability(419, "Research_TerranVehicleAndShipPlating_quick", cmd_quick, 3700),
#     Function.ability(420, "Research_TerranVehicleAndShipPlatingLevel1_quick", cmd_quick, 864, 3700),
#     Function.ability(421, "Research_TerranVehicleAndShipPlatingLevel2_quick", cmd_quick, 865, 3700),
#     Function.ability(422, "Research_TerranVehicleAndShipPlatingLevel3_quick", cmd_quick, 866, 3700),
#     Function.ability(423, "Research_TerranVehicleWeapons_quick", cmd_quick, 3701),
#     Function.ability(424, "Research_TerranVehicleWeaponsLevel1_quick", cmd_quick, 855, 3701),
#     Function.ability(425, "Research_TerranVehicleWeaponsLevel2_quick", cmd_quick, 856, 3701),
#     Function.ability(426, "Research_TerranVehicleWeaponsLevel3_quick", cmd_quick, 857, 3701),

zerg_actions = [
    29,     # Behavior_GenerateCreepOff_quick
    30,     # Behavior_GenerateCreepOn_quick
    33,     # Behavior_HoldFireOff_Lurker_quick
    36,     # Behavior_HoldFireOn_Lurker_quick
    41,     # Build_BanelingNest_screen
    45,     # Build_CreepTumor_screen
    46,     # Build_CreepTumor_Queen_screen
    47,     # Build_CreepTumor_Tumor_screen
    51,     # Build_EvolutionChamber_screen
    52,     # Build_Extractor_screen
    59,     # Build_Hatchery_screen
    60,     # Build_HydraliskDen_screen
    61,     # Build_InfestationPit_screen
    524,    # Build_LurkerDen_screen
    67,     # Build_NydusNetwork_screen
    68,     # Build_NydusWorm_screen
    80,     # Build_RoachWarren_screen
    84,     # Build_SpawningPool_screen
    85,     # Build_SpineCrawler_screen
    86,     # Build_Spire_screen
    87,     # Build_SporeCrawler_screen
    102,    # Build_UltraliskCavern_screen

    103,    # BurrowDown_quick
    104,    # BurrowDown_Baneling_quick
    105,    # BurrowDown_Drone_quick
    106,    # BurrowDown_Hydralisk_quick
    107,    # BurrowDown_Infestor_quick
    108,    # BurrowDown_InfestorTerran_quick
    109,    # BurrowDown_Lurker_quick
    110,    # BurrowDown_Queen_quick
    111,    # BurrowDown_Ravager_quick
    112,    # BurrowDown_Roach_quick
    113,    # BurrowDown_SwarmHost_quick
    114,    # BurrowDown_Ultralisk_quick
    115,    # BurrowDown_WidowMine_quick
    116,    # BurrowDown_Zergling_quick
    117,    # BurrowUp_quick
    118,    # BurrowUp_autocast
    119,    # BurrowUp_Baneling_quick
    120,    # BurrowUp_Baneling_autocast
    121,    # BurrowUp_Drone_quick
    122,    # BurrowUp_Hydralisk_quick
    123,    # BurrowUp_Hydralisk_autocast
    # ~~~ 139

    145,    # Cancel_CreepTumor_quick
    149,     # Cancel_MorphBroodlord_quick
    150,    # Cancel_MorphGreaterSpire_quick
    151,    # Cancel_MorphHive_quick
    152,    # Cancel_MorphLair_quick
    153,    # Cancel_MorphLurker_quick
    # ~~~ 161
    162,    # Cancel_NeuralParasite_quick
    164,    # Cancel_SpineCrawlerRoot_quick
    165,    # Cancel_SporeCrawlerRoot_quick

    176,    # Effect_Abduct_screen
    179,    # Effect_BlindingCloud_screen
    184,    # Effect_CausticSpray_screen
    188,    # Effect_Contaminate_screen
    189,    # Effect_CorrosiveBile_screen
    191,    # Effect_Explode_quick
    194,    # Effect_FungalGrowth_screen
    200,    # Effect_HunterSeekerMissile_screen
    203,    # Effect_InfestedTerrans_screen
    207,    # Effect_LocustSwoop_screen
    212,    # Effect_NeuralParasite_screen
    215,    # Effect_ParasiticBomb_screen
    228,    # Effect_SpawnChangeling_quick
    229,    # Effect_SpawnLocusts_screen
    230,    # Effect_Spray_screen
    231,    # Effect_Spray_Protoss_screen
    232,    # Effect_Spray_Terran_screen
    233,    # Effect_Spray_Zerg_screen
    549,    # Effect_Spray_minimap
    550,    # Effect_Spray_Protoss_minimap
    551,    # Effect_Spray_Terran_minimap
    552,    # Effect_Spray_Zerg_minimap
    242,    # Effect_Transfusion_screen
    243,    # Effect_ViperConsume_screen

    265,    # Harvest_Gather_Drone_screen
    270,    # Harvest_Return_Drone_quick
    290,    # Load_NydusNetwork_screen
    291,    # Load_NydusWorm_screen
    292,    # Load_Overlord_screen
    297,    # Morph_BroodLord_quick
    299,    # Morph_GreaterSpire_quick
    302,    # Morph_Hive_quick
    303,    # Morph_Lair_quick
    306,    # Morph_Lurker_quick
    307,    # Morph_LurkerDen_quick
    310,    # Morph_OverlordTransport_quick
    311,    # Morph_Overseer_quick
    536,    # Morph_OverseerMode_quick
    313,    # Morph_Ravager_quick
    315,    # Morph_SpineCrawlerRoot_screen
    316,    # Morph_SporeCrawlerRoot_screen
    323,    # Morph_Uproot_quick
    324,    # Morph_SpineCrawlerUproot_quick
    325,    # Morph_SporeCrawlerUproot_quick

    339,    # Rally_Hatchery_Units_screen
    340,    # Rally_Hatchery_Units_minimap
    347,    # Rally_Hatchery_Workers_screen
    348,    # Rally_Hatchery_Workers_minimap
    539,    # Research_AdaptiveTalons_quick
    569,    # Research_AnabolicSynthesis_quick
    357,    # Research_Burrow_quick
    358,    # Research_CentrifugalHooks_quick
    360,    # Research_ChitinousPlating_quick
    368,    # Research_GroovedSpines_quick
    374,    # Research_MuscularAugments_quick
    376,    # Research_NeuralParasite_quick
    377,    # Research_PathogenGlands_quick
    380,    # Research_PneumatizedCarapace_quick
    458,    # Train_Baneling_quick
    463,    # Train_Corruptor_quick
    467,    # Train_Drone_quick
    472,    # Train_Hydralisk_quick
    474,    # Train_Infestor_quick
    480,    # Train_Mutalisk_quick
    483,    # Train_Overlord_quick
    486,    # Train_Queen_quick
    489,    # Train_Roach_quick
    494,    # Train_SwarmHost_quick
    497,    # Train_Ultralisk_quick
    499,    # Train_Viper_quick
    504,    # Train_Zergling_quick

    514,    # UnloadAll_NydusNetwork_quick
    515,    # UnloadAll_NydusWorm_quick
    519,    # UnloadAllAt_Overlord_screen
    521,    # UnloadAllAt_Overlord_minimap

]

protoss_actions = [
    8,      # select_warp_gates
    9,      # select_larva
    37,     # Behavior_PulsarBeamOff_quick
    38,     # Behavior_PulsarBeamOn_quick
    40,     # Build_Assimilator_screen
    48,     # Build_CyberneticsCore_screen
    49,     # Build_DarkShrine_screen
    54,     # Build_FleetBeacon_screen
    55,     # Build_Forge_screen
    57,     # Build_Gateway_screen
    62,     # Build_Interceptors_quick
    63,     # Build_Interceptors_autocast
    65,     # Build_Nexus_screen
    69,     # Build_PhotonCannon_screen
    70,     # Build_Pylon_screen
    81,     # Build_RoboticsBay_screen
    82,     # Build_RoboticsFacility_screen
    525,    # Build_ShieldBattery_screen
    88,     # Build_Stargate_screen
    100,    # Build_TemplarArchive_screen
    101,    # Build_TwilightCouncil_screen

    147,    # Cancel_GravitonBeam_quick
    167,    # Cancel_StasisTrap_quick
    546,    # Cancel_VoidRayPrismaticAlignment_quick

    177,    # Effect_AdeptPhaseShift_screen
    547,    # Effect_AdeptPhaseShift_minimap
    180,    # Effect_Blink_screen
    543,    # Effect_Blink_minimap
    181,    # Effect_Blink_Stalker_screen
    544,    # Effect_Blink_Stalker_minimap
    182,    # Effect_ShadowStride_screen
    545,    # Effect_ShadowStride_minimap
    185,    # Effect_Charge_screen
    186,    # Effect_Charge_autocast
    187,    # Effect_ChronoBoost_screen
    527,    # Effect_ChronoBoostEnergyCost_screen
    192,    # Effect_Feedback_screen
    193,    # Effect_ForceField_screen
    196,    # Effect_GravitonBeam_screen
    187,    # Effect_GuardianShield_quick
    208,    # Effect_MassRecall_screen
    209,    # Effect_MassRecall_Mothership_screen
    210,    # Effect_MassRecall_MothershipCore_screen
    529,    # Effect_MassRecall_Nexus_screen
    211,    # Effect_MedivacIgniteAfterburners_quick
    214,    # Effect_OracleRevelation_screen
    216,    # Effect_PhotonOvercharge_screen
    219,    # Effect_PurificationNova_screen
    533,    # Effect_Restore_screen
    534,    # Effect_Restore_autocast
    241,    # Effect_TimeWarp_screen
    244,    # Effect_VoidRayPrismaticAlignment_quick
    248,    # Hallucination_Adept_quick
    249,    # Hallucination_Archon_quick
    # ~~~260
    267,    # Harvest_Gather_Probe_screen
    272,    # Harvest_Return_Probe_quick

    293,    # Load_WarpPrism_screen
    296,    # Morph_Archon_quick
    298,    # Morph_Gateway_quick
    308,    # Morph_Mothership_quick
    535,    # Morph_ObserverMode_quick
    309,    # Morph_OrbitalCommand_quick
    538,    # Morph_SurveillanceMode_quick
    328,    # Morph_WarpGate_quick
    560,    # Morph_WarpGate_autocast
    329,    # Morph_WarpPrismPhasingMode_quick
    330,    # Morph_WarpPrismTransportMode_quick
    349,    # Rally_Nexus_screen
    350,    # Rally_Nexus_minimap

    351,    # Research_AdeptResonatingGlaives_quick
    356,    # Research_Blink_quick
    359,    # Research_Charge_quick

    364,    # Research_ExtendedThermalLance_quick
    366,    # Research_GraviticBooster_quick
    367,    # Research_GraviticDrive_quick
    372,    # Research_InterceptorGravitonCatapult_quick
    379,    # Research_PhoenixAnionPulseCrystals_quick

    381,    # Research_ProtossAirArmor_quick
    # ~~~ 400

    401,    # Research_PsiStorm_quick
    457,    # Train_Adept_quick
    461,    # Train_Carrier_quick
    462,    # Train_Colossus_quick
    465,    # Train_DarkTemplar_quick
    466,    # Train_Disruptor_quick
    471,    # Train_HighTemplar_quick
    473,    # Train_Immortal_quick
    541,    # Train_Mothership_quick
    479,    # Train_MothershipCore_quick
    481,    # Train_Observer_quick
    482,    # Train_Oracle_quick
    484,    # Train_Phoenix_quick
    485,    # Train_Probe_quick
    491,    # Train_Sentry_quick
    493,    # Train_Stalker_quick
    495,    # Train_Tempest_quick
    500,    # Train_VoidRay_quick
    501,    # Train_WarpPrism_quick
    503,    # Train_Zealot_quick
    505,    # TrainWarp_Adept_screen
    506,    # TrainWarp_DarkTemplar_screen
    507,    # TrainWarp_HighTemplar_screen
    508,    # TrainWarp_Sentry_screen
    509,    # TrainWarp_Stalker_screen
    510,    # TrainWarp_Zealot_screen
    522,    # UnloadAllAt_WarpPrism_screen
    523,    # UnloadAllAt_WarpPrism_minimap
]

# 19    Scan_Move_screen
# 20    Scan_Move_minimap
# 90    Build_StasisTrap_screen         프토
# 141   Cancel_AdeptPhaseShift_quick    프토
# 142   Cancel_AdeptShadePhaseShift_quick   프토
# 144   Cancel_BuildInProgress_quick

# 148   Cancel_LockOn_quick     테란
# 169   Cancel_HangarQueue5_quick
##################################### 테란
# 170   Cancel_Queue1_quick
# 171   Cancel_Queue5_quick
# 172   Cancel_QueueAddOn_quick
# 173   Cancel_QueueCancelToSelection_quick
# 174   Cancel_QueuePassive_quick
# 175   Cancel_QueuePassiveCancelToSelection_quick
######################################

# 526   Effect_AntiArmorMissile_screen
# 201   Effect_ImmortalBarrier_quick
# 202   Effect_ImmortalBarrier_autocast
# 204   Effect_InjectLarva_screen 저그

# 206   Effect_LockOn_screen 테란
# 557   Effect_LockOn_autocast 테란
# 218   Effect_PsiStorm_screen

# 245   Effect_WidowMineAttack_screen 테란 #
# 246   Effect_WidowMineAttack_autocast

# 261   Halt_quick
# 262   Halt_Building_quick
# 263   Halt_TerranBuild_quick
# 537   Morph_OversightMode_quick
# 314   Morph_Root_screen
# 365   Research_GlialRegeneration_quick
# 404   Research_ShadowStrike_quick


print(len(all_actions))
print(len(terran_actions))
print(len(zerg_actions))
print(len(protoss_actions))