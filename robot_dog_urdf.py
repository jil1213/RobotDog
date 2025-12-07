
import exudyn as exu
from exudyn.utilities import * #includes itemInterface and rigidBodyUtilities
import exudyn.graphics as graphics #only import if it does not conflict
#from exudyn.rigidBodyUtilities import *
from exudyn.robotics import *
from exudyn.robotics.motion import Trajectory, ProfileConstantAcceleration, ProfilePTP
from exudyn.robotics.utilities import GetRoboticsToolboxInternalModel, LoadURDFrobot, GetURDFrobotData

import numpy as np

SC = exu.SystemContainer()
mbs = SC.AddSystem()

# ------------------------------------------------------------
# URDF LADEN
# ------------------------------------------------------------

urdfBasePath = "a1_documentation/urdf/"     # folder
urdfFile     = "a1.urdf"                  # file inside this folder

# URDF laden
robotDict = LoadURDFrobot(urdfFile, urdfBasePath)

robot = robotDict['robot']
urdf  = robotDict['urdf']

robotData = GetURDFrobotData(robot, urdf, returnStaticGraphicsList=False)

linkList = robotData['linkList']
nJ = robotData['numberOfJoints']
print("Dog loaded with", nJ, "joints")
# ------------------------------------------------------------
# ROBOT-Objekt in Exudyn erzeugen
# ------------------------------------------------------------
dog = Robot(
    gravity=[0,0,-9.81],
    base=RobotBase(HT=HTtranslate([0,0,0]),
                   visualization=VRobotBase()),
    tool=RobotTool(HT=HTtranslate([0,0,0]),
                   visualization=VRobotTool()),     
    referenceConfiguration=[]
)

# Map Link-Namen auf Index
linkNumbers = {'None': -1}
for cnt, link in enumerate(linkList):
    linkNumbers[link['name']] = cnt

# ------------------------------------------------------------
# LINKS AUS URDF INS ROBOT-OBJEKT EINFÜGEN
# ------------------------------------------------------------
for cnt, link in enumerate(linkList):
    parentName = link['parentName']
    parentIndex = linkNumbers[parentName]

    mass = link['mass']
    inertia = link['inertiaCOM']

    # Ersatz falls Masse fehlt
    if mass == 0:
        mass = 1.0
        inertia = InertiaSphere(1.0, 0.1).InertiaCOM()

    dog.AddLink(
        RobotLink(
            mass=mass,
            parent=parentIndex,
            COM=link['com'],
            inertia=inertia,
            preHT=link['preHT'],
            jointType=link['jointType'],
            PDcontrol=(2000, 10),
            visualization=VRobotLink(
                graphicsData=link['graphicsDataList'],
                showMBSjoint=True
            )
        )
    )

# ------------------------------------------------------------
# KINEMATIKTREE ERZEUGEN
# ------------------------------------------------------------
dogMBS = dog.CreateKinematicTree(mbs)
oKT = dogMBS['objectKinematicTree']
nKT = dogMBS['nodeGeneric']

# Anfangskonfiguration aus dem URDF
mbs.SetNodeParameter(nKT, 'initialCoordinates', robotData['staticJointValues'])

# ------------------------------------------------------------
# ASSEMBLE + SIMULATION
# ------------------------------------------------------------
mbs.Assemble()

SC.visualizationSettings.bodies.kinematicTree.showJoints = True
SC.visualizationSettings.bodies.kinematicTree.frameSize = 0.1

exu.StartRenderer()
mbs.SolveStatic()      # Hund einfach hängen lassen / Gravitation ausbalancieren
SC.renderer.DoIdleTasks()
exu.StopRenderer()

