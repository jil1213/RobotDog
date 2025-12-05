import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics
import numpy as np

SC = exu.SystemContainer()
mbs = SC.AddSystem()

# -------------------------------------------------
# Parameter
# -------------------------------------------------
L_body = 1.0
W_body = 0.3
H_body = 0.2
body_offset = 0.8

L_thigh = 0.3
L_shin  = 0.2
W_leg = 0.1
density = 500
gravity = [0,0,-9.81]

# -------------------------------------------------
# Boden
# -------------------------------------------------
gGround = graphics.CheckerBoard(normal=[0,0,1], size=8, size2=8, nTiles=8, zOffset=-1e-5)
oGround = mbs.AddObject(ObjectGround(referencePosition=[0,0,0],
                                     visualization=VObjectGround(graphicsData=[gGround])))

mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround))

gContact = mbs.AddGeneralContact()
gContact.frictionProportionalZone = 0.01
gContact.SetFrictionPairings(np.diag([1.0, 0.0]))  # Material 0 hat Reibung, 1 nicht

# WICHTIG: Suchbaum-Box manuell setzen (verhindert "size must not be zero")
gContact.SetSearchTreeBox(
    pMin=[-10,-10,-1],
    pMax=[ 10, 10, 1]
)

[meshPoints, meshTrigs] = graphics.ToPointsAndTrigs(gGround)

gContact.AddTrianglesRigidBodyBased(
    rigidBodyMarkerIndex = mGround,
    contactStiffness = 1e5,
    contactDamping = 1e3,
    frictionMaterialIndex = 0,
    pointList = meshPoints,
    triangleList = meshTrigs
)

# -------------------------------------------------
# Inertias
# -------------------------------------------------
bodyInertia = InertiaCuboid(density, [L_body, W_body, H_body]).Translated([0,0,0])
thighInertia = InertiaCuboid(density, [W_leg, W_leg, L_thigh]).Translated([0,0,0])#0.5*L_thigh])
shinInertia  = InertiaCuboid(density, [W_leg, W_leg, L_shin]).Translated([0,0,0])#0.5*L_shin])

# -------------------------------------------------
# Node
# -------------------------------------------------
# 3 Plattform DOF + 4 Hüfte + 4 Knie = 11 DOF
nJoints = 3 + 4 + 4 + 3

referenceCoordinates = [0]*nJoints
deg=math.pi/180

# initialAngles = [
#     0,0,body_offset,0,0,0,
#     -40*deg,  # Hüfte vorne links
#     -40*deg,  # Hüfte vorne rechts
#     -80*deg,  # Hüfte hinten links
#     -80*deg,  # Hüfte hinten rechts
#     70*deg,   # Knie vorne links
#     70*deg,   # Knie vorne rechts
#     120*deg,  # Knie hinten links
#     120*deg   # Knie hinten rechts
# ]
initialAngles = [
    0,0,body_offset,0,0,0,
0,0,0,0,-0.1,-0.1,-0.1,-0.1
]


nKT = mbs.AddNode(NodeGenericODE2(
    referenceCoordinates=referenceCoordinates,
    initialCoordinates=initialAngles,
    initialCoordinates_t=[0]*nJoints,
    numberOfODE2Coordinates=nJoints))

# -------------------------------------------------
# KinematicTree
# -------------------------------------------------
jointTypes = [
    exu.JointType.PrismaticX,
    exu.JointType.PrismaticY,
    exu.JointType.PrismaticZ,
    exu.JointType.RevoluteX,  
    exu.JointType.RevoluteY,
    exu.JointType.RevoluteZ,
]

# 4 Hüftgelenke
jointTypes += [exu.JointType.RevoluteY]*4

# 4 Kniegelenke
jointTypes += [exu.JointType.RevoluteY]*4


linkParents = [-1, 0, 1,2,3,4, 5, 5, 5, 5, 6, 7, 8, 9]

platformIndex = 5
print(len(jointTypes))

# -------------------------------------------------
# Grafik
# -------------------------------------------------
gBody = [
    graphics.Brick([0, 0, 0],
                   [L_body, W_body, H_body],
                   color=graphics.color.red),
    graphics.Basis(length=0.2)
]

gThigh = [
    graphics.Cylinder([0,0,0], [0,0,-L_thigh], W_leg*0.5, color=graphics.color.blue),
    graphics.Basis(length=0.1)
]

gShin = [
    graphics.Cylinder([0,0,0], [0,0,-L_shin], W_leg*0.4, color=graphics.color.green),
    graphics.Basis(length=0.1)
]

# -------------------------------------------------
# JointOffsets, Massen, COMs
# -------------------------------------------------
jointTransformations = exu.Matrix3DList()
jointOffsets = exu.Vector3DList()
linkCOMs = exu.Vector3DList()
linkInertiasCOM = exu.Matrix3DList()
linkMasses = []
gList = []

# Standard Transformation (keine Drehung)
for i in range(len(jointTypes)):
    jointTransformations.Append(np.eye(3))

# Joints 0 und 1
for i in range(5):
    jointOffsets.Append([0,0,0])
    linkInertiasCOM.Append(np.zeros((3,3)))
    linkCOMs.Append([0,0,0])
    linkMasses.append(0)
    gList.append([])



# Plattform
jointOffsets.Append([0,0,0])
linkInertiasCOM.Append(bodyInertia.InertiaCOM())
linkCOMs.Append(bodyInertia.COM())
linkMasses.append(bodyInertia.Mass())
gList.append(gBody)

# Hüften (4 Oberschenkelaufhängungen)
hipX = 0.5*L_body
hipY = 0.5*W_body + 0.5*W_leg
hipZ = 0

hipPositions = [
    [-hipX,  hipY, hipZ],
    [ hipX,  hipY, hipZ],
    [-hipX, -hipY, hipZ],
    [ hipX, -hipY, hipZ]
]

# Oberschenkel
for pos in hipPositions:
    jointOffsets.Append(pos)
    linkInertiasCOM.Append(thighInertia.InertiaCOM())
    linkCOMs.Append(thighInertia.COM())
    linkMasses.append(thighInertia.Mass())
    gList.append(gThigh)

# Kniegelenk sitzt am Ende des Oberschenkels
for i in range(4):
    jointOffsets.Append([0,0,-L_thigh])  # Knie am Beinende
    linkInertiasCOM.Append(shinInertia.InertiaCOM())
    linkCOMs.Append(shinInertia.COM())
    linkMasses.append(shinInertia.Mass())
    gList.append(gShin)





def UF_KinematicTreeForces(mbs, t, itemIndex, q, q_t):
    # q = alle Gelenkkoordinaten
    # q_t = Gelenkgeschwindigkeiten

    # ---- JOINT LIMITS ----
    deg = np.pi/180
    limitK = 2000

    jointLimits = {
        6: (-90*deg,  90*deg),   # Hüften
        7: (-90*deg,  90*deg),
        8: (-90*deg,  90*deg),
        9: (-90*deg,  90*deg),

        10: (-90*deg, -10*deg),   # Knie
        11: (-90*deg, -10*deg),
        12: (-90*deg, -10*deg),
        13: (-90*deg, -10*deg),
    }

    # Ausgang: Nullkräfte
    jointForces = [0]*len(q)

    # Gelenkgrenzen prüfen
    for j, (qmin, qmax) in jointLimits.items():
        if q[j] < qmin:
            jointForces[j] = limitK*(qmin - q[j])
        elif q[j] > qmax:
            jointForces[j] = limitK*(qmax - q[j])

    return jointForces


# -------------------------------------------------
# KinematicTree Objekt
# -------------------------------------------------
oKT = mbs.AddObject(ObjectKinematicTree(
    nodeNumber=nKT,
    jointTypes=jointTypes,
    linkParents=linkParents,
    jointTransformations=jointTransformations,
    jointOffsets=jointOffsets,
    linkInertiasCOM=linkInertiasCOM,
    linkCOMs=linkCOMs,
    linkMasses=linkMasses,
    baseOffset=[0.,0.,0.],
    gravity=gravity,
    forceUserFunction=UF_KinematicTreeForces,
    visualization=VObjectKinematicTree(graphicsDataList=gList)
))





mLegs = []

# KNIE GELENK
for i in range(4):
    mLeg = mbs.AddMarker(MarkerKinematicTreeRigid(objectNumber=oKT,
                                                linkNumber=platformIndex+1+i,
                                                localPosition=[0,0, -L_thigh]))
    mLegs.append(mLeg)
    gContact.AddSphereWithMarker(mLeg,
                                radius=0.5*W_leg,
                                contactStiffness=1e5,
                                contactDamping=1e3,
                                frictionMaterialIndex=0)
    
# FÜSSE (Ende des Unterschenkels)
for i in range(4):
    mLeg = mbs.AddMarker(MarkerKinematicTreeRigid(objectNumber=oKT,
                                                linkNumber=linkParents[-1]+1+i,
                                                localPosition=[0,0, -L_shin]))
    mLegs.append(mLeg)
    gContact.AddSphereWithMarker(mLeg,
                                radius=W_leg,
                                contactStiffness=1e5,
                                contactDamping=1e3,
                                frictionMaterialIndex=0)



# #BODY Points on the Edges of The BODY
sph_rad=0.1
offsets = [
    [ L_body/2 - sph_rad,  W_body/2 - sph_rad, -H_body/2 + sph_rad],   # vorne rechts unten
    [ L_body/2 - sph_rad, -W_body/2 + sph_rad, -H_body/2 + sph_rad],   # vorne links unten
    [-L_body/2 + sph_rad,  W_body/2 - sph_rad, -H_body/2 + sph_rad],   # hinten rechts unten
    [-L_body/2 + sph_rad, -W_body/2 + sph_rad, -H_body/2 + sph_rad],   # hinten links unten

    # [ L_body/2 - sph_rad,  W_body/2 - sph_rad,  H_body/2 - sph_rad],   # vorne rechts oben
    # [ L_body/2 - sph_rad, -W_body/2 + sph_rad,  H_body/2 - sph_rad],   # vorne links oben
    # [-L_body/2 + sph_rad,  W_body/2 - sph_rad,  H_body/2 - sph_rad],   # hinten rechts oben
    # [-L_body/2 + sph_rad, -W_body/2 + sph_rad,  H_body/2 - sph_rad],   # hinten links oben
]

markers = []

for pos in offsets:
    m = mbs.AddMarker(
        MarkerKinematicTreeRigid(
            objectNumber=oKT,
            linkNumber=platformIndex,
            localPosition=pos
        )
    )
    markers.append(m)

    gContact.AddSphereWithMarker(
        markerIndex=m,
        radius=sph_rad,
        contactStiffness=1e5,
        contactDamping=1e3,
        frictionMaterialIndex=0
    )

# -------------------------------------------------
# Render
# -------------------------------------------------
mbs.Assemble()

SC.visualizationSettings.general.drawWorldBasis = True
SC.visualizationSettings.bodies.kinematicTree.showJointFrames = True
SC.visualizationSettings.bodies.kinematicTree.frameSize = 0.15


simSettings = exu.SimulationSettings()

simSettings.timeIntegration.numberOfSteps = 4000   # 4000 Schritte
simSettings.timeIntegration.endTime = 4            # 4 Sekunden
simSettings.timeIntegration.generalizedAlpha.spectralRadius = 0.8
simSettings.timeIntegration.verboseMode = 1        # Ausgabe an

# Solver etwas stabiler machen
simSettings.timeIntegration.newton.useModifiedNewton = True
simSettings.displayStatistics = True

# -------------------------------------------------
# Start simulation
# -------------------------------------------------
#mbs.SolveDynamic(simSettings)

exu.StartRenderer()
mbs.WaitForUserToContinue()
#mbs.SolveDynamic(simSettings)
mbs.SolveDynamic(simSettings)
SC.renderer.DoIdleTasks()
exu.StopRenderer()
