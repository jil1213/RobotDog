import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics
import numpy as np
from exudyn.robotics.motion import Trajectory, ProfileConstantAcceleration, ProfilePTP

def GetModel():
    #add class which can be returned to enable user to access parameters

    referenceCoordinates = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0]
    L_body = 1.0   #Länge Body
    W_body = 0.3
    H_body = 0.2
    platformMass = 5
    body_offset = 0.8  # Fallhöhe des Bodys über dem Boden
    L_thigh = 0.3
    L_shin  = 0.2
    W_leg = 0.1
    density = 500
    legMass = 0.05
    gContact = None 



    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    gravity = [0,0,-9.81]
    # -------------------------------------------------
    # Boden
    # -------------------------------------------------

    k = 10e3
    d = 0.01*k
    frictionCoeff = 1.5#0.8 #0.5
    ss = 1

    planeL = 16

    #zContact = -0.7+0.01214+6.45586829e-02 #contact at beginning, leg = [0,0.2*pi,-0.3*pi]
    p0 = [0,0,0]
    mbs.variables['zContact'] = W_leg/2 #zContact

    gGroundSimple = graphics.CheckerBoard(p0,size=planeL, nTiles=1) #only 1 tile for efficiency

    gGround = graphics.CheckerBoard(p0,size=planeL, nTiles=16)
    oGround = mbs.AddObject(ObjectGround(referencePosition=[0,0,0],
                                        visualization=VObjectGround(graphicsData=[gGround])))
    mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround))


    gContact = mbs.AddGeneralContact()
    gContact.frictionProportionalZone = 0.3#0.01

    gContact.SetFrictionPairings(frictionCoeff*np.eye(1))
    gContact.SetSearchTreeCellSize(numberOfCells=[ss,ss,1])
    # WICHTIG: Suchbaum-Box manuell setzen (verhindert "size must not be zero")
    stFact = 0.3
    gContact.SetSearchTreeBox(pMin=np.array([-0.5*planeL*stFact,-0.5*planeL*stFact,-1]),
                               pMax=np.array([0.5*planeL*stFact,0.5*planeL*stFact,0.1]))

    [meshPoints, meshTrigs] = graphics.ToPointsAndTrigs(gGroundSimple)

    gContact.AddTrianglesRigidBodyBased(
        rigidBodyMarkerIndex = mGround,
        contactStiffness = k,
        contactDamping = d,
        frictionMaterialIndex = 0,
        pointList = meshPoints,
        triangleList = meshTrigs,
        staticTriangles=False,
    )
    ### End eKontakt mit dem Boden ------

    # -------------------------------------------------
    # Inertias
    # -------------------------------------------------
    # if platformInertia is None:
    #     platformInertia = InertiaCuboid(density, [L_body, W_body, H_body]).Translated([0,0,0])
    # else: 
    #     platformInertia = RigidBodyInertia(mass=platformMass, inertiaTensorAtCOM=np.diag(platformInertia))
    # if legInertia is None:
    #     thighInertia = InertiaCylinder(density, length=L_thigh, outerRadius=0.5 *W_leg, axis=2).Translated([0,0,0])
    #     shinInertia = InertiaCylinder(density, length=L_shin, outerRadius=0.5 *W_leg, axis=2).Translated([0,0,0])
    # else: 
    #     thighInertia = RigidBodyInertia(mass=legMass, inertiaTensorAtCOM=np.diag(legInertia))
    #     shinInertia  = RigidBodyInertia(mass=legMass, inertiaTensorAtCOM=np.diag(legInertia))


    platformInertia = InertiaCuboid(density, [L_body, W_body, H_body]).Translated([0,0,0])
    thighInertia = InertiaCylinder(density, length=L_thigh, outerRadius=0.5 *W_leg, axis=2).Translated([0,0,0])
    shinInertia = InertiaCylinder(density, length=L_shin, outerRadius=0.5 *W_leg, axis=2).Translated([0,0,0])

    # -------------------------------------------------
    # Node
    # -------------------------------------------------
    # 3 Plattform XYZ-Translation + 3 Plattform Drehung + 4 Hüfte + 4 Knie = 14 DOF
    nJoints = 3 + 3 + 4 + 4 + 4
    # referenceCoordinates = [0]*nJoints
    deg=math.pi/180



    q0 = [0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nKT = mbs.AddNode(NodeGenericODE2(
        referenceCoordinates=referenceCoordinates,
        initialCoordinates=q0,
        initialCoordinates_t=[0]*nJoints,
        numberOfODE2Coordinates=nJoints))
    # -------------------------------------------------
    # KinematicTree
    # -------------------------------------------------
    # 6 Gelenke der Plattform
    jointTypes = [
        exu.JointType.PrismaticX,
        exu.JointType.PrismaticY,
        exu.JointType.PrismaticZ,
        exu.JointType.RevoluteX,  
        exu.JointType.RevoluteY,
        exu.JointType.RevoluteZ,
    ]

    # 4 Hüftgelenke X-Achse
    jointTypes += [exu.JointType.RevoluteX]*4
    
    # 4 Hüftgelenke Y-Achse
    jointTypes += [exu.JointType.RevoluteY]*4

    # 4 Kniegelenke
    jointTypes += [exu.JointType.RevoluteY]*4

    #rd.linkParents = [-1, 0, 1, 2, 3, 4, 5, 5, 5, 5,5,5,5,5, 6, 7, 8, 9] #    

    linkParents = [
        -1, 0, 1, 2, 3, 4,   # 0–5 Floating base

        5,   # 6  HipX FR
        5,   # 7  HipX BR
        5,   # 8  HipX FL
        5,   # 9  HipX BL
        6,   # 10 HipY FR
        7,   # 11 HipY BR
        8,   # 12 HipY FL
        9,   # 13 HipY BL

        10,   # 14 Knee FL
        11,   # 15 Knee FR
        12,  # 16 Knee BL
        13   # 17 Knee BR
    ]


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
    
    gZero =[]

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
    # Ersten 6 Gelenke sind Masselos 
    for i in range(5):
        jointOffsets.Append([0,0,0])
        linkInertiasCOM.Append(np.zeros((3,3)))
        linkCOMs.Append([0,0,0])
        linkMasses.append(0)
        gList.append([])

    # Plattform
    jointOffsets.Append([0,0,0])
    linkInertiasCOM.Append(platformInertia.InertiaCOM())
    linkCOMs.Append(platformInertia.COM())
    linkMasses.append(platformInertia.Mass())
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

    # Oberschenkel X-Achse
    for pos in hipPositions:
        jointOffsets.Append(pos)
        linkInertiasCOM.Append(np.zeros((3,3)))
        linkCOMs.Append([0,0,0])
        linkMasses.append(0)
        gList.append(gZero)

    # Oberschenkel Y-Achse

    hipPositions = np.zeros((4,3))
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

    # -------------------------------------------------
    # KinematicTree Objekt
    # -------------------------------------------------
    jointPControlVector = [0]*6 + [200]*12
    jointDControlVector = [0]*6 + [20]*12

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
        jointPControlVector=jointPControlVector,
        jointDControlVector=jointDControlVector,
        jointPositionOffsetVector=[0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        jointVelocityOffsetVector=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #forceUserFunction=UF_KinematicTreeForces,
        visualization=VObjectKinematicTree(graphicsDataList=gList)
    ))

    mLegs = []

    # FÜSSE (Ende des Unterschenkels)
    for i in range(4):
        mLeg = mbs.AddMarker(MarkerKinematicTreeRigid(objectNumber=oKT,
                                                    linkNumber=linkParents[-1]+1+i,
                                                    localPosition=[0,0, -L_shin]))
        mLegs.append(mLeg)
        gContact.AddSphereWithMarker(mLeg,
                                    radius=W_leg/2,
                                    contactStiffness=1e5,
                                    contactDamping=1e3,
                                    frictionMaterialIndex=0)


    # -------------------------------------------------
    # Add Sensors
    # -------------------------------------------------
    sPlatformPos = mbs.AddSensor(SensorKinematicTree(objectNumber=oKT, linkNumber = platformIndex,
                                                            storeInternal=True, outputVariableType=exu.OutputVariableType.Position))
    sPlatformVel = mbs.AddSensor(SensorKinematicTree(objectNumber=oKT, linkNumber = platformIndex,
                                                            storeInternal=True, outputVariableType=exu.OutputVariableType.Velocity))
    sPlatformAng = mbs.AddSensor(SensorKinematicTree(objectNumber=oKT, linkNumber = platformIndex,
                                                            storeInternal=True, outputVariableType=exu.OutputVariableType.Rotation))
    sPlatformAngVel = mbs.AddSensor(SensorKinematicTree(objectNumber=oKT, linkNumber = platformIndex,
                                                            storeInternal=True, outputVariableType=exu.OutputVariableType.AngularVelocity))

    for i in range(8):
        mbs.AddSensor(SensorKinematicTree(objectNumber=oKT, linkNumber = platformIndex+1+i,
                                                                storeInternal=True, outputVariableType=exu.OutputVariableType.Rotation))
        mbs.AddSensor(SensorKinematicTree(objectNumber=oKT, linkNumber = platformIndex+1+i,
                                                                storeInternal=True, outputVariableType=exu.OutputVariableType.AngularVelocity))


    # -------------------------------------------------
    # configurations and trajectory
    # -------------------------------------------------
    
    # 0-Lage Alle Beine Ausgestreckt
    q0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    q1 = [0,0,0,0,0,0, # [x, y, z, Rx, Ry, Rz,]
          np.pi/2,np.pi/2,-np.pi/2,-np.pi/2, # [HipX_FR, HipX_BR, HipX_FL, HipX_BL]
          0,0,0,0, # [HipY_FR, HipY_BR, HipY_FL, HipY_BL]
          0,0,0,0] # [Knee_FL, Knee_FR, Knee_BL, Knee_BR]
    
    q2 = [0,0,0,0,0,0, # [x, y, z, Rx, Ry, Rz,]
          0,0,0,0, # [HipX_FR, HipX_BR, HipX_FL, HipX_BL]
          np.pi/2,np.pi/2,np.pi/2,np.pi/2, # [HipY_FR, HipY_BR, HipY_FL, HipY_BL]
          0,0,0,0] # [Knee_FL, Knee_FR, Knee_BL, Knee_BR]
    q3 = [0,0,0,0,0,0, # [x, y, z, Rx, Ry, Rz,]
          0,0,0,0, # [HipX_FR, HipX_BR, HipX_FL, HipX_BL]
          0,0,0,0, # [HipY_FR, HipY_BR, HipY_FL, HipY_BL]
          np.pi/4,np.pi/4,np.pi/4,np.pi/4] # [Knee_FL, Knee_FR, Knee_BL, Knee_BR]

    
    #trajectory generated with optimal acceleration profiles:
    trajectory = Trajectory(initialCoordinates=q0, initialTime=0)
    trajectory.Add(ProfileConstantAcceleration(q1,0.2))
    trajectory.Add(ProfileConstantAcceleration(q0,0.5))
    trajectory.Add(ProfileConstantAcceleration(q3,0.2))
    trajectory.Add(ProfileConstantAcceleration(q0,0.5))
    trajectory.Add(ProfileConstantAcceleration(q2,0.2))
    


    mbs.Assemble()
    mbs.variables['trajectory'] = trajectory

    # intialize variables for RL training
    mbs.variables['legsInit'] = [
        0.0, 0.0, 0.0, 0.0,       # HipX FL,FR,BL,BR
        0.0, 0.0, 0.0, 0.0,       # HipY FL,FR,BL,BR
        -0.1, -0.1, -0.1, -0.1]    # Knee FL,FR,BL,BR
    mbs.variables['legMarkers'] = mLegs
    mbs.variables['legRadius'] = W_leg
    mbs.variables['zContact'] = W_leg/2#zContact

    return mbs, SC, oKT, nKT



def PreStepUF(mbs, t):
    #rd = mbs.variables['rd']

    [u, v, a] = mbs.variables['trajectory'].Evaluate(t)

    mbs.SetObjectParameter(oKT, 'jointPositionOffsetVector', u)
    mbs.SetObjectParameter(oKT, 'jointVelocityOffsetVector', v)

    return True

if __name__ == "__main__":

    # -------------------------------------------------
    # Render only Model to check 
    # -------------------------------------------------

    mbs, SC, oKT, nKT, = GetModel()

    # mbs.variables['rd'] = rd
    # mbs.variables['trajectory'] = rd.trajectory


    mbs.SetPreStepUserFunction(PreStepUF)

    SC.visualizationSettings.general.drawWorldBasis = True
    SC.visualizationSettings.bodies.kinematicTree.showJointFrames = True
    SC.visualizationSettings.bodies.kinematicTree.frameSize = 0.15
    SC.visualizationSettings.contact.showSpheres = True


    simSettings = exu.SimulationSettings()

    simSettings.timeIntegration.numberOfSteps = 80000   # 200.000 Schritte (hängt von der Zeit ab)
    simSettings.timeIntegration.endTime = 4           # 4 Sekunden (braucht er ca. zum Umfallen)
    simSettings.timeIntegration.generalizedAlpha.spectralRadius = 0.8
    simSettings.timeIntegration.verboseMode = 1        # Ausgabe an
    simSettings.timeIntegration.discontinuous.useRecommendedStepSize = False

    simSettings.linearSolverType = exu.LinearSolverType.EigenSparse # 

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
    #mbs.SolveDynamic(simSettings)

    mbs.SolveDynamic(simSettings,
                    solverType=exu.DynamicSolverType.VelocityVerlet # Expliziter Solver
                    ) 
    SC.renderer.DoIdleTasks()
    exu.StopRenderer()
    mbs.SolutionViewer()


    # Plot Sensors
    mbs.PlotSensor(sensorNumbers=[0],components=[0],closeAll=True) # Bewegung Plattform in x-Richtung
    mbs.PlotSensor(sensorNumbers=[1],components=[0],closeAll=False) # Geschwingkeit Plattform in x-Richtung
    mbs.PlotSensor(sensorNumbers=[2],components=[2],closeAll=False) # Winkel Plattform
    mbs.PlotSensor(sensorNumbers=[3],components=[2],closeAll=False) # Winkelgeschwingkeit Plattform
