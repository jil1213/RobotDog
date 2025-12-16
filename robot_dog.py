import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics
import numpy as np
from exudyn.robotics.motion import Trajectory, ProfileConstantAcceleration, ProfilePTP

def RobotDog(
            platformInertia = None,
            legInertia = None,
            referenceCoordinates = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],
            L_body = 1.0,   #Länge Body
            W_body = 0.3,
            H_body = 0.2,
            platformMass = 5,
            body_offset = 0.8,  # Fallhöhe des Bodys über dem Boden
            dimGroundX = 8, dimGroundY = 8,
            gravity = [0,0,-9.81],
            L_thigh = 0.3,
            L_shin  = 0.2,
            W_leg = 0.1,
            density = 500,
            legMass = 0.05,
            pControl = 0,
            dControl = 0.02,
            usePenalty = True, #use penalty formulation in case useGeneralContact=False
            frictionProportionalZone = 0.01, # own parameter other example has 0.025
            frictionCoeff = 0.8, stiffnessGround = 1e5,
            gContact = None, 
            useGeneralContact = False #generalcontact shows large errors currently
            ):
    #add class which can be returned to enable user to access parameters
    class rd: pass

    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    g = [0,0,-9.81]
    # -------------------------------------------------
    # Boden
    # -------------------------------------------------

    k = 10e4
    d = 0.01*k
    frictionCoeff = 0.8 #0.5
    ss = 1

    planeL = 16

    zContact = -0.7+0.01214+6.45586829e-02 #contact at beginning, leg = [0,0.2*pi,-0.3*pi]
    p0 = [0,0,0]
    mbs.variables['zContact'] = zContact

    rd.gGroundSimple = graphics.CheckerBoard(p0,size=planeL, nTiles=1) #only 1 tile for efficiency

    rd.gGround = graphics.CheckerBoard(p0,size=planeL, nTiles=16)
    rd.oGround = mbs.AddObject(ObjectGround(referencePosition=[0,0,0],
                                        visualization=VObjectGround(graphicsData=[rd.gGround])))

    rd.mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=rd.oGround))

    ###   Kontakt mit dem Boden ------
    rd.frictionCoeff = frictionCoeff
    rd.stiffnessGround = stiffnessGround
    rd.dampingGround = rd.stiffnessGround*0.01
    #if gContact == None and useGeneralContact: # it has to be initialized all the time to run the simulation
    frictionIndexGround = 0
    frictionIndexFree = 1

    rd.gContact = mbs.AddGeneralContact()
    rd.gContact.frictionProportionalZone = 0.01

    rd.gContact.SetFrictionPairings(frictionCoeff*np.eye(1))
    rd.gContact.SetSearchTreeCellSize(numberOfCells=[ss,ss,1])
    # WICHTIG: Suchbaum-Box manuell setzen (verhindert "size must not be zero")
    stFact = 0.3
    rd.gContact.SetSearchTreeBox(pMin=np.array([-0.5*planeL*stFact,-0.5*planeL*stFact,-1]),
                               pMax=np.array([0.5*planeL*stFact,0.5*planeL*stFact,0.1]))

    [meshPoints, meshTrigs] = graphics.ToPointsAndTrigs(rd.gGroundSimple)

    rd.gContact.AddTrianglesRigidBodyBased(
        rigidBodyMarkerIndex = rd.mGround,
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
    if platformInertia is None:
        rd.platformInertia = InertiaCuboid(density, [L_body, W_body, H_body]).Translated([0,0,0])
    else: 
        rd.platformInertia = RigidBodyInertia(mass=platformMass, inertiaTensorAtCOM=np.diag(platformInertia))
    if legInertia is None:
        rd.thighInertia = InertiaCylinder(density, length=L_thigh, outerRadius=0.5 *W_leg, axis=2).Translated([0,0,0])
        rd.shinInertia = InertiaCylinder(density, length=L_shin, outerRadius=0.5 *W_leg, axis=2).Translated([0,0,0])
    else: 
        rd.thighInertia = RigidBodyInertia(mass=legMass, inertiaTensorAtCOM=np.diag(legInertia))
        rd.shinInertia  = RigidBodyInertia(mass=legMass, inertiaTensorAtCOM=np.diag(legInertia))

    # -------------------------------------------------
    # Node
    # -------------------------------------------------
    # 3 Plattform XYZ-Translation + 3 Plattform Drehung + 4 Hüfte + 4 Knie = 14 DOF
    rd.nJoints = 3 + 3 + 4 + 4 + 4
    # referenceCoordinates = [0]*rd.nJoints
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
    0,0,0,0,0,0,0,0,-0.1,-0.1,-0.1,-0.1
    ]

    q0 = [0, 0, 0.6-0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rd.nKT = mbs.AddNode(NodeGenericODE2(
        referenceCoordinates=referenceCoordinates,
        initialCoordinates=q0,
        initialCoordinates_t=[0]*rd.nJoints,
        numberOfODE2Coordinates=rd.nJoints))

    # -------------------------------------------------
    # KinematicTree
    # -------------------------------------------------
    # 6 Gelenke der Plattform
    rd.jointTypes = [
        exu.JointType.PrismaticX,
        exu.JointType.PrismaticY,
        exu.JointType.PrismaticZ,
        exu.JointType.RevoluteX,  
        exu.JointType.RevoluteY,
        exu.JointType.RevoluteZ,
    ]

    # 4 Hüftgelenke X-Achse
    rd.jointTypes += [exu.JointType.RevoluteX]*4
    
    # 4 Hüftgelenke Y-Achse
    rd.jointTypes += [exu.JointType.RevoluteY]*4

    # 4 Kniegelenke
    rd.jointTypes += [exu.JointType.RevoluteY]*4


    #rd.linkParents = [-1, 0, 1, 2, 3, 4, 5, 5, 5, 5,5,5,5,5, 6, 7, 8, 9] #    

    rd.linkParents = [
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


    rd.platformIndex = 5
    print(len(rd.jointTypes))

    # -------------------------------------------------
    # Grafik
    # -------------------------------------------------
    rd.gBody = [
        graphics.Brick([0, 0, 0],
                    [L_body, W_body, H_body],
                    color=graphics.color.red),
        graphics.Basis(length=0.2)
    ]

    rd.gThigh = [
        graphics.Cylinder([0,0,0], [0,0,-L_thigh], W_leg*0.5, color=graphics.color.blue),
        graphics.Basis(length=0.1)
    ]

    rd.gShin = [
        graphics.Cylinder([0,0,0], [0,0,-L_shin], W_leg*0.4, color=graphics.color.green),
        graphics.Basis(length=0.1)
    ]
    
    rd.gZero =[]

    # -------------------------------------------------
    # JointOffsets, Massen, COMs
    # -------------------------------------------------
    rd.jointTransformations = exu.Matrix3DList()
    rd.jointOffsets = exu.Vector3DList()
    rd.linkCOMs = exu.Vector3DList()
    rd.linkInertiasCOM = exu.Matrix3DList()
    rd.linkMasses = []
    rd.gList = []

    # Standard Transformation (keine Drehung)
    for i in range(len(rd.jointTypes)):
        rd.jointTransformations.Append(np.eye(3))

    # Ersten 6 Gelenke sind Masselos 
    for i in range(5):
        rd.jointOffsets.Append([0,0,0])
        rd.linkInertiasCOM.Append(np.zeros((3,3)))
        rd.linkCOMs.Append([0,0,0])
        rd.linkMasses.append(0)
        rd.gList.append([])

    # Plattform
    rd.jointOffsets.Append([0,0,0])
    rd.linkInertiasCOM.Append(rd.platformInertia.InertiaCOM())
    rd.linkCOMs.Append(rd.platformInertia.COM())
    rd.linkMasses.append(rd.platformInertia.Mass())
    rd.gList.append(rd.gBody)

    # Hüften (4 Oberschenkelaufhängungen)
    hipX = 0.5*L_body
    hipY = 0.5*W_body + 0.5*W_leg
    hipZ = 0

    rd.hipPositions = [
        [-hipX,  hipY, hipZ],
        [ hipX,  hipY, hipZ],
        [-hipX, -hipY, hipZ],
        [ hipX, -hipY, hipZ]
    ]

    # Oberschenkel X-Achse
    for pos in rd.hipPositions:
        rd.jointOffsets.Append(pos)
        rd.linkInertiasCOM.Append(np.zeros((3,3)))
        rd.linkCOMs.Append([0,0,0])
        rd.linkMasses.append(0)
        rd.gList.append(rd.gZero)

    # Oberschenkel Y-Achse

    rd.hipPositions = np.zeros((4,3))

    for pos in rd.hipPositions:
        rd.jointOffsets.Append(pos)
        rd.linkInertiasCOM.Append(np.zeros((3,3)))
        rd.linkCOMs.Append([0,0,0])
        rd.linkMasses.append(0)
        rd.gList.append(rd.gThigh)


    # Kniegelenk sitzt am Ende des Oberschenkels
    for i in range(4):
        rd.jointOffsets.Append([0,0,-L_thigh])  # Knie am Beinende
        rd.linkInertiasCOM.Append(rd.shinInertia.InertiaCOM())
        rd.linkCOMs.Append(rd.shinInertia.COM())
        rd.linkMasses.append(rd.shinInertia.Mass())
        rd.gList.append(rd.gShin)


    # -------------------------------------------------
    # KinematicTree Objekt
    # -------------------------------------------------
    rd.jointPControlVector = [0]*6 + [300]*12
    rd.jointDControlVector = [0]*6 + [30]*12

    rd.oKT = mbs.AddObject(ObjectKinematicTree(
        nodeNumber=rd.nKT,
        jointTypes=rd.jointTypes,
        linkParents=rd.linkParents,
        jointTransformations=rd.jointTransformations,
        jointOffsets=rd.jointOffsets,
        linkInertiasCOM=rd.linkInertiasCOM,
        linkCOMs=rd.linkCOMs,
        linkMasses=rd.linkMasses,
        baseOffset=[0.,0.,0.],
        gravity=gravity,
        jointPControlVector=rd.jointPControlVector,
        jointDControlVector=rd.jointDControlVector,
        jointPositionOffsetVector=[0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        jointVelocityOffsetVector=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #forceUserFunction=UF_KinematicTreeForces,
        visualization=VObjectKinematicTree(graphicsDataList=rd.gList)
    ))

    rd.mLegs = []


    # FÜSSE (Ende des Unterschenkels)
    for i in range(4):
        rd.mLeg = mbs.AddMarker(MarkerKinematicTreeRigid(objectNumber=rd.oKT,
                                                    linkNumber=rd.linkParents[-1]+1+i,
                                                    localPosition=[0,0, -L_shin]))
        rd.mLegs.append(rd.mLeg)
        rd.gContact.AddSphereWithMarker(rd.mLeg,
                                    radius=W_leg,
                                    contactStiffness=1e5,
                                    contactDamping=1e3,
                                    frictionMaterialIndex=0)


    # -------------------------------------------------
    # Add Sensors
    # -------------------------------------------------
    sPlatformPos = mbs.AddSensor(SensorKinematicTree(objectNumber=rd.oKT, linkNumber = rd.platformIndex,
                                                            storeInternal=True, outputVariableType=exu.OutputVariableType.Position))
    sPlatformVel = mbs.AddSensor(SensorKinematicTree(objectNumber=rd.oKT, linkNumber = rd.platformIndex,
                                                            storeInternal=True, outputVariableType=exu.OutputVariableType.Velocity))
    sPlatformAng = mbs.AddSensor(SensorKinematicTree(objectNumber=rd.oKT, linkNumber = rd.platformIndex,
                                                            storeInternal=True, outputVariableType=exu.OutputVariableType.Rotation))
    sPlatformAngVel = mbs.AddSensor(SensorKinematicTree(objectNumber=rd.oKT, linkNumber = rd.platformIndex,
                                                            storeInternal=True, outputVariableType=exu.OutputVariableType.AngularVelocity))

    for i in range(8):
        mbs.AddSensor(SensorKinematicTree(objectNumber=rd.oKT, linkNumber = rd.platformIndex+1+i,
                                                                storeInternal=True, outputVariableType=exu.OutputVariableType.Rotation))
        mbs.AddSensor(SensorKinematicTree(objectNumber=rd.oKT, linkNumber = rd.platformIndex+1+i,
                                                                storeInternal=True, outputVariableType=exu.OutputVariableType.AngularVelocity))


    # -------------------------------------------------
    # configurations and trajectory
    # -------------------------------------------------

    q0 = [0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    q1 = [0, 0, 0.8, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    q2 = [0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rd.trajectory = Trajectory(initialCoordinates=q0, initialTime=0)
    rd.trajectory.Add(ProfileConstantAcceleration(q1, 1.0))
    rd.trajectory.Add(ProfileConstantAcceleration(q2, 1.0))


    # legsInit = [0,36*pi/180,-54*pi/180.]

    # q0 = np.zeros(rd.nJoints) #zero angle configuration +6 Floating Base
    # q1 = np.zeros(rd.nJoints) 

    # #test simple motion
    # #position fpr PD-Control:
    # #ordering of legs:
    # #front left, front right, back left, back right
    # q0 = [0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # # #q1 = [0,0,0, 0,0,0] + list(np.array(leg))*4
    # # # q1 = [0,0,0, 0,0,0, 0,0.1*pi,-0.1*pi, 0,0,0, 0,0,0, 0,0,0]
    # # # q1 = [0,0,0, 0,0,0, 0,0,0, 0,0.1*pi,-0.1*pi, 0,0,0, 0,0,0]
    # # q1 = [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0.1*pi,-0.1*pi, 0,0,0]
    q1 = [0,0,0, 0,0,0, 0.25*pi,0,0,-0.25*pi,0,0,0.25*pi,0,0,-0.25*pi,0,0,]
    q2 = [0,0,0, 0,0,0, 0,0.5*pi,-0.9*pi,0,0.5*pi,-0.9*pi,0,0.5*pi,-0.9*pi,0,0.5*pi,-0.9*pi] # [[0]*6, hip_y, hip_x, knee_x, ....]
    q3 = [0,0,0, 0,0,0, 0,0.5*pi,-0.9*pi,0,0.5*pi,-0.9*pi,0,0.5*pi,-0.9*pi,0,0.5*pi,-0.9*pi] # [[0]*6, hip_y, hip_x, knee_x, ....]
    q4 = [0,0,0,0,0,0,0,0.5*pi,-0.9*pi,0,0.5*pi,-0.9*pi,0,0,0,0,0,0]

    # q1 = [0,0,0,0,0,0,0,0.4*pi,-0.5*pi,0,0,0,0,0,0,0,0,0] # [[0]*6, hip_y, hip_x, knee_x, ....]
    # q2 = q0

    #trajectory generated with optimal acceleration profiles:
    rd.trajectory = Trajectory(initialCoordinates=q0, initialTime=0)
    rd.trajectory.Add(ProfileConstantAcceleration(q0,0.3))
    rd.trajectory.Add(ProfileConstantAcceleration(q1,1))
    rd.trajectory.Add(ProfileConstantAcceleration(q2,1))
    rd.trajectory.Add(ProfileConstantAcceleration(q0,0.7))
    rd.trajectory.Add(ProfileConstantAcceleration(q3,0.7))
    rd.trajectory.Add(ProfileConstantAcceleration(q4,0.7))
    rd.trajectory.Add(ProfileConstantAcceleration(q0,0.4))

    mbs.Assemble()
    mbs.variables['rd'] = rd
    mbs.variables['trajectory'] = rd.trajectory

    # intialize variables for RL training
    mbs.variables['legsInit'] = [
        0.0, 0.0, 0.0, 0.0,       # HipX FL,FR,BL,BR
        0.0, 0.0, 0.0, 0.0,       # HipY FL,FR,BL,BR
        -0.1, -0.1, -0.1, -0.1]    # Knee FL,FR,BL,BR
    mbs.variables['legMarkers'] = rd.mLegs
    mbs.variables['legRadius'] = W_leg
    mbs.variables['zContact'] = zContact

    return mbs, SC, rd.oKT, rd.nKT




def PreStepUF(mbs, t):
    rd = mbs.variables['rd']

    [u, v, a] = mbs.variables['trajectory'].Evaluate(t)

    mbs.SetObjectParameter(rd.oKT, 'jointPositionOffsetVector', u)
    mbs.SetObjectParameter(rd.oKT, 'jointVelocityOffsetVector', v)

    return True

if __name__ == "__main__":

    # -------------------------------------------------
    # Render only Model to check 
    # -------------------------------------------------

    mbs, SC, oKT, nKT, = RobotDog()

    # mbs.variables['rd'] = rd
    # mbs.variables['trajectory'] = rd.trajectory

    mbs.Assemble()

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
