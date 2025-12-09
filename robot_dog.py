import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics
import numpy as np

def RobotDog(SC, mbs, 
            platformInertia = None,
            legInertia = None,
            referenceCoordinates = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
            frictionCoeff = 1, stiffnessGround = 1e5,
            gContact = None, 
            useGeneralContact = False #generalcontact shows large errors currently
            ):
    #add class which can be returned to enable user to access parameters
    class rd: pass


    # -------------------------------------------------
    # Boden
    # -------------------------------------------------
    rd.gGround = graphics.CheckerBoard(normal=[0,0,1], size=dimGroundX, size2=dimGroundY, nTiles=8, zOffset=-1e-5)
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
    rd.gContact.frictionProportionalZone = frictionProportionalZone

    rd.gContact.SetFrictionPairings(np.diag([rd.frictionCoeff, 0.0]))  # Material 0 hat Reibung, 1 nicht

    # WICHTIG: Suchbaum-Box manuell setzen (verhindert "size must not be zero")
    rd.gContact.SetSearchTreeBox(
        pMin=[-10,-10,-1],
        pMax=[ 10, 10, 1]
    )

    [meshPoints, meshTrigs] = graphics.ToPointsAndTrigs(rd.gGround)

    rd.gContact.AddTrianglesRigidBodyBased(
        rigidBodyMarkerIndex = rd.mGround,
        contactStiffness = frictionProportionalZone,
        contactDamping = rd.dampingGround,
        frictionMaterialIndex = frictionIndexGround,
        pointList = meshPoints,
        triangleList = meshTrigs
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
    rd.nJoints = 3 + 3 + 4 + 4 
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
    0,0,0,0,-0.1,-0.1,-0.1,-0.1
    ]


    rd.nKT = mbs.AddNode(NodeGenericODE2(
        referenceCoordinates=referenceCoordinates,
        initialCoordinates=initialAngles,
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

    # 4 Hüftgelenke
    rd.jointTypes += [exu.JointType.RevoluteY]*4

    # 4 Kniegelenke
    rd.jointTypes += [exu.JointType.RevoluteY]*4


    rd.linkParents = [-1, 0, 1, 2, 3, 4, 5, 5, 5, 5, 6, 7, 8, 9] #    

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

    # Oberschenkel
    for pos in rd.hipPositions:
        rd.jointOffsets.Append(pos)
        rd.linkInertiasCOM.Append(rd.thighInertia.InertiaCOM())
        rd.linkCOMs.Append(rd.thighInertia.COM())
        rd.linkMasses.append(rd.thighInertia.Mass())
        rd.gList.append(rd.gThigh)

    # Kniegelenk sitzt am Ende des Oberschenkels
    for i in range(4):
        rd.jointOffsets.Append([0,0,-L_thigh])  # Knie am Beinende
        rd.linkInertiasCOM.Append(rd.shinInertia.InertiaCOM())
        rd.linkCOMs.Append(rd.shinInertia.COM())
        rd.linkMasses.append(rd.shinInertia.Mass())
        rd.gList.append(rd.gShin)





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
        forceUserFunction=UF_KinematicTreeForces,
        visualization=VObjectKinematicTree(graphicsDataList=rd.gList)
    ))

    rd.mLegs = []

    # KNIE GELENK
    for i in range(4):
        rd.mLeg = mbs.AddMarker(MarkerKinematicTreeRigid(objectNumber=rd.oKT,
                                                    linkNumber=rd.platformIndex+1+i,
                                                    localPosition=[0,0, -L_thigh]))
        rd.mLegs.append(rd.mLeg)
        rd.gContact.AddSphereWithMarker(rd.mLeg,
                                    radius=0.5*W_leg,
                                    contactStiffness=1e5,
                                    contactDamping=1e3,
                                    frictionMaterialIndex=0)
        
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

    rd.markers = []

    for pos in offsets:
        m = mbs.AddMarker(
            MarkerKinematicTreeRigid(
                objectNumber=rd.oKT,
                linkNumber=rd.platformIndex,
                localPosition=pos
            )
        )
        rd.markers.append(m)

        rd.gContact.AddSphereWithMarker(
            markerIndex=m,
            radius=sph_rad,
            contactStiffness=1e5,
            contactDamping=1e3,
            frictionMaterialIndex=0
        )




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
# Render only Model to check 
# -------------------------------------------------
SC = exu.SystemContainer()
mbs = SC.AddSystem()

useGeneralContact = False
usePenalty = True
rd = RobotDog(SC, mbs,useGeneralContact=useGeneralContact, 
                                usePenalty=usePenalty)
mbs.Assemble()


SC.visualizationSettings.general.drawWorldBasis = True
SC.visualizationSettings.bodies.kinematicTree.showJointFrames = True
SC.visualizationSettings.bodies.kinematicTree.frameSize = 0.15


simSettings = exu.SimulationSettings()

simSettings.timeIntegration.numberOfSteps = 40000   # 200.000 Schritte (hängt von der Zeit ab)
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
mbs.SolveDynamic(simSettings,
                  # solverType=exu.DynamicSolverType.VelocityVerlet # Expliziter Solver
                  ) 
SC.renderer.DoIdleTasks()
exu.StopRenderer()
mbs.SolutionViewer()


# Plot Sensors
mbs.PlotSensor(sensorNumbers=[0],components=[0],closeAll=True) # Bewegung Plattform in x-Richtung
mbs.PlotSensor(sensorNumbers=[1],components=[0],closeAll=False) # Geschwingkeit Plattform in x-Richtung
mbs.PlotSensor(sensorNumbers=[2],components=[2],closeAll=False) # Winkel Plattform
mbs.PlotSensor(sensorNumbers=[3],components=[2],closeAll=False) # Winkelgeschwingkeit Plattform

