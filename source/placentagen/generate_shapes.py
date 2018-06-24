#!/usr/bin/env python
import numpy as np
from scipy.spatial import Delaunay
from . import pg_utilities
from opencmiss.iron import iron
import random
"""
.. module:: generate_shapes
  :synopsis: Contains code to generate placental shapes for generic placental models.

:synopsis:Contains code to generate placental shapes for generic placental models \n
 (i.e. from literature measures without specific data from an individual
 
"""

def equispaced_data_in_ellipsoid(n, volume, thickness, ellipticity):
    """ Generates equally spaced data points in an ellipsoid.

    Inputs:
       - n: number of data points which we aim to generate
       - volume: volume of ellipsoid
       - thickness: placental thickness (z-dimension)
       - ellipticity: ratio of y to x axis dimensions

    Returns:
       - Edata: A nx3 array of datapoints, with each point being defined by its x-,y-, and z- coordinates

    A way you might want to use me is:

    >>> n = 100
    >>> volume = 10
    >>> thickness = 3
    >>> ellipticity = 1.1
    >>> equispaced_data_in_ellipsoid(n, volume, thickness, ellipticity)

   This will return 100 data points in an ellipse with z-axis thickness 3, volume 10, and with the y-axis dimension 1.1 times the x-axis dimension.

    """
    # Generates equally spaced data points in an ellipsoid with the following inputs
    # n=number of data points which we aim to generate
    # volume=volume of ellipsoid
    # thickness = placental thickness (z-dimension)
    # ellipticity = ratio of y to x axis dimensions
    data_spacing = (volume / n) ** (1.0 / 3.0)
    radii = pg_utilities.calculate_ellipse_radii(volume, thickness, ellipticity)
    z_radius = radii['z_radius']
    x_radius = radii['x_radius']
    y_radius = radii['y_radius']

    # Aiming to generate seed points that fill a cuboid encompasing the placental volume then remove seed points that
    # are external to the ellipsoid

    num_data = 0  # zero the total number of data points

    # Calculate the number of points that should lie in each dimension in a cube
    nd_x = np.floor(2.0 * (x_radius + data_spacing) / data_spacing)
    nd_y = np.floor(2.0 * (y_radius + data_spacing) / data_spacing)
    nd_z = np.floor(2.0 * (z_radius + data_spacing) / data_spacing)
    nd_x = int(nd_x)
    nd_y = int(nd_y)
    nd_z = int(nd_z)
    # Set up edge node coordinates
    x_coord = np.linspace(-x_radius - data_spacing / 2.0, x_radius + data_spacing / 2.0, nd_x)
    y_coord = np.linspace(-y_radius - data_spacing / 2.0, y_radius + data_spacing / 2.0, nd_y)
    z_coord = np.linspace(-z_radius - data_spacing / 2.0, z_radius + data_spacing / 2.0, nd_z)

    # Use these vectors to form a unifromly spaced grid
    data_coords = np.vstack(np.meshgrid(x_coord, y_coord, z_coord)).reshape(3, -1).T

    # Store nodes that lie within ellipsoid
    Edata = np.zeros((nd_x * nd_y * nd_z, 3))
    for i in range(len(data_coords)):  # Loop through grid
        coord_check = pg_utilities.check_in_ellipsoid(data_coords[i][0], data_coords[i][1], data_coords[i][2], x_radius,
                                                      y_radius, z_radius)

        if coord_check is True:  # Has to be strictly in the ellipsoid
            Edata[num_data, :] = data_coords[i, :]  # add to data array
            num_data = num_data + 1
    Edata.resize(num_data, 3)  # resize data array to correct size

    print('Data points within ellipsoid allocated. Total = ' + str(len(Edata)))

    return Edata


def uniform_data_on_ellipsoid(n, volume, thickness, ellipticity, random_seed):
    """ Generates equally spaced data points on the positive z-surface of an ellipsoid

    Inputs:
       - n: number of data points which we aim to generate
       - volume: volume of ellipsoid
       - thickness: placental thickness (z-dimension)
       - ellipticity: ratio of y to x axis dimensions

    Returns:
       - chorion_data: A nx3 array of datapoints, with each point being defined by its x-,y-, and z- coordinates

    A way you might want to use me is:

    >>> n = 100
    >>> volume = 10
    >>> thickness = 3
    >>> ellipticity = 1.1
    >>> equispaced_data_on_ellipsoid(n, volume, thickness, ellipticity)

   This will return 100 data points on the positive z-surface ellipse with z-axis thickness 3, volume 10, and with the y-axis dimension 1.1 times the x-axis dimension.

    """
    radii = pg_utilities.calculate_ellipse_radii(volume, thickness, ellipticity)
    z_radius = radii['z_radius']
    x_radius = radii['x_radius']
    y_radius = radii['y_radius']
    area_estimate = np.pi * x_radius * y_radius
    data_spacing = 0.85 * np.sqrt(area_estimate / n)

    chorion_data = np.zeros((n, 3))
    np.random.seed(random_seed)
    generated_seed = 0
    acceptable_attempts = n * 1000  # try not to have too many failures
    attempts = 0

    while generated_seed < n and attempts < acceptable_attempts:
        # generate random x-y coordinates between negative and positive radii
        new_x = np.random.uniform(-x_radius, x_radius)
        new_y = np.random.uniform(-y_radius, y_radius)
        # check if new coordinate is on the ellipse
        if ((new_x / x_radius) ** 2 + (new_y / y_radius) ** 2) < 1:  # on the surface
            if generated_seed == 0:
                generated_seed = generated_seed + 1
                new_z = pg_utilities.z_from_xy(new_x, new_y, x_radius, y_radius, z_radius)
                chorion_data[generated_seed - 1][:] = [new_x, new_y, new_z]
            else:
                reject = False
                for j in range(0, generated_seed + 1):
                    distance = (chorion_data[j - 1][0] - new_x) ** 2 + (chorion_data[j - 1][1] - new_y) ** 2
                    distance = np.sqrt(distance)
                    if distance <= data_spacing:
                        reject = True
                        break
                if reject is False:
                    generated_seed = generated_seed + 1
                    new_z = pg_utilities.z_from_xy(new_x, new_y, x_radius, y_radius, z_radius)
                    chorion_data[generated_seed - 1][:] = [new_x, new_y, new_z]

        attempts = attempts + 1
    chorion_data.resize(generated_seed, 3)  # resize data array to correct size
    print('Data points on ellipsoid allocated. Total = ' + str(len(chorion_data)))

    return chorion_data


def gen_rectangular_mesh(volume, thickness, ellipticity, x_spacing, y_spacing, z_spacing):
    # Generates equally spaced data nodes and elements and constructs a rectangular 'mesh' that covers the space that is
    # made up of an ellipsoidal placenta
    # volume=volume of ellipsoid
    # thickness = placental thickness (z-dimension)
    # ellipticity = ratio of y to x axis dimensions
    # X,Y,Z spacing is the number of elements required in each of the x, y z directions

    # Calculate the dimensions of the ellipsoid
    radii = pg_utilities.calculate_ellipse_radii(volume, thickness, ellipticity)
    z_radius = radii['z_radius']
    x_radius = radii['x_radius']
    y_radius = radii['y_radius']

    print(x_radius,y_radius,z_radius)
    # z height of ellipsoid is 2* zradius
    # We want number of nodes to cover height and have prescribed spaing
    nnod_x = int(np.ceil(x_radius * 2.0 / x_spacing)) + 1
    x_width = x_spacing * (nnod_x - 1)
    nnod_y = int(np.ceil(y_radius * 2.0 / y_spacing)) + 1
    y_width = y_spacing * (nnod_y - 1)
    nnod_z = int(np.ceil(z_radius * 2.0 / z_spacing)) + 1
    z_width = z_spacing * (nnod_z - 1)

    # Create linspaces for x y and z coordinates
    x = np.linspace(-x_width / 2.0, x_width / 2.0, nnod_x)  # linspace for x axis
    y = np.linspace(-y_width / 2.0, y_width / 2.0, nnod_y)  # linspace for y axis
    z = np.linspace(-z_width / 2.0, z_width / 2.0, nnod_z)  # linspace for z axis
    node_loc_temp = np.vstack(np.meshgrid(y, z, x)).reshape(3, -1).T  # generate nodes for rectangular mesh

    node_loc = np.zeros((nnod_x*nnod_y*nnod_z,3))
    for i in range(0,len(node_loc)):
        node_loc[i][0] = node_loc_temp[i][2]
        node_loc[i][1] = node_loc_temp[i][0]
        node_loc[i][2] = node_loc_temp[i][1]

    # Generating the element connectivity of each cube element, 8 nodes for each 3D cube element
    num_elems = (nnod_x - 1) * (nnod_y - 1) * (nnod_z - 1)
    elems = np.zeros((num_elems, 9),
                     dtype=int)  # this stores first element number and then the nodes of each mesh element
    element_number = 0

    ne = 0
    # loop through elements
    for k in range(1, nnod_z):
        for j in range(1, nnod_y):
            for i in range(1, nnod_x):
                elems[ne][0] = ne  # store element number
                elems[ne][1] = (i - 1) + (nnod_x) * (j - 1) + nnod_x * nnod_y * (k - 1) #lowest coordinates
                elems[ne][2] = elems[ne][1] + 1 #add one in x
                elems[ne][3] = elems[ne][1] + nnod_x #go through x and find first in y
                elems[ne][4] = elems[ne][3] + 1 #add one in y
                elems[ne][5] = elems[ne][1] + nnod_x * nnod_y #same as 1 -4 but at higher z -coord
                elems[ne][6] = elems[ne][2] + nnod_x * nnod_y
                elems[ne][7] = elems[ne][3] + nnod_x * nnod_y
                elems[ne][8] = elems[ne][4] + nnod_x * nnod_y
                ne = ne + 1

    return {'nodes': node_loc, 'elems': elems, 'total_nodes': nnod_x * nnod_y * nnod_z,
            'total_elems': (nnod_x - 1) * (nnod_y - 1) * (nnod_z - 1)}

def gen_mesh_darcy(rectangular_mesh,volume, thickness, ellipticity, spacing):
    #rect_mesh=gen_rectangular_mesh(volume, thickness, ellipticity, spacing, spacing, spacing)
    nodes=rectangular_mesh['nodes']
    #print nodes
    radii = pg_utilities.calculate_ellipse_radii(volume, thickness, ellipticity)
    z_radius = radii['z_radius']
    x_radius = radii['x_radius']
    y_radius = radii['y_radius']
    #print x_radius, y_radius, z_radius
    #nodeSpacing = (n/(2*x_radius*2*y_radius*2*z_radius)) **(1./3)#####33

    #Set up edge node vectors:
    #xVector = np.linspace(-x_radius, x_radius, 2*x_radius*nodeSpacing)##########
    #yVector = np.linspace(-y_radius, y_radius, 2*y_radius*nodeSpacing)###########
    #zVector = np.linspace(-z_radius, z_radius, 2*z_radius*nodeSpacing)#########

    #Use these vectors to make a uniform cuboid grid
    #nodes = np.vstack(np.meshgrid(xVector,yVector,zVector)).reshape(3,-1).T#############
    
    ellipsoid_node=np.zeros((len(nodes),3))
    
    count=0
    for nnode in range (0, len(nodes)):
       coord_point = nodes[nnode][0:3]
       
       inside=pg_utilities.check_in_on_ellipsoid(coord_point[0], coord_point[1], coord_point[2], x_radius, y_radius, z_radius)
       if inside:
          ellipsoid_node[count,:]=coord_point[:]
          count=count+1
    ellipsoid_node.resize(count,3)     
    #ellipsoid_node=ellipsoid_node[np.all(ellipsoid_node != 0, axis=1)]
    
    xyList = ellipsoid_node[:,[0,1]]
    xyListUnique = np.vstack({tuple(row) for row in xyList})#one layer containing most xy (i.e xy of middle most layer)
    
    for xyColumn in xyListUnique:
        
        xyNodes = np.where(np.all(xyList == xyColumn, axis = 1))[0]#List all nodes in each column
        if len(xyNodes) > 1:
           x_coord=ellipsoid_node[xyNodes[0],0]
           y_coord=ellipsoid_node[xyNodes[0],1]
           ellipsoid_node[xyNodes[len(xyNodes) - 1],2] = pg_utilities.z_from_xy(x_coord, y_coord, x_radius, y_radius, z_radius)   
           ellipsoid_node[xyNodes[0],2] =-1*( pg_utilities.z_from_xy(x_coord, y_coord, x_radius, y_radius, z_radius))

    
    pyMesh = Delaunay(ellipsoid_node)

    #Build arrays to pass into openCMISS conversion:
    node_loc = pyMesh.points
    temp_elems = pyMesh.simplices
   #CHECK ELEMENTS FOR 0 VOLUME:

   #Initialise tolerance and loop variables:
    min_vol = 0.00001
    index = 0
    indexArr = []

    for element in temp_elems:
     #print element
     x_coor = []
     y_coor = []
     z_coor = []
     for node in element:
     #find coordinates of nodes
        x_coor.append(node_loc[node][0])
        y_coor.append(node_loc[node][1])
        z_coor.append(node_loc[node][2])
     #Use coordinates to calculate volume of element
     
     vmat = np.vstack((x_coor,y_coor,z_coor,[1.0,1.0,1.0,1.0]))#matrix of coor of element
     elem_volume = (1/6.0) * abs(np.linalg.det(vmat))
     
     #update index list of good elements
     if elem_volume > min_vol:
       
       indexArr.append(index)
     index = index+1
     
    #update arrays without 0 volume elements, to pass into openCMISS
    elems = temp_elems[indexArr,:]
    for i in range(len(elems)):
       elems[i] = [x+1 for x in elems[i]]
    element_array = range(1, len(elems)+1)
    node_array = range(1, len(node_loc)+1)
    #print node_loc
    #print elems
    #print element_array
    #print node_array
    
    return {'nodes': node_loc, 'elems': elems, 'element_array':element_array,'node_array': node_array}


def darcynode_in_sampling_grid(rectangular_mesh, darcy_node_loc):
    
    
    darcy_node_elems = np.zeros(len(darcy_node_loc), dtype=int)
    elems = rectangular_mesh['elems']
    nodes = rectangular_mesh['nodes']
    startx = np.min(nodes[:, 0])
    xside = nodes[elems[0][8]][0] - nodes[elems[0][1]][0]
    endx = np.max(nodes[:, 0])
    nelem_x = (endx - startx) / xside
    starty = np.min(nodes[:, 1])
    yside = nodes[elems[0][8]][1] - nodes[elems[0][1]][1]
    endy = np.max(nodes[:, 1])
    nelem_y = (endy - starty) / yside
    startz = np.min(nodes[:, 2])
    zside = nodes[elems[0][8]][2] - nodes[elems[0][1]][2]
    endz = np.max(nodes[:, 2])
    nelem_z = (endz - startz) / zside

    for nt in range(0, len(darcy_node_loc)):
        coord_node = darcy_node_loc[nt][0:3]
        xelem_num = np.floor((coord_node[0] - startx) / xside)
        yelem_num = np.floor((coord_node[1] - starty) / yside)
        zelem_num = np.floor((coord_node[2] - startz) / zside)
        nelem = int(xelem_num + (yelem_num) * nelem_x + (zelem_num) * (nelem_x * nelem_y))
        darcy_node_elems[nt] = nelem  # record what element the darcy node is in
    
    return {'darcy_node_elems': darcy_node_elems}



def mapping_darcy_sampl_gr(darcy_node_elems, non_empty_rects,conductivity,porosity):

     mapped_con_por=np.zeros((len(darcy_node_elems),3))
     for el in range (0,len(darcy_node_elems)):
         mapped_con_por[el,0]=el+1
         mapped_con_por[el,1]=conductivity[np.where(non_empty_rects==darcy_node_elems[el])]
         mapped_con_por[el,2]=porosity[np.where(non_empty_rects==darcy_node_elems[el])]
     #print mapped_con_por
     return mapped_con_por



def iron_darcy(mapped_con_por,node_coordinates,element_array,node_array,element_nodes_array,volume, thickness, ellipticity):

   # Set problem parameters
   (coordinateSystemUserNumber,
       regionUserNumber,
       basisUserNumber,
       generatedMeshUserNumber,
       meshUserNumber,
       decompositionUserNumber,
       geometricFieldUserNumber,
       equationsSetFieldUserNumber,
       dependentFieldUserNumber,
       materialFieldUserNumber,
       equationsSetUserNumber,
       problemUserNumber) = range(1,13)

   iron.DiagnosticsSetOn(iron.DiagnosticTypes.IN, [1, 2, 3, 4, 5], "Diagnostics",
                        ["DOMAIN_MAPPINGS_LOCAL_FROM_GLOBAL_CALCULATE"])

   numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
   computationalNodeNumber = iron.ComputationalNodeNumberGet()
   initial_conc= 0.0
   number_of_dimensions = 3
   number_of_mesh_components = 1
   total_number_of_elements = len(element_array)
   total_number_of_nodes = len(node_array)
   mesh_component_number = 1
   nodes_per_elem = 4  # for a tet mesh

   # Create a RC coordinate system
   coordinateSystem = iron.CoordinateSystem()
   coordinateSystem.CreateStart(coordinateSystemUserNumber)
   coordinateSystem.dimension = 3
   coordinateSystem.CreateFinish()

   # Create a region
   region = iron.Region()
   region.CreateStart(regionUserNumber,iron.WorldRegion)
   region.label = "DarcyRegion"
   region.coordinateSystem = coordinateSystem
   region.CreateFinish()

   #  Create a tri-linear simplex basis
   basis = iron.Basis()
   basis.CreateStart(basisUserNumber)
   basis.TypeSet(iron.BasisTypes.SIMPLEX)
   basis.numberOfXi = 3
   basis.interpolationXi = [iron.BasisInterpolationSpecifications.LINEAR_SIMPLEX]*3
   basis.CreateFinish()

   # Start the creation of the imported mesh in the region
   mesh = iron.Mesh()
   mesh.CreateStart(meshUserNumber, region, number_of_dimensions)
   mesh.NumberOfComponentsSet(number_of_mesh_components)
   mesh.NumberOfElementsSet(total_number_of_elements)

   # Define nodes for the mesh
   nodes = iron.Nodes()
   nodes.CreateStart(region, total_number_of_nodes)

   # Refers to nodes by their user number as described in the original mesh
   nodes.UserNumbersAllSet(node_array)
   nodes.CreateFinish()

   elements = iron.MeshElements()
   elements.CreateStart(mesh, mesh_component_number, basis)

   # Set the nodes pertaining to each element
   for idx, elem_num in enumerate(element_array):
       elements.NodesSet(idx + 1, element_nodes_array[idx])

   # Refers to elements by their user number as described in the original mesh
   elements.UserNumbersAllSet(element_array)
   elements.CreateFinish()

   mesh.CreateFinish()

   # Create a decomposition for the mesh
   decomposition = iron.Decomposition()
   decomposition.CreateStart(decompositionUserNumber, mesh)
   decomposition.type = iron.DecompositionTypes.CALCULATED
   decomposition.numberOfDomains = numberOfComputationalNodes
   decomposition.CreateFinish()

   # Create a field for the geometry
   geometricField = iron.Field()
   geometricField.CreateStart(geometricFieldUserNumber, region)
   geometricField.meshDecomposition = decomposition
   geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 1, 1)
   geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 2, 1)
   geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 3, 1)
   geometricField.CreateFinish()

   # Update the geometric field parameters
   geometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)

   for idx, node_num in enumerate(node_array):
       [x, y, z] = node_coordinates[idx]

       geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1,
                                            int(node_num), 1, x)
       geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1,
                                            int(node_num), 2, y)
       geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1,
                                            int(node_num), 3, z)

   geometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)

   # Create standard Darcy equations set
   equationsSetField = iron.Field()
   equationsSet = iron.EquationsSet()
   equationsSetSpecification = [iron.EquationsSetClasses.FLUID_MECHANICS,
           iron.EquationsSetTypes.DARCY_EQUATION,
           iron.EquationsSetSubtypes.STANDARD_DARCY]
   equationsSet.CreateStart(equationsSetUserNumber,region,geometricField,
           equationsSetSpecification,equationsSetFieldUserNumber,equationsSetField)
   equationsSet.CreateFinish()

   # Create dependent field
   dependentField = iron.Field()
   equationsSet.DependentCreateStart(dependentFieldUserNumber,dependentField)
   dependentField.VariableLabelSet(iron.FieldVariableTypes.U, "Dependent")
   dependentField.DOFOrderTypeSet(iron.FieldVariableTypes.U,iron.FieldDOFOrderTypes.SEPARATED)
   dependentField.DOFOrderTypeSet(iron.FieldVariableTypes.DELUDELN,iron.FieldDOFOrderTypes.SEPARATED)
   equationsSet.DependentCreateFinish()

   # Create material field
   materialField = iron.Field()
   equationsSet.MaterialsCreateStart(materialFieldUserNumber,materialField)
   materialField.VariableLabelSet(iron.FieldVariableTypes.U, "Material")
   equationsSet.MaterialsCreateFinish()
   '''
   kFileRaw = open('kData.txt', 'r')
   kFile = kFileRaw.readlines()
   startLines = range(0,len(kFile))

   for i in range(len(kFile)):
     kFile[i] = kFile[i].split()

   kList = []

   for i in startLines:
     nodeK = []
     nodeK.append(float(kFile[i][0]))
     nodeK.append((float(kFile[i][4])/0.003)) #perm over vis = k/0.003
     nodeK.append(float(kFile[i][5])) #porosity
     kList.append(nodeK)
     print nodeK
   kFileRaw.close()
   '''
   porosity=0.3
   perm_over_vis =0.8
   mapped_con_por
   materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,porosity)
   materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2,perm_over_vis)
   
   for knode in range(0,len( mapped_con_por)):
     materialField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, int(mapped_con_por[knode,0]),1, mapped_con_por[knode,2]) # set porosity
     materialField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, int(mapped_con_por[knode,0]),2, mapped_con_por[knode,1]/0.003) # set perm_over_vis

   # Initialise dependent field

   dependentField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,initial_conc)

   # Create equations
   equations = iron.Equations()
   equationsSet.EquationsCreateStart(equations)
   equations.sparsityType = iron.EquationsSparsityTypes.SPARSE
   equations.outputType = iron.EquationsOutputTypes.NONE
   equationsSet.EquationsCreateFinish()

   # Create Darcy equation problem
   problem = iron.Problem()
   problemSpecification = [iron.ProblemClasses.FLUID_MECHANICS,
           iron.ProblemTypes.DARCY_EQUATION,
           iron.ProblemSubtypes.STANDARD_DARCY]
   problem.CreateStart(problemUserNumber, problemSpecification)
   problem.CreateFinish()

   # Create control loops
   problem.ControlLoopCreateStart()
   problem.ControlLoopCreateFinish()

   # Create problem solver
   solver = iron.Solver()
   problem.SolversCreateStart()
   problem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,solver)
   solver.outputType = iron.SolverOutputTypes.SOLVER
   solver.linearType = iron.LinearSolverTypes.ITERATIVE
   solver.linearIterativeAbsoluteTolerance = 1.0E-12
   solver.linearIterativeRelativeTolerance = 1.0E-12
   problem.SolversCreateFinish()

   ## Create solver equations and add equations set to solver equations
   solver = iron.Solver()
   solverEquations = iron.SolverEquations()
   problem.SolverEquationsCreateStart()
   problem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,solver)
   solver.SolverEquationsGet(solverEquations)
   solverEquations.sparsityType = iron.SolverEquationsSparsityTypes.SPARSE
   equationsSetIndex = solverEquations.EquationsSetAdd(equationsSet)
   problem.SolverEquationsCreateFinish()

   # Create boundary conditions
   boundaryConditions = iron.BoundaryConditions()
   solverEquations.BoundaryConditionsCreateStart(boundaryConditions)

   #list all x-y couples in ellipsoid
   xyList = node_coordinates[:,[0,1]]
   #Remove duplicates
   xyListUnique = np.vstack({tuple(row) for row in xyList})
   #Set blood vessels properties:
   arteries = 40
   veins = 40
   BVCount = arteries + veins
   #Randomly generate unique xy positions of blood vessels.
   random.seed(500)
   radii = pg_utilities.calculate_ellipse_radii(volume, thickness, ellipticity)
   z_radius = radii['z_radius']
   x_radius = radii['x_radius']
   y_radius = radii['y_radius']
   bv_y = []
   bv_x = np.array(random.sample(np.linspace(-x_radius,x_radius, 100000), BVCount))
   max_y = np.sqrt(y_radius**2 * (1 - (bv_x**2/x_radius**2)))
   for maxY in max_y:
     bv_y.append(random.choice(np.linspace(-maxY,maxY, 100000)))
   # get bv_x and bv_y in form of xy:
   bv_xy = np.zeros((len(bv_x), 2))
   bv_xy[:,0] = bv_x
   bv_xy[:,1] = bv_y
   bloodVessels = []
   nodeSpacing=1.5
   for i in range(0,len(bv_xy)): #for each blood vessel:
     #Cycle through xyListUnique to find closest nodes.
     validNodes = []
     for nodeX in xyListUnique:
       xfound = nodeX[0] < (bv_xy[i][0] + nodeSpacing*2) and nodeX[0] > (bv_xy[i][0] - nodeSpacing*2)
       yfound = nodeX[1] < (bv_xy[i][1] + nodeSpacing*2) and nodeX[1] > (bv_xy[i][1] - nodeSpacing*2)
       if  xfound and yfound:
         validNodes.append(nodeX)

     #Now find closest:
     distance = []
     for nodeColumn in validNodes:
       distance.append(np.sqrt((bv_xy[i][0] - nodeColumn[0])**2 + (bv_xy[i][1] - nodeColumn[1])**2))
     closestNode = validNodes[np.argmin(distance)]
     bloodVessels.append(closestNode)
     #search node list for highest z closest node and set to pressure.
     xyNodes = np.where(np.all(xyList == closestNode, axis = 1))[0]
     xyNodes = [x+1 for x in xyNodes]

     if i < arteries:
       artNode = xyNodes[len(xyNodes) - 1]
       boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,artNode,4,iron.BoundaryConditionsTypes.FIXED,100.0)
     else:
       veinNode = xyNodes[len(xyNodes) - 1]
       boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,veinNode,4,iron.BoundaryConditionsTypes.FIXED,0.0)

   tol = 0.00001
   #Set all other surface nodes to 0 velocity:
   for pair in xyListUnique:
     bvNode = False
     for bv in bloodVessels:
       if ((pair[0] - bv[0]) < tol)  & ((pair[1] - bv[1]) < tol):
         bvNode = True

     #Set v to 0
     #Find top and bottom node:
     xyNodes = np.where(np.all(xyList == pair, axis = 1))[0]
     xyNodes = [x+1 for x in xyNodes]
     topNode = xyNodes[len(xyNodes) - 1]
     bottomNode = xyNodes[0]

     #Set bottom node to 0 velocity
     boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,bottomNode,3,iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
     boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,bottomNode,2,iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
     boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,bottomNode,1,iron.BoundaryConditionsTypes.FIXED_WALL,0.0)

     #If NOT thin and NOT BV - set top node to 0 velocity
     if (topNode != bottomNode) & (bvNode == False):
       boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,topNode,1,iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
       boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,topNode,2,iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
       boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,topNode,3,iron.BoundaryConditionsTypes.FIXED_WALL,0.0)


   solverEquations.BoundaryConditionsCreateFinish()
   # Solve the problem
   problem.Solve()
   print "I come up to here"

# Export results
#baseName = "Darcy"
#dataFormat = "PLAIN_TEXT"
#fml = iron.FieldMLIO()
#fml.OutputCreate(mesh, "", baseName, dataFormat)
#fml.OutputAddFieldNoType(baseName+".geometric", dataFormat, geometricField,
#    iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
#fml.OutputAddFieldNoType(baseName+".phi", dataFormat, dependentField,
#    iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
#fml.OutputWrite("DarcyResults.xml")
#fml.Finalise()

   # Export results
   fields = iron.Fields()
   fields.CreateRegion(region)
   fields.NodesExport("DarcyResults","FORTRAN")
   #fields.ElementsExport("DarcyResults","FORTRAN")
   fields.Finalise()


   iron.Finalise()






