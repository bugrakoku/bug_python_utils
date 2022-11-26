
def GetARoom(nWalls=4, h = 300, Rmin =3000, Rmax = 10000, noiseMultiplier = 0, outlierPercent = 0.0, addBottom = False, addTop = False, randomizeWalls=False, shuffleColumns = False, rotateRoom = False):
    '''
    Generates a room with nWalls walls and height h
    Corners of adjacent walls are first selected by their angles and distance to origin
    Corner distances are in [Rmin, Rmax]
    INPUTS:
    nWalls: number of walls, 4 by default
    h: room height, 300cm by default
    Rmin: Minimum corner distance to origin
    Rmax: Maximum corner distance to origin
    noiseMultiplier: adds uniform noise multiplied with this value to the final data
    outlierPercent: should be in [0-1] range, ideall small :)
        adds outliers to the data where this value is the percentage of outliers to be added
    addBottom: adds the base of the of the room to data, False by default
    addTop: adds ceiling to the room to data, False by default
    randomizeWalls: intersection of walls are not uniformly distributed, slightly perturbed
        if enabled, non-convex rooms for nWalls more than 4 is possible
    shuffleColumns: False by default. 
        Normally, points from wall1, wall2, ..., bottom, top, outliers are grouped in the result
        If set to True, columns are shuffled
    rotateRoom: False by default, if set to True, resulting points will be randomly rotated around X and Y axes
        Note that only the Points values are rotated, Walls or plane coordinates are NOT
        Given that rotation matrices are also returned you can always use them to get back to 
        where Wall and top/bottom panes are originally defined
    RETURNS: 
    Points, NumberOfPointsInEachStructure, WallCorners, BottomPlane, TopPlane, ShuffledCoordinates
    Points: a 3xN matrix where N is the number of points in the point cloud,
        This is the data matrix to be used in clustering data
        Rest of the parameters are for ground truth and algorithm testing purposes
    NumberOfPointsInEachStructure: Points are generated in the following order
        Wall1, Wall2, ...WallLast, Points on the Floor, Points on the Ceiling, Outlier points
        Note that Points on the floor and ceiling as well as the outliers 
        only exits if they are requrested via input parameters
    WallCorners: collection of arrays where each array contain the 3D coordinates of 4 wall corners
    BottomPlane: coordinates of the polygon that defines the bottom plane, i.e. floor
    TopPlane: coordinates of the polygon that defines the top plane, i.e. ceiling
    ShuffledCoordinates: If shuffleColumns are enabled, this vector contains where the original columns ended up
        Using this you can unshuffle the datamatrix
    Rx,Ry: If rotateRoom is set to True, datapoints are rotated by Rx and Ry matrices
        Using these matrices you can get the room back where the floor is on the XY plane
    '''

    def ZeroMean(M):
        '''
        As implied zero means data matrix M which is assumed to be DXN where
        D is the dimension of the data, and there N data points in the columns of M
        Function returns zero meaned copy of M along with the mean vector in case M is to be restored later
        '''
        CM = M.mean(axis=1).reshape((M.shape[0],1)) # get the mean of the data i.e. the center of mass
        return M - CM, CM

    def formWall(a1 = 0, r1 = 300, a2 = pi/2, r2 = 400, h = 250):
        #Generate 4 points where each point
        wall = np.array([])
        #append x,y,z for each corner
        # c1 at 0
        wall = np.array([[r1 * cos(a1), r1 * sin(a1), 0] ]) # get the first corner
        wall = np.vstack((wall, wall[-1].copy())) # copy first as the second
        wall[-1][-1] = h # elevate it above the previous corner
        wall = np.vstack((wall, [r2 * cos(a2), r2 * sin(a2), 0 ])) # get the second corner
        wall = np.vstack((wall, wall[-1].copy())) # copy the previous corner
        wall[-1][-1] = h # elevate it above
        return wall.T # so that coordinates are on the columns

    def PointsOnWall(Wall, nPoints=100):
        '''
        put nPoints on the wall, where wall is a numpy array containing the corner coordinates of a rectangular wall
        by default is it assumed that wall is perpendicular to XY Plane, no checks for other orientations
        returns a 3xnPoints numpy array
        '''
        # first find the centroid of the wall
        W0, CM = ZeroMean(Wall)

        #using the first columns find a basis for plane projection (i.e. a line) on XY plane
        P1 = W0[:2,0].reshape((2,1)) # only get x,y coordinates
        XYpts = P1 @ (np.random.rand(1,nPoints)*2-1) # get the XY coordinates on the line define by P1 within wall limits
        zL = np.max(W0[2,:]) - np.min(W0[2,:]) # get the span of wall along z axis
        zCoord = np.random.rand(1,nPoints) * zL - zL/2 # generate random z coordinates 
        Pts = np.vstack((XYpts,zCoord)) # stack them as 3D coordinates
        Pts += CM # move them back to wall
        # done, I hope
        return Pts

    def PointsOnPlane(Plane, nPoints=100):
        # get the number of triangles on the polygon wrt to the center
        
        nS = Plane.shape[1] # number of pie slices 
        PointsPerSlice = [int(nPoints/nS) if i < nS-1 else int(nPoints/nS)+(nPoints%nS) for i in range(nS)]
        Points = np.random.rand(3,1) # just to init the numpy array
        for i,nPS in enumerate(PointsPerSlice): # for each slice prepeare points
            # get two vectors
            v1 = Plane[:,i].reshape(3,1)
            v2 = Plane[:,(i+1)%nS].reshape(3,1) # when v1 is the last column v2 is the first column, i.e. the last slice
            # now generate random combinations of v1 and v2
            probV1 = np.random.rand(1,nPS)
            probV2 = np.random.rand(1,nPS)
            #newOnes = (v1 @ np.random.rand(1,nPS) + v2 @ np.random.rand(1,nPS))/2
            newOnes = (v1 @ probV1 + v2 @ probV2)/2
            Points = np.hstack((Points, newOnes))
        
        return Points[:,1:] # skip the first dummy one
    
    cAngles = np.arange(0, 2*pi, 2*pi/nWalls) # corner angles at fixed angular distance
    cR = np.repeat(np.random.randint(Rmin,Rmax), nWalls) # all corners are at a fixed distance from the center
    if randomizeWalls: # then slightly play with corners
        cAngles += np.random.randn(*cAngles.shape) * 2 * pi / nWalls / 10 # pie angles are randomly but slightly perturved
        cR = np.random.randint(Rmin, Rmax, cAngles.shape) # each corner is at a random distance

    Wallz = [] # will hold walls as a collection of 3D corners
    Points = np.random.rand(3,1) # just to init the numpy array
    # random number of points on each wall
    minPointCount = 50
    maxPointCount = 200
    PointCount = np.random.randint(minPointCount, maxPointCount, (nWalls,1))

    for i, _ in enumerate(cAngles):
        cWall = formWall(cAngles[i], cR[i], cAngles[(i+1)%nWalls], cR[(i+1)%nWalls], h) # current wall
        Wallz.append(cWall) # add to list of walls
        newOnes = PointsOnWall(cWall, PointCount[i,0])
        Points = np.hstack((Points, newOnes))
    
    Wallz = np.array(Wallz) # convert to numpy array
    BottomPlane = Wallz[:,:,0].T # so that data is on the columns
    TopPlane = Wallz[:,:,0].T # so that data is on the columns
    Points = Points[:,1:] # get rid of the dummy at the origin

    # if bottom needed
    if addBottom:
        numBottomPoints = [np.random.randint(minPointCount, maxPointCount)] # determine number of points
        Points = np.hstack((Points, PointsOnPlane(BottomPlane, numBottomPoints[0]))) # generate points and add to Points
        PointCount = np.vstack((PointCount, numBottomPoints)) # update point count
    # if top is needed
    if addTop:
        numTopPoints = [np.random.randint(minPointCount, maxPointCount)] # determine number of points
        topPoints = PointsOnPlane(TopPlane, numTopPoints[0]) # generate points, but they are on the bottom plane by default
        topPoints[2,:] = h # elevate points to top plane
        Points = np.hstack((Points,topPoints )) # add points to Points
        PointCount = np.vstack((PointCount, numTopPoints)) # update points count

    # if noise it so be added
    Points = Points + np.random.randn(*Points.shape) * noiseMultiplier

    # if there should be outliers
    if outlierPercent > 0.0: # add outliers
        numOutliers = int(np.sum(PointCount) * outlierPercent) # get number of outliers to generate
        rM = np.max(Points[0,:])
        Pind = np.arange(Points.shape[1])
        np.random.shuffle(Pind) # shuffle column indices and select a subset of it to generate outliers
        PindOut = Pind[:numOutliers] # these indices correspond to column to be used in outlier generation
        # generate multipliers that are in 0-0.5 or 1.5-3 range
        OutMultiplier = np.random.rand(numOutliers)*3 # generate random multipliers in [0-3] range
        # that are in 0.5-1.5 range divide them by 3 so that outliers are not close to acutal values more than 50%
        OutMultiplier[np.where(OutMultiplier>0.5) and np.where(OutMultiplier<1.5)] = OutMultiplier[np.where(OutMultiplier>0.5) and np.where(OutMultiplier<1.5)]/3 
        PointOutliers = Points[:,PindOut] @ np.diag(OutMultiplier) 
        Points = np.hstack((Points,PointOutliers ))
        PointCount = np.vstack((PointCount, numOutliers)) # update points count

    # if columns are to be shuffled, we are not using just np.random.suffle since we want to know which columns move to where 
    ShuffledCols = None
    if shuffleColumns: # then shuffle 
        ShuffledCols = np.arange(Points.shape[1]) # get a sorted vector that correspond to current columns
        np.random.shuffle(ShuffledCols) # shuffle column indices
        Points = Points[:,ShuffledCols] # finally shuffle the columns of Points
    
    # if the room is to be rotated
    # by default no rotation
    Rx = np.eye(3)
    Ry = np.eye(3)
    if rotateRoom:
        # randomize Rx, Ry
        ThetaX = np.random.rand()*pi/2 # a maximum of 90 degrees rotation
        ThetaY = np.random.rand()*pi/2 # a maximum of 90 degrees rotation
        # generate rotation matrices
        Rx = np.array([[1,0,0],[0,cos(ThetaX),-sin(ThetaX)],[0,sin(ThetaX),cos(ThetaX)]])
        Ry = np.array([[cos(ThetaY),0,sin(ThetaY)],[0,1,0],[-sin(ThetaY),0, cos(ThetaY)]])
        Points = Ry @ Rx @ Points

    # finally
    return Points, [x[0] for x in PointCount], Wallz, BottomPlane, TopPlane, ShuffledCols, Rx, Ry

