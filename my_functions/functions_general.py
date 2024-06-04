"""
this module includes all the general functions used in another modules
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from scipy import integrate
import scipy.io as sio


def rho(*r):
    """
    return abs of the vector r
    :param r: [x1, x2, x3...]
    :return: |r|
    """
    ans = 0
    for x in r:
        ans += x ** 2
    return np.sqrt(ans)


def phi(x, y):
    """
    angle phi in the plane
    """
    return np.angle(x + 1j * y)


def dots_move_center(dots):
    """
    moving dots to the center of the object
    """
    center = np.sum(dots, axis=0) / len(dots)
    return dots - center


def distance_between_points(point1, point2):
    """
    distance between 2 points in any dimensions
    :param point1: [x1, ...]
    :param point2: [x2, ...]
    :return: geometrical distance
    """
    deltas = np.array(point1) - np.array(point2)
    return rho(*deltas)


def create_mesh_XYZ(xMax, yMax, zMax, xRes=40, yRes=40, zRes=40,
                    xMin=None, yMin=None, zMin=None, indexing='ij', random=(None, None, None), **kwargs):
    """
    creating the mesh using np.meshgrid
    :param xMax: [xMin, xMax] are the boundaries for the meshgrid along x
    :param yMax: [yMin, yMax] are the boundaries for the meshgrid along y
    :param zMax: [zMin, zMax] are the boundaries for the meshgrid along z
    :param xRes: resolution along x
    :param yRes: resolution along y
    :param zRes: resolution along z
    :param xMin: xMin=xMax by default.
    :param yMin: yMin=yMax by default.
    :param zMin: zMin=zMax by default.
    :param indexing: ij is the classic matrix (0,0) left top
    :return: mesh
    """
    if xMin is None:
        xMin = -xMax
    if yMin is None:
        yMin = -yMax
    if zMin is None:
        zMin = -zMax
    if random[0] is None:
        xArray = np.linspace(xMin, xMax, xRes)
    else:
        xArray = np.sort((xMax + xMin) / 2 + (np.random.rand(xRes) - 0.5) * (xMax - xMin))
        # xArray = ((xMax + xMin) / 2 + (np.random.rand(xRes) - 0.5) * (xMax - xMin))

    if random[1] is None:
        yArray = np.linspace(yMin, yMax, yRes)
    else:
        yArray = np.sort((yMax + yMin) / 2 + (np.random.rand(yRes) - 0.5) * (yMax - yMin))
        # yArray = ((yMax + yMin) / 2 + (np.random.rand(yRes) - 0.5) * (yMax - yMin))
    if random[2] is None:
        zArray = np.linspace(zMin, zMax, zRes)
    else:
        zArray = np.sort((zMax + zMin) / 2 + (np.random.rand(zRes) - 0.5) * (zMax - zMin))
        # zArray = ((zMax + zMin) / 2 + (np.random.rand(zRes) - 0.5) * (zMax - zMin))
    return np.meshgrid(xArray, yArray, zArray, indexing=indexing, **kwargs)


def create_mesh_XY_old(xMax, yMax, xRes=50, yRes=50,
                       xMin=None, yMin=None, indexing='ij', **kwargs):
    """
    creating the mesh using np.meshgrid
    :param xMax: [xMin, xMax] are the boundaries for the meshgrid along x
    :param yMax: [yMin, yMax] are the boundaries for the meshgrid along y
    :param xRes: resolution along x
    :param yRes: resolution along y
    :param xMin: xMin=xMax by default.
    :param yMin: yMin=yMax by default.
    :param indexing: ij is the classic matrix (0,0) left top
    :return: mesh
    """
    if xMin is None:
        xMin = -xMax
    if yMin is None:
        yMin = -yMax
    xArray = np.linspace(xMin, xMax, xRes)
    yArray = np.linspace(yMin, yMax, yRes)
    return np.meshgrid(xArray, yArray, indexing=indexing, **kwargs)


def create_mesh_XY(xMinMax=None, yMinMax=None, xRes=50, yRes=50,
                   indexing='ij', **kwargs):
    """
    creating the mesh using np.meshgrid
    :param xMinMax: [xMin, xMax] are the boundaries for the meshgrid along x
    :param yMinMax: [yMin, yMax] are the boundaries for the meshgrid along y
    :param xRes: resolution along x
    :param yRes: resolution along y
    :param indexing: ij is the classic matrix (0,0) left top
    :return: mesh
    """
    if xMinMax is None:
        xMinMax = (0 - xRes // 2, 0 + xRes // 2)
    if yMinMax is None:
        yMinMax = (0 - yRes // 2, 0 + yRes // 2)
    xArray = np.linspace(*xMinMax, xRes)
    yArray = np.linspace(*yMinMax, yRes)
    return np.meshgrid(xArray, yArray, indexing=indexing, **kwargs)


def interpolation_real(field, xArray=None, yArray=None, **kwargs):
    """
    Interpolation of any real 2d matrix into the function
    :param field: initial Real 2D array
    :param xArray: x interval (range)
    :param yArray: y interval (range)
    :param kwargs: extra parameters for CloughTocher2DInterpolator
    :return: CloughTocher2DInterpolator return
    """
    xResolution, yResolution = np.shape(field)
    if xArray is None:
        xArray = list(range(xResolution))
    if yArray is None:
        yArray = list(range(yResolution))
    xArrayFull = np.zeros(xResolution * yResolution)
    yArrayFull = np.zeros(xResolution * yResolution)
    fArray1D = np.zeros(xResolution * yResolution)
    for i in range(xResolution * yResolution):
        xArrayFull[i] = xArray[i // yResolution]
        yArrayFull[i] = yArray[i % yResolution]
        fArray1D[i] = field[i // yResolution, i % yResolution]
    return CloughTocher2DInterpolator(list(zip(xArrayFull, yArrayFull)), fArray1D, **kwargs)


# function interpolate complex 2D array of any data into the function(x, y)
def interpolation_complex(field, xArray=None, yArray=None, mesh=None, fill_value=False):
    """
    function interpolate complex 2D array of any data into the function(x, y)
    :param field: initial complex 2D array
    :param xArray: x interval (range)
    :param yArray: y interval (range)
    :return: Real CloughTocher2DInterpolator, Imag CloughTocher2DInterpolator
    """
    if mesh is not None:
        xArray, yArray = arrays_from_mesh(mesh)
    fieldReal = np.real(field)
    fieldImag = np.imag(field)
    real = interpolation_real(fieldReal, xArray, yArray, fill_value=fill_value)
    imag = interpolation_real(fieldImag, xArray, yArray, fill_value=fill_value)

    def f(x, y):
        return real(x, y) + 1j * imag(x, y)

    return f
    # return interpolation_real(fieldReal, xArray, yArray), interpolation_real(fieldImag, xArray, yArray)


def integral_of_function_1D(integrandFunc, x1, x2, epsabs=1.e-5, maxp1=50, limit=50, **kwargs):
    """
    scipy.integrate can only work with real numbers so this function splits the integrand to imaginary and real
    parts and integrates the separately, then combine the answers together
    :param integrandFunc: integrand
    :param x1: lower limit
    :param x2: upper limit
    :return: integral value, (real error, imag error)
    """

    def real_f(x):
        return np.real(integrandFunc(x))

    def imag_f(x):
        return np.imag(integrandFunc(x))

    real_integral = integrate.quad(real_f, x1, x2, epsabs=epsabs, maxp1=maxp1, limit=limit, **kwargs)
    imag_integral = integrate.quad(imag_f, x1, x2, epsabs=epsabs, maxp1=maxp1, limit=limit, **kwargs)
    return real_integral[0] + 1j * imag_integral[0], (real_integral[1:], imag_integral[1:])


def arrays_from_mesh(mesh, indexing='ij'):
    """
    Functions returns the tuple of x1Array, x2Array... of the mesh
    :param mesh: no-sparse mesh, for 3D: [3][Nx, Ny, Nz]
    :return: for 3D: xArray, yArray, zArray
    """
    xList = []
    if indexing == 'ij':
        for i, m in enumerate(mesh):
            row = [0] * len(np.shape(m))
            row[i] = slice(None, None)
            xList.append(m[tuple(row)])
    else:
        if len(np.shape(mesh[0])) == 2:
            for i, m in enumerate(mesh):
                row = [0] * len(np.shape(m))
                row[len(np.shape(m)) - 1 - i] = slice(None, None)
                xList.append(m[tuple(row)])
        elif len(np.shape(mesh[0])) == 3:
            indexing = [1, 0, 2]
            for i, m in enumerate(mesh):
                row = [0] * len(np.shape(m))
                row[indexing[i]] = slice(None, None)
                xList.append(m[tuple(row)])
        else:
            print("'xy' cannot be recreated for 4+ dimensions")

    xTuple = tuple(xList)
    return xTuple


def reading_file_mat(fileName, fieldToRead="p_charges", printV=False):
    """
    function read the mat file and conver 1 of its fields into numpy array
    :param fileName:
    :param fieldToRead: which field to conver (require the name)
    :param printV: if you don't know the name, set it True
    """
    matFile = sio.loadmat(fileName, appendmat=False)
    if fieldToRead not in matFile:
        printV = True
    if printV:
        print(matFile)
        exit()
    return np.array(matFile[fieldToRead])


def dots3D_rescale(dots, mesh):
    """
    rescale dots from [3, 5, 7] into [x[3], y[5], z[7]]
    :param dots: [[nx,ny,nz]...]
    :return: [[x,y,z]...]
    """
    xyz = arrays_from_mesh(mesh)
    dotsScaled = [[xyz[0][x], xyz[1][y], xyz[2][z]] for x, y, z in dots]
    return np.array(dotsScaled)


def random_list(values, diapason, diapason_complex=None):
    """
    Function returns values + random * diapason
    :param values: we are changing this values
    :param diapason: to a random value from [value - diap, value + diap]
    :return: new modified values
    """
    import random
    if diapason_complex is not None:
        answer = [x + random.uniform(-d, +d) + 1j * random.uniform(-dC, +dC) for x, d, dC
                  in zip(values, diapason, diapason_complex)]
    else:
        answer = [x + random.uniform(-d, +d) for x, d in zip(values, diapason)]
    return answer


##############################################


def propagator_split_step_3D(E, dz=1, xArray=None, yArray=None, zSteps=1, n0=1, k0=1):
    if xArray is None:
        xArray = np.array(range(np.shape(E)[0]))
    if yArray is None:
        yArray = np.array(range(np.shape(E)[1]))
    xResolution, yResolution = len(xArray), len(yArray)
    zResolution = zSteps + 1
    intervalX = xArray[-1] - xArray[0]
    intervalY = yArray[-1] - yArray[0]

    # xyMesh = np.array(np.meshgrid(xArray, yArray, indexing='ij'))
    if xResolution // 2 == 1:
        kxArray = np.linspace(-1. * np.pi * (xResolution - 2) / intervalX,
                              1. * np.pi * (xResolution - 2) / intervalX, xResolution)
        kyArray = np.linspace(-1. * np.pi * (yResolution - 2) / intervalY,
                              1. * np.pi * (yResolution - 2) / intervalY, yResolution)
    else:
        kxArray = np.linspace(-1. * np.pi * (xResolution - 0) / intervalX,
                              1. * np.pi * (xResolution - 2) / intervalX, xResolution)
        kyArray = np.linspace(-1. * np.pi * (yResolution - 0) / intervalY,
                              1. * np.pi * (yResolution - 2) / intervalY, yResolution)

    KxyMesh = np.array(np.meshgrid(kxArray, kyArray, indexing='ij'))

    def nonlinearity_spec(E):
        return dz * 0

    # works fine!
    def linear_step(field):
        temporaryField = np.fft.fftshift(np.fft.fftn(field))
        temporaryField = (temporaryField *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[0] ** 2) *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[1] ** 2))  # something here in /2
        return np.fft.ifftn(np.fft.ifftshift(temporaryField))

    fieldReturn = np.zeros((xResolution, yResolution, zResolution), dtype=complex)
    fieldReturn[:, :, 0] = E
    for k in range(1, zResolution):
        fieldReturn[:, :, k] = linear_step(fieldReturn[:, :, k - 1])
        fieldReturn[:, :, k] = fieldReturn[:, :, k] * np.exp(nonlinearity_spec(fieldReturn[:, :, k]))

    return fieldReturn


def propagator_split_step_3D_linear(E, dz=1, xArray=None, yArray=None, zSteps=1, n0=1, k0=1):
    if xArray is None:
        xArray = np.array(range(np.shape(E)[0]))
    if yArray is None:
        yArray = np.array(range(np.shape(E)[1]))
    xResolution, yResolution = len(xArray), len(yArray)
    zResolution = zSteps + 1
    intervalX = xArray[-1] - xArray[0]
    intervalY = yArray[-1] - yArray[0]
    if xResolution // 2 == 1:
        kxArray = np.linspace(-1. * np.pi * (xResolution - 2) / intervalX,
                              1. * np.pi * (xResolution - 2) / intervalX, xResolution)
        kyArray = np.linspace(-1. * np.pi * (yResolution - 2) / intervalY,
                              1. * np.pi * (yResolution - 2) / intervalY, yResolution)
    else:
        kxArray = np.linspace(-1. * np.pi * (xResolution - 0) / intervalX,
                              1. * np.pi * (xResolution - 2) / intervalX, xResolution)
        kyArray = np.linspace(-1. * np.pi * (yResolution - 0) / intervalY,
                              1. * np.pi * (yResolution - 2) / intervalY, yResolution)

    KxyMesh = np.array(np.meshgrid(kxArray, kyArray, indexing='ij'))
    fieldReturn = np.zeros((xResolution, yResolution, zResolution), dtype=complex)
    fieldReturn[:, :, 0] = np.fft.fftshift(np.fft.fftn(E))
    for k in range(1, zResolution):
        fieldReturn[:, :, k] = (fieldReturn[:, :, k - 1] *
                                np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[0] ** 2) *
                                np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[1] ** 2))
        fieldReturn[:, :, k - 1] = np.fft.ifftn(np.fft.ifftshift(fieldReturn[:, :, k - 1]))
    fieldReturn[:, :, -1] = np.fft.ifftn(np.fft.ifftshift(fieldReturn[:, :, -1]))
    return fieldReturn


def one_plane_propagator(fieldPlane, dz, stepsNumber, n0=1, k0=1):  # , shapeWrong=False
    # if shapeWrong is not False:
    #     if shapeWrong is True:
    #         print(f'using the middle plane in one_plane_propagator (shapeWrong = True)')
    #         fieldPlane = fieldPlane[:, :, np.shape(fieldPlane)[2] // 2]
    #     else:
    #         fieldPlane = fieldPlane[:, :, np.shape(fieldPlane)[2] // 2 + shapeWrong]
    fieldPropMinus = propagator_split_step_3D(fieldPlane, dz=-dz, zSteps=stepsNumber, n0=n0, k0=k0)

    fieldPropPLus = propagator_split_step_3D(fieldPlane, dz=dz, zSteps=stepsNumber, n0=n0, k0=k0)
    fieldPropTotal = np.concatenate((np.flip(fieldPropMinus, axis=2), fieldPropPLus[:, :, 1:-1]), axis=2)
    return fieldPropTotal


def cut_filter(E, radiusPix=1, circle=True, phaseOnly=False):
    ans = np.copy(E)
    xCenter, yCenter = np.shape(ans)[0] // 2, np.shape(ans)[0] // 2
    if circle:
        for i in range(np.shape(ans)[0]):
            for j in range(np.shape(ans)[1]):
                if np.sqrt((xCenter - i) ** 2 + (yCenter - j) ** 2) > radiusPix:
                    ans[i, j] = 0
    else:
        if phaseOnly:
            zeros = np.abs(np.copy(ans))
            zeros = zeros.astype(complex, copy=False)
        else:
            zeros = np.zeros(np.shape(ans), dtype=complex)
        zeros[xCenter - radiusPix:xCenter + radiusPix + 1, yCenter - radiusPix:yCenter + radiusPix + 1] \
            = ans[xCenter - radiusPix:xCenter + radiusPix + 1, yCenter - radiusPix:yCenter + radiusPix + 1]
        ans = zeros
    return ans


if __name__ == '__main__':
    import my_functions.beams_and_pulses as bp
    import my_functions.plotings as pl

    xMinMax, yMinMax = 3, 3
    xRes = yRes = 50
    xyMesh = create_mesh_XY_old(xMinMax, yMinMax, xRes, yRes)
    beam = bp.LG_simple(*xyMesh, l=2) + bp.LG_simple(*xyMesh, l=1)
    # fieldFunction[0] + 1j * fieldFunction[1]
    xArray = np.linspace(-xMinMax, xMinMax, xRes)
    yArray = np.linspace(-yMinMax, yMinMax, yRes)
    beamInterpolated = interpolation_complex(beam, xArray, yArray)
