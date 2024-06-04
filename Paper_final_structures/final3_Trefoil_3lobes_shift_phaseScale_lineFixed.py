from functions_based import *
import my_functions.plotings as pl
import knots_ML.data_generation as dg

import numpy as np
from scipy.special import assoc_laguerre
import my_functions.functions_general as fg
import math
import scipy.io


# def scale_along_axis(mesh, axis, s):
#     # Normalize the axis vector
#     axis = np.array(axis) / np.linalg.norm(axis)
#
#     # Reshape the axis to (3, 1, 1, 1) so it can be broadcasted properly
#     axis = axis[:, np.newaxis, np.newaxis, np.newaxis]
#
#     # Compute the dot product of each point in the mesh with the axis
#     dot_product = np.sum(mesh * axis, axis=0)
#
#     # Project each point onto the axis
#     proj = dot_product * axis
#
#     # Compute the component of each point orthogonal to the axis
#     orth = mesh - proj
#
#     # Scale the projection
#     proj_scaled = s * proj
#
#     # Compute the new points
#     new_mesh = proj_scaled + orth
#
#     return new_mesh


def scale_along_axis(xy, axis, scale):
    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)

    # Project all points onto the axis vector
    proj = np.dot(xy, axis)

    # Calculate the vector component that is orthogonal to the axis
    ortho = xy - np.outer(proj, axis)

    # Scale the projections
    proj_scaled = proj * scale

    # Calculate the final points
    xy_final = np.outer(proj_scaled, axis) + ortho

    return xy_final[:, 0], xy_final[:, 1]

def reflect_along_axis(xy, axis, reflection_point):
    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)

    # Translate points such that reflection point is at origin
    xy_translated = xy - reflection_point

    # Project all points onto the axis vector
    proj = np.dot(xy_translated, axis)

    # Calculate the vector component that is orthogonal to the axis
    ortho = xy_translated - np.outer(proj, axis)

    # Reflect the projections (change their sign)
    proj_reflected = -proj

    # Calculate the new points
    xy_final = np.outer(proj_reflected, axis) + ortho

    # Translate points back
    xy_final += reflection_point

    return xy_final[:, 0], xy_final[:, 1]

def gauss_z(x, y, z, width):
    if width is None:
        return 1
    else:
        return np.exp(-(z) ** 2 / width ** 2)  # + np.exp(-(z + 0.5) ** 2 / width ** 2)


def LG_simple_xyz(x, y, z, l=1, p=0, width=1, k0=1, x0=0, y0=0, z0=0, width_gauss=None):
    """
    Classic LG beam
    :param l: azimuthal index
    :param p: radial index
    :param width: beam waste
    :param k0: wave number
    :param x0: center of the beam in x
    :param y0: center of the beam in y
    :return: complex field
    """

    def laguerre_polynomial(x, l, p):
        return assoc_laguerre(x, p, l)

    x = x - x0
    y = y - y0
    z = z - z0
    zR = k0 * width ** 2

    E = (np.sqrt(math.factorial(p) / (np.pi * math.factorial(np.abs(l) + p)))
         * fg.rho(x, y) ** np.abs(l) * np.exp(1j * l * fg.phi(x, y))
         / (width ** (np.abs(l) + 1) * (1 + 1j * z / zR) ** (np.abs(l) + 1))
         * ((1 - 1j * z / zR) / (1 + 1j * z / zR)) ** p
         * np.exp(-fg.rho(x, y) ** 2 / (2 * width ** 2 * (1 + 1j * z / zR)))
         * laguerre_polynomial(fg.rho(x, y) ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)), np.abs(l), p)
         )
    if width_gauss is not None:
        return E * gauss_z(x=x, y=y, z=z, width=width_gauss)
    else:
        return E


def LG_spectre_coeff_3D(field, l, p, xM=(-1, 1), yM=(-1, 1), zM=(-1, 1), width=1., k0=1., mesh=None,
                        functions=bp.LG_simple):
    """
    Function calculates a single coefficient of LG_l_p in the LG spectrum of the field
    :param field: complex electric field
    :param l: azimuthal index of LG beam
    :param p: radial index of LG beam
    :param xM: x boundaries for an LG beam (if Mesh is None)
    :param yM: y boundaries for an LG beam (if Mesh is None)
    :param width: LG beam width
    :param k0: k0 in LG beam but I believe it doesn't affect anything since we are in z=0
    :param mesh: mesh for LG beam. if None, xM and yM are used
    :return: complex weight of LG_l_p in the spectrum
    """
    if mesh is None:
        shape = np.shape(field)
        mesh = fg.create_mesh_XY(xMinMax=xM, yMinMax=yM, zMinMax=yM, xRes=shape[0], yRes=shape[1], zRes=shape[2])
        dS = ((xM[1] - xM[0]) / (shape[0] - 1)) * ((yM[1] - yM[0]) / (shape[1] - 1))
    else:
        xArray, yArray, zArray = fg.arrays_from_mesh(mesh)
        dS = (xArray[1] - xArray[0]) * (yArray[1] - yArray[0]) * (zArray[1] - zArray[0])
    # print(123, xArray)
    # shape = np.shape(field)
    # xyMesh = fg.create_mesh_XY_old(xMax=xM[1], yMax=yM[1], xRes=shape[0], yRes=shape[1], xMin=xM[0], yMin=yM[0])
    LGlp = functions(*mesh, l=l, p=p, width=width, k0=k0)
    # plt.imshow(LGlp)
    # plt.show()
    # print('hi')
    return np.sum(field * np.conj(LGlp)) * dS


def LG_spectrum(beam, l=(-3, 3), p=(0, 5), xM=(-1, 1), yM=(-1, 1), width=1., k0=1., mesh=None, plot=True,
                functions=bp.LG_simple, **kwargs):
    """

    :param beam:
    :param l:
    :param p:
    :param xM:
    :param yM:
    :param width:
    :param k0:
    :param mesh:
    :param plot:
    :return:
    """
    print('hi')
    l1, l2 = l
    p1, p2 = p
    spectrum = np.zeros((l2 - l1 + 1, p2 - p1 + 1), dtype=complex)
    # spectrumReal = []
    # modes = []
    for l in np.arange(l1, l2 + 1):
        for p in np.arange(p1, p2 + 1):
            value = LG_spectre_coeff_3D(beam, l=l, p=p, xM=xM, yM=yM, width=width, k0=k0, mesh=mesh,
                                        functions=functions, **kwargs)
            # print(l, p, ': ', value, np.abs(value))
            spectrum[l - l1, p] = value
    # if np.abs(value) > 0.5:
    # spectrumReal.append(value)
    # modes.append((l, p))
    # print(modes)
    if plot:
        import matplotlib.pyplot as plt
        pl.plot_2D(np.abs(spectrum), x=np.arange(l1 - 0.5, l2 + 1 + 0.5), y=np.arange(p1 - 0.5, p2 + 1 + 0.5),
                   interpolation='none', grid=True, xname='l', yname='p', show=False)
        plt.yticks(np.arange(p1, p2 + 1))
        plt.xticks(np.arange(l1, l2 + 1))
        plt.show()
    return spectrum


def u(x, y, z):
    numerator = x ** 2 + y ** 2 + z ** 2 - 1 + 2j * z
    denominator = x ** 2 + y ** 2 + z ** 2 + 1
    return numerator / denominator


def v(x, y, z):
    numerator = 2 * (x + 1j * y)
    denominator = x ** 2 + y ** 2 + z ** 2 + 1
    return numerator / denominator


"""used modules"""
plot_milnor_field = 0
plot_milnor_lines = 0
plot_braids = 0
real_field = 1
plot_real_field = 1
plot_real_lines = 0
modes_cutoff = 0.01
modes_cutoff = 0.0001
modes_cutoff = 0.03
modes_cutoff = 0.02
modes_cutoff = 0.05
# modes_cutoff = 0.10


shift = 0.325  # 0.2
shift = 0.325
# shift = 0.325  # using for a big 1
# shift = 0.3  # using for a big 1
# shift = 0.15  # 1 line each side boundary condition
# shift = 0.225   # 2 lines each side boundary condition
# shift = 0.3  # 2 lines each side boundary condition
# shift = 0.35  # 2 lines each side boundary condition
# shift = 0
scale = 1.0
# scale = 1.3
# scale = 1.0
line_number = 0
# line_number = 1
# line_number = 2
# line_number = 90
# line_number = 90
"""beam parameters"""
w = 1.3
# w = 1.2
# w = 1.15
w = 1.1
w = 0.9  # for a big one
w = 0.90
name_f = f'w={w}_shift={shift}_coff={modes_cutoff}_scale10'

cmapF = 'hsv'
cmapE = 'hot'
w_real = 1.6  # 1.3 for 3d 1.6 for 2d.

# print(np.radians(3.45))
# print((6 - 6 * np.cos(np.pi / 6)), (6 - 6 * np.cos(np.pi / 6)) / 6, np.arcsin((6 - 6 * np.cos(np.pi / 6)) * np.cos(np.pi / 6) / 6))
# exit()

# A, B, C = -1 * np.pi,  1 * np.pi, 0.25 * np.pi
# A, B, C = 0 - 0.0 * np.pi,  0 + 0.1 * np.pi, 0.5 * np.pi
# C_lobe1, C_lobe2, C_lobe3 = 0.25 * np.pi, 0.0 * np.pi, 0.0 * np.pi

y_scale1 = 1 / scale
y_scale2 = 1 / scale
y_scale3 = 1 / scale
C_lobe1, C_lobe2, C_lobe3 = np.pi * 0 / 6 * 3 / 2, 0.0 * np.pi, 0.0 * np.pi

BETTA = 0 / 3
ALPHA = BETTA * (2 / 3) / 1
l1, l2, l3 = 1, 1, 1
# x_shift1, x_shift2, x_shift3 = +shift * l1, -shift * np.sin(np.pi / 6) * l2, -shift * np.sin(np.pi / 6) * l3
# y_shift1, y_shift2, y_shift3 = -0.0 * l1, +shift * np.cos(np.pi / 6) * l2, -shift * np.cos(np.pi / 6) * l3
z_shift1, z_shift2, z_shift3 = 0, 0, 0
x_lim_3D, y_lim_3D, z_lim_3D = (-5.5, 5.5), (-5.5, 5.5), (-1, 1)
x_lim_3D, y_lim_3D, z_lim_3D = (-6, 6), (-6, 6), (-1, 1)
x_lim_3D, y_lim_3D, z_lim_3D = (-4*1.6, 4*1.6), (-4*1.6, 4*1.6), (-0.75, 0.75)
# x_lim_3D, y_lim_3D, z_lim_3D = (-2.5*1.6, 2.5*1.6), (-2.5*1.6, 2.5*1.6), (-1.5, 1.5)

# res_x_3D_k, res_y_3D_k, res_z_3D_k = 120, 120, 120
# res_x_3D_k, res_y_3D_k, res_z_3D_k = 60, 60, 60
res_x_3D_k, res_y_3D_k, res_z_3D_k = 80, 80, 80
res_x_3D_k, res_y_3D_k, res_z_3D_k = 80, 80, 80
x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-3.0, 3.0), (-3.0, 3.0), (-1.2, 1.2)
x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-2.2, 2.2), (-2.2, 2.2), (-0.75, 0.75)
x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-2.4, 2.4), (-2.4, 2.4), (-0.75, 0.75)  # scale
x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-2.5, 2.5), (-2.5, 2.5), (-0.75, 0.75)
x_lim_3D_k, y_lim_3D_k, z_lim_3D_k = (-2.2*1.6, 2.2*1.6), (-2.2*1.6, 2.2*1.6), (-1.5, 1.5)# scale
# x_lim_3D, y_lim_3D, z_lim_3D = (-2.5, 2.5), (-2.5, 2.5), (-1, 1)
res_x_3D, res_y_3D, res_z_3D = 100, 100, 100
res_x_3D, res_y_3D, res_z_3D = 551, 551, 3
x_shift1 = +shift * np.cos(ALPHA) * l1
y_shift1 = -shift * np.sin(ALPHA) * l1
# x_shift2 = -shift * np.sin(np.pi / 6 - ALPHA) * l2
# y_shift2 = +shift * np.cos(np.pi / 6 - ALPHA) * l2
# x_shift3 = -shift * np.sin(np.pi / 6 + ALPHA) * l3
# y_shift3 = -shift * np.cos(np.pi / 6 + ALPHA) * l3
x_shift2 = -shift * np.sin(np.pi / 6 + ALPHA) * l2
y_shift2 = +shift * np.cos(np.pi / 6 + ALPHA) * l2
x_shift3 = -shift * np.sin(np.pi / 6 - ALPHA) * l3
y_shift3 = -shift * np.cos(np.pi / 6 - ALPHA) * l3


def braid(x, y, z, angle=0, pow_cos=1, pow_sin=1, theta=0, a_cos=1, a_sin=1,
          braids_modification=None):
    def cos_v(x, y, z, power=1):
        return (v(x, y, z) ** power + np.conj(v(x, y, z)) ** power) / 2

    def sin_v(x, y, z, power=1):
        return (v(x, y, z) ** power - np.conj(v(x, y, z)) ** power) / 2j

    x_new = np.array(x)
    y_new = np.array(y)
    z_new = np.array(z)
    angle_3D = np.ones(np.shape(z)) * angle
    a_cos_3D = np.ones(np.shape(z)) * a_cos
    a_sin_3D = np.ones(np.shape(z)) * a_sin
    phase = np.angle(x_new + 1j * y_new)
    # plot_field(phase)
    # plt.show()

    if 1:
        ANGLE = 30
        # ANGLE = 25
        alpha3_1 = ANGLE
        alpha3_2 = 0
        alpha3_3 = 120

        alpha1 = ANGLE
        alpha2 = 0
        alpha3 = 240
        alpha1 = ANGLE
        alpha2 = 0
        # alpha3 = 0
        if braids_modification == 0:

            print('Braid 1:\nLobe 1')
            A, B = -2 / 3 * np.pi, 2 / 3 * np.pi

            phase_mask = (phase > A) & (phase < B)

            # angle_3D[phase_mask] += C_lobe1
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(ANGLE), np.radians(0), np.radians(0))
            )
            x_new[phase_mask] += x_shift1
            y_new[phase_mask] += y_shift1
            z_new[phase_mask] += z_shift1
            A2, B2 = 1 / 3 * np.pi, -1 / 3 * np.pi
            phase_mask2 = (phase < A2) & (phase >= B2)

            angle_rad = 0
            axis = np.array([np.cos(angle_rad), np.sin(angle_rad)])[::-1]
            coords = np.dstack((x_new[phase_mask2], y_new[phase_mask2]))[0]
            x_new[phase_mask2], y_new[phase_mask2] = scale_along_axis(coords, axis, y_scale1)

            print('Lobe 2')
            # A, B = -2 / 3 * np.pi, 2 / 3 * np.pi
            phase_mask = (phase <= A) & (phase >= -np.pi * 1)
            phase_2 = np.angle(np.array(x) + 1j * (np.array(y) - y_shift2))
            phase_mask3 = (
                ((phase_2 >= np.pi - np.radians(3.805)) & (phase_2 < np.pi * 1)) +
                ((x_new <= 0) & (y_new <= y_shift2) & (0 <= y_new))
            )
            angle_3D[phase_mask3] += np.pi
            # phase_mask3 = (
            #         (phase > np.pi * 0.95) & (phase <= np.pi * 1) |
            #         (np.array(x) < 0) & (np.array(y) < y_shift2) & (0 <= np.array(y))
            # )
            # angle_3D[phase_mask3] += np.pi
            # angle_3D[phase_mask] += C_lobe2
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(0), np.radians(0), np.radians(alpha3))
            )
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(alpha1), np.radians(alpha2), np.radians(0))
            )
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(0), np.radians(0), np.radians(-alpha3))
            )
            # phase_new = np.angle(x_new + 1j * y_new)
            # phase_mask3 = (
            #         (phase > np.pi * 0.95) & (phase <= np.pi * 1) |
            #         (x_new < 0) & (y_new < y_shift2) & (0 <= y_new)
            # )
            # angle_3D[phase_mask3] += np.pi
            x_new[phase_mask] += x_shift2
            y_new[phase_mask] += y_shift2
            z_new[phase_mask] += z_shift2


            # if line_number > 0:
            #     angle_3D[:res_x_3D // 2, res_y_3D // 2 - 1, :] += np.pi
            # if line_number > 1:
            #     angle_3D[:res_x_3D // 2, res_y_3D // 2 - 2, :] += np.pi
            # if line_number > 2:
            #     angle_3D[:res_x_3D // 2, res_y_3D // 2 - 3, :] += np.pi
            # if line_number > 3:
            #     angle_3D[:res_x_3D // 2, res_y_3D // 2 - 4, :] += np.pi
            # if line_number > 4:
            #     angle_3D[:res_x_3D // 2, res_y_3D // 2 - 5, :] += np.pi

            print('Lobe 3')
            # A, B = -2 / 3 * np.pi, 2 / 3 * np.pi
            phase_mask = (phase >= B) & (phase <= np.pi * 1)

            angle_3D[phase_mask] += C_lobe3
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(0), np.radians(0), np.radians(alpha3_3))
            )
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(alpha3_1), np.radians(alpha3_2), np.radians(0))
            )
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(0), np.radians(0), np.radians(-alpha3_3))
            )
            phase_3 = np.angle(np.array(x) + 1j * (np.array(y) - y_shift3))
            phase_mask3 = (
                    ((phase_3 <= -np.pi + np.radians(3.805)) & (phase_3 >= -np.pi * 1))
                    + ((np.array(x) <= 0) & (np.array(y) > y_shift3) & (np.array(y) <= 0))
            )
            angle_3D[phase_mask3] = np.pi
            x_new[phase_mask] += x_shift3
            y_new[phase_mask] += y_shift3
            z_new[phase_mask] += z_shift3
            # if line_number > 0:
            #     angle_3D[:res_x_3D // 2, res_y_3D // 2 + 0, :] += np.pi
            # if line_number > 1:
            #     angle_3D[:res_x_3D // 2, res_y_3D // 2 + 1, :] += np.pi
            # if line_number > 2:
            #     angle_3D[:res_x_3D // 2, res_y_3D // 2 + 2, :] += np.pi
            # if line_number > 3:
            #     angle_3D[:res_x_3D // 2, res_y_3D // 2 + 3, :] += np.pi
            # if line_number > 4:
            #     angle_3D[:res_x_3D // 2, res_y_3D // 2 + 4, :] += np.pi
            # if line_number > 5:
            #     angle_3D[:res_x_3D // 2, res_y_3D // 2 + 5, :] += np.pi

        elif braids_modification == 1:
            # Lobe 2
            print('Braid 2\nLobe 2')
            # A, B = 0, 0
            phase_mask = (phase >= 0) & (phase <= np.pi * 1)
            # angle_3D[phase_mask] += C_lobe2
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(0), np.radians(0), np.radians(alpha3))
            )
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(alpha1), np.radians(alpha2), np.radians(0))
            )
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(0), np.radians(0), np.radians(-alpha3))
            )
            x_new[phase_mask] += x_shift2
            y_new[phase_mask] += y_shift2
            z_new[phase_mask] += z_shift2

            A2, B2 = np.pi, np.pi / 3
            phase_mask2 = (phase <= A2) & (phase >= B2)
            angle_rad = (np.pi / 2 + np.pi / 6 -np.pi / 2)
            axis = np.array([np.cos(angle_rad), np.sin(angle_rad)])

            coords = np.dstack((x_new[phase_mask2], y_new[phase_mask2]))[0]
            x_new[phase_mask2], y_new[phase_mask2] = scale_along_axis(coords, axis, y_scale2)

            # Lobe 3
            print('Lobe 3')
            # A, B = 0, 0
            # phase_mask = (phase <= A)
            phase_mask = (phase < 0) & (phase > -np.pi * 1)

            angle_3D[phase_mask] += C_lobe3
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(0), np.radians(0), np.radians(alpha3_3))
            )
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(alpha3_1), np.radians(alpha3_2), np.radians(0))
            )
            x_new[phase_mask], y_new[phase_mask], z_new[phase_mask] = (
                rotate_meshgrid(x_new[phase_mask], y_new[phase_mask], z_new[phase_mask],
                                np.radians(0), np.radians(0), np.radians(-alpha3_3))
            )
            x_new[phase_mask] += x_shift3
            y_new[phase_mask] += y_shift3
            z_new[phase_mask] += z_shift3
            A2, B2 = -np.pi / 3, -np.pi
            phase_mask2 = (phase <= A2) & (phase >= B2)
            angle_rad = (-np.pi / 2 - np.pi / 6 + np.pi/2)
            axis = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            coords = np.dstack((x_new[phase_mask2], y_new[phase_mask2]))[0]
            x_new[phase_mask2], y_new[phase_mask2] = scale_along_axis(coords, axis, y_scale3)

    if braids_modification in [braids_modification]:  # [braids_modification]:  # [0, 1] for turning off the braids
        return u(x_new, y_new, z_new) * np.exp(1j * theta) - (
                cos_v(x_new, y_new, z_new, pow_cos) / a_cos_3D + 1j
                * sin_v(x_new, y_new, z_new, pow_sin) / a_sin_3D) * np.exp(1j * angle_3D)
    else:
        return 1


# cos_v(x, y, z, pow_cos) / a_cos + 1j * sin_v(x, y, z, pow_sin) / a_sin) * np.exp(1j * angle_3D)


def braid_before_trans(x, y, z, angle=0, pow_cos=1, pow_sin=1, theta=0, a_cos=1, a_sin=1,
                       braids_modification=None):
    def cos_v(x, y, z, power=1):
        return (np.exp(1j * z) ** power + np.conj(np.exp(1j * z)) ** power) / 2

    def sin_v(x, y, z, power=1):
        return (np.exp(1j * z) ** power - np.conj(np.exp(1j * z)) ** power) / 2j

    # if z>0:
    #     angle +=np.py//4
    angle_3D = np.ones(np.shape(z)) * angle
    a_cos_3D = np.ones(np.shape(z)) * a_cos
    a_sin_3D = np.ones(np.shape(z)) * a_sin
    _, _, zAr = fg.arrays_from_mesh((x, y, z))
    # shape = np.shape(angle_3D)
    # angle_3D[:, :, shape[2] // 2 :] *= 1.3
    x_new = np.array(x)
    y_new = np.array(y)
    z_new = np.array(z)

    if 1:
        if braids_modification == 0:
            # plot_field(x_new)
            # plt.show()
            print('Braid 1:\nLobe 1')
            A, B = -3 / 3 * np.pi, 1 / 3 * np.pi
            # A, B = -2 / 3 * np.pi, 2 / 3 * np.pi
            # braid_scale = 1.0  # 1.2
            # x_scale = 1  # 1/1.2
            # x_shift = 0  # 0.5

            # phase_mask = (phase >= A) & (phase <= B)
            phase_mask = (zAr > A) & (zAr < B)
            # indexes = np.where(phase_mask)
            # angle_3D[indexes] += C_lobe1
            angle_3D[phase_mask] += C_lobe1
            # x_new[phase_mask] *= x_scale
            # angle_3D[phase_mask] += C_lobe2
            x_new[phase_mask] += x_shift1
            y_new[phase_mask] += y_shift1
            z_new[phase_mask] += z_shift1
            # a_cos_3D = np.ones(np.shape(z)) * a_cos
            # a_sin_3D = np.ones(np.shape(z)) * a_sin
            # a_cos_3D[phase_mask] *= braid_scale
            # a_sin_3D[phase_mask] *= braid_scale
            # Lobe 2

            print('Lobe 2')
            A, B = 1 / 3 * np.pi, 3 / 3 * np.pi
            phase_mask = (zAr > A) & (zAr <= B)
            angle_3D[phase_mask] += C_lobe2
            x_new[phase_mask] += x_shift2
            y_new[phase_mask] += y_shift2
            z_new[phase_mask] += z_shift2

        # z_new[phase_mask] += z_shift2
        # Lobe 3

        # print('Lobe 3')
        # A, B = -2 / 3 * np.pi, 2 / 3 * np.pi
        # phase_mask = (phase > B) & (phase < np.pi  * 1)
        # angle_3D[phase_mask] += C_lobe3
        # x_new[phase_mask] += x_shift3
        # y_new[phase_mask] += y_shift3
        # z_new[phase_mask] += z_shift3

        # plot_field(x_new)
        # plt.show()
        elif braids_modification == 1:
            # Lobe 2

            print('Lobe 2')
            A, B = -3 / 3 * np.pi, -1 / 3 * np.pi
            phase_mask = (zAr > A) & (zAr <= B)
            angle_3D[phase_mask] += C_lobe2
            x_new[phase_mask] += x_shift2
            y_new[phase_mask] += y_shift2
            z_new[phase_mask] += z_shift2

            # Lobe 3
            print('Lobe 3')
            A, B = -1 / 3 * np.pi, 3 / 3 * np.pi
            phase_mask = (zAr > A) & (zAr < B)
            angle_3D[phase_mask] += C_lobe3
            x_new[phase_mask] += x_shift3
            y_new[phase_mask] += y_shift3
            z_new[phase_mask] += z_shift3
    # plot_field(x_new)
    # plt.show()
    # exit()

    return (x + 1j * y) * np.exp(1j * theta) - (
            cos_v(x, y, z, pow_cos) / a_cos_3D + 1j * sin_v(x, y, z, pow_sin) / a_sin_3D) * np.exp(1j * angle_3D)


# return (x + 1j * y) * np.exp(1j * theta) - (
#     cos_v(x, y, z, pow_cos) / a_cos + 1j * sin_v(x, y, z, pow_sin) / a_sin) * np.exp(1j * angle)


def field_of_braids_separate_trefoil(mesh_3D, braid_func=braid, scale=None):
    xyz_array = [
        (mesh_3D[0], mesh_3D[1], mesh_3D[2]),
        (mesh_3D[0], mesh_3D[1], mesh_3D[2])
    ]
    # starting angle for each braid
    angle_array = np.array([0, 1. * np.pi]) + BETTA  ##0 * np.pi/3
    print(angle_array)
    # powers in cos in sin
    pow_cos_array = [1.5, 1.5]
    pow_sin_array = [1.5, 1.5]
    # conjugating the braid (in "Milnor" space)
    conj_array = [0, 0]
    # moving x+iy (same as in the paper)
    theta_array = [0.0 * np.pi, 0 * np.pi]
    # braid scaling
    a_cos_array = [1, 1]
    a_sin_array = [1, 1]
    braids_modification = [0, 1]
    # braids_modification = [5, 6]

    if theta_array is None:
        theta_array = [0] * len(angle_array)
    if a_cos_array is None:
        a_cos_array = [1] * len(angle_array)
    if a_sin_array is None:
        a_sin_array = [1] * len(angle_array)

    if braid_func is not braid:
        scale = [0.5, 0.5, 1 * np.pi]
    if scale is not None:
        for i, xyz in enumerate(xyz_array):
            shape = np.shape(xyz)
            xyz_new = np.array(xyz).reshape((3, -1)).T * scale
            xyz_array[i] = tuple(xyz_new.T.reshape(shape))

    ans = 1

    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i],
                                           braids_modification=braids_modification[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i],
                              braids_modification=braids_modification[i])

    return ans



# w = 1.3

# LG spectrum
moments = {'p': (0, 6), 'l': (-5, 5)}
"""mesh parameters"""

x_3D = np.linspace(*x_lim_3D, res_x_3D)
y_3D = np.linspace(*y_lim_3D, res_y_3D)
z_3D = np.linspace(*z_lim_3D, res_z_3D)
mesh_3D = np.meshgrid(x_3D, y_3D, z_3D, indexing='ij')  #

mesh_2D = np.meshgrid(x_3D, y_3D, indexing='ij')  #
phase = np.angle(mesh_3D[0] + 1j * mesh_3D[1])
mesh_2D_xz = np.meshgrid(x_3D, z_3D, indexing='ij')  #
R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
boundary_3D = [[0, 0, 0], [res_x_3D, res_y_3D, res_z_3D]]
boundary_3D_k = [[0, 0, 0], [res_x_3D_k, res_y_3D_k, res_z_3D_k]]
"""creating the field"""
# mesh for each brade (in "Milnor" space)
xyz_array = [
    (mesh_3D[0], mesh_3D[1], mesh_3D[2]),
    (mesh_3D[0], mesh_3D[1], mesh_3D[2])
]
y_ind = res_y_3D // 2 + 0
# starting angle for each braid
angle_array = [0, 1. * np.pi]
# powers in cos in sin
pow_cos_array = [1.5, 1.5]
pow_sin_array = [1.5, 1.5]
# conjugating the braid (in "Milnor" space)
conj_array = [0, 0]
# moving x+iy (same as in the paper)
theta_array = [0.0 * np.pi, 0 * np.pi]
# braid scaling
a_cos_array = [1, 1]
a_sin_array = [1, 1]
field = field_of_braids_separate_trefoil(mesh_3D, braid_func=braid)
"""field transformations"""
# cone transformation
field_milnor = field * (1 + R ** 2) ** 3
field_gauss = field_milnor * bp.LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)
field_norm = dg.normalization_field(field_gauss)
moment0 = moments['l'][0]
values_total = 0
y_value = 0
w_spec = 1.0
width_gauss = 0.75
width_gauss = 1.0
# k_0_spec = 1.6
# field_norm = field_norm * gauss_z(*mesh_3D, width=width_gauss)
# pl.plot_3D_density(np.abs(field_norm))
# plt.show()
if plot_milnor_field:
    plot_field(field_norm)
    plt.show()
# plot_field(field_norm[:, :, res_z_3D//2 - 10])
# plt.show()
# plot_field(field_norm[:, :, res_z_3D//2 - 20])
# plt.show()
# plot_field(field_norm[:, :, res_z_3D//2 - 30])
# plt.show()
# plot_field(field_norm[:, :, res_z_3D//2 - 40])
# plt.show()
# plot_field(field_norm[:, :, res_z_3D // 2 - 50])
# plt.show()
# plot_field(field_norm[:, :, res_z_3D // 2 - 60])
# plt.show()
# plot_field(field_norm[:, y_ind - 2, :])
# plt.show()
# plot_field(field_norm[:, y_ind - 1, :])
# plt.show()
# plot_field(field_norm[:, y_ind, :])
# plt.show()
# plot_field(field_norm[:, y_ind + 1, :])
# plt.show()
# plot_field(field_norm[:, y_ind + 2, :])
# plt.show()
if plot_milnor_lines:
    _, dots_init = sing.get_singularities(np.angle(field_norm), axesAll=False, returnDict=True)
    dp.plotDots(dots_init, boundary_3D, color='blue', show=True, size=7)
    plt.show()

if plot_braids:
    braid = field_of_braids_separate_trefoil(mesh_3D, braid_func=braid_before_trans)
    plot_field(braid[:, :, 0])
    plt.show()
    plot_field(braid)
    plt.show()
    plot_field(braid[:, :, -1])
    plt.show()
# _, dots_init = sing.get_singularities(np.angle(braid), axesAll=False, returnDict=True)
# dp.plotDots(dots_init, boundary_3D, color='red', show=True, size=7)
# plt.show()

# building 'LG' field
#################################################################################

# new_function = functools.partial(LG_simple_xz, y=y_value, width_gauss=width_gauss)#, width=w * w_spec)
new_function = functools.partial(LG_simple_xyz, width_gauss=width_gauss)  # , width=w * w_spec)
field_norm = field_norm * gauss_z(*mesh_3D, width=width_gauss)
# field_norm = np.load('trefoil3d.npy') * gauss_z(x=mesh_2D_xz[0], y=0, z=mesh_2D_xz[1], width=width_gauss)
# plot_field(new_function(*mesh_2D_xz, l=1, p=1))
# plot_field(np.load('trefoil3d.npy'))
# plt.show()
# exit()
# plot_field(field_norm[:, y_ind, :])
# plot_field(field_norm)
# plt.show()
# exit()
# values = LG_spectrum(
#     field_norm[:, y_ind, :], **moments, mesh=mesh_2D_xz, plot=True, width=w * w_spec, k0=1,
#     functions=new_function
# )
# !!!!!!!!!!!

##################################################################################################
if real_field:
    values = cbs.LG_spectrum(
        field_norm[:, :, res_z_3D // 2], **moments, mesh=mesh_2D, plot=True, width=w * w_spec, k0=1,
    )
    # values = LG_spectrum(
    #     field_norm[:, :, :], **moments, mesh=mesh_3D, plot=True, width=w * w_spec, k0=k_0_spec,
    #     functions=new_function
    # )
    
    field_new_3D = np.zeros((res_x_3D, res_y_3D, res_z_3D)).astype(np.complex128)
    total = 0
    l_save = []
    p_save = []
    weight_save = []
    
    # for l, p_array in enumerate(values):
    # 	for p, value in enumerate(p_array):
    # 		if abs(value) > 0.01 * abs(values).max():
    # 			total += 1
    # 			l_save.append(l + moment0)
    # 			p_save.append(p)
    # 			weight_save.append(value)
    # 			# weights_important[f'{l + moment0}, {p}'] = value
    # 			field_new_3D += value * bp.LG_simple(*mesh_3D, l=l + moment0, p=p,
    # 			                                     width=w * w_spec, k0=1, x0=0, y0=0, z0=0)
    # weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    # scipy.io.savemat('weights_trefoil_shifted_2_w13.mat', weights_important)
    
    for l, p_array in enumerate(values):
        for p, value in enumerate(p_array):
            if abs(value) > modes_cutoff * abs(values).max():
                total += 1
                l_save.append(l + moment0)
                p_save.append(p)
                weight_save.append(value)
                # weights_important[f'{l + moment0}, {p}'] = value
                field_new_3D += value * bp.LG_simple(*mesh_3D, l=l + moment0, p=p,
                                                     width=1 * w_spec * w_real, k0=1, x0=0, y0=0, z0=0)
    field_new_3D = field_new_3D / np.abs(field_new_3D).max()
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    print(weights_important)

    # scipy.io.savemat('weights_trefoil_moved_to_center_14_scaling.mat', weights_important)
if plot_real_field and real_field:
    plot_field(field_new_3D, titles=('', ''), intensity=False, cmapF=cmapF, cmapE=cmapE, axes=False)

    plt.show()
    # plot_field(field_new_3D[:, y_ind, :])
    # plt.show()

if plot_real_lines and real_field:
    x_3D_k = np.linspace(*x_lim_3D_k, res_x_3D_k)
    y_3D_k = np.linspace(*y_lim_3D_k, res_y_3D_k)
    z_3D_k = np.linspace(*z_lim_3D_k, res_z_3D_k)
    mesh_3D_k = np.meshgrid(x_3D_k, y_3D_k, z_3D_k, indexing='ij')  #
    field_new_3D_k = np.zeros((res_x_3D_k, res_y_3D_k, res_z_3D_k)).astype(np.complex128)
    for l, p_array in enumerate(values):
        for p, value in enumerate(p_array):
            if abs(value) > modes_cutoff * abs(values).max()    and l + moment0<=3 and p<=3:
                field_new_3D_k += value * bp.LG_simple(*mesh_3D_k, l=l + moment0, p=p,
                                                       width=w_real * w_spec, k0=1, x0=0, y0=0, z0=0)
    _, dots_init = sing.get_singularities(np.angle(field_new_3D_k), axesAll=True, returnDict=True)

    dots_init = np.array([[dot[0], res_y_3D_k - dot[1], dot[2]] for dot in dots_init])
    # dots_init = np.array([[dot[0], dot[1], dot[2]] for dot in dots_init])
    dots_z0 = []
    for dot in dots_init:
        if dot[2] == res_z_3D_k // 2:
            dots_z0.append(dot)
    # if len(dots_z0) > 2:
    #     dots_z0 = [dots_z0[0], dots_z0[-1]]
    dots_z0 = np.array(dots_z0)
    dotSize = 12
    fig = dp.plotDots_Hopf(dots_init, boundary_3D_k, color='red', show=False, size=dotSize)
    dp.plotDots_Hopf(dots_z0, boundary_3D_k, color='black', show=False, size=dotSize * 1.75, fig=fig)

    # fig.write_html(f'trefoil_rotated_shifted.html')
    # np.save(f'trefoil_fixed_scaled14'
    #         f'_resX{res_x_3D_k}_resZ{res_z_3D_k}_limX{x_lim_3D_k[1]}_limZ{z_lim_3D_k[1]}', dots_init)
    fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))

    fig.show()
    exit()
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=-2.3, y=-2.3, z=2.3)  # Adjust x, y, and z to set the default angle of view
            )
        )
    )
    fig.write_html(f'trefoil_scaling_{name_f}_t.html')
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=0, y=-4, z=0)  # Adjust x, y, and z to set the default angle of view
            )
        )
    )
    fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    fig.write_html(f'trefoil_scaling_{name_f}_xz.html')
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=-4, y=0, z=0),  # Adjust x, y, and z to set the default angle of view
                up=dict(x=0, y=0, z=1)
            )
        )
    )
    fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))

    fig.write_html(f'trefoil_scaling_{name_f}_yz.html')
    # fig.update_layout(
    #     scene=dict(
    #         camera=dict(
    #             eye=dict(x=0, y=0, z=3),  # Adjust x, y, and z to set the default angle of view
    #             up=dict(x=0, y=1, z=0)
    #         )
    #     )
    # )
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=0, y=0, z=4),  # Adjust x, y, and z to set the default angle of view
                up=dict(x=0, y=1, z=0)
            )
        )
    )
    fig.update_layout(scene=dict(camera=dict(projection=dict(type='orthographic'))))
    fig.write_html(f'trefoil_scaling_{name_f}_xy.html')
###################################################################
