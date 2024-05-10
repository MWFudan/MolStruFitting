import numpy as np
import re
from scipy import optimize
from math import sqrt

mass_table = {
    'H': 1.00782503207,
    'D': 2.0141017778,
    '2H': 2.0141017778,

    'He': 4.0026032541,

    'C': 12.0,
    '13C': 13.0033548378,

    'O': 15.99491461956,
    '16O': 15.99491461956,
    '18O': 17.9991610,

    'F': 18.99840316,

    'P': 30.973761998,

    'S': 31.97207100,
    '34S': 33.96786690,

    'Cl': 34.96885268,
    '35Cl': 34.96885268,
    '37Cl': 36.96590259
}

re_coor = re.compile(r'\s*(\d{0,3}[A-Z][a-z]?)\s+(\S+)\s+(\S+)\s+(\S+)\s*')

def convert_coor_text_to_mass_and_coors(coor_text):
    result = re.findall(re_coor, coor_text)
    if not result:
        return
    mass = []
    coor = []
    for i in result:
        mass.append(mass_table[i[0]])
        coor.append(list(map(float, i[1:])))
    return np.array(mass), np.array(coor).reshape(-1)

def calc_ABC(coor, mass):
    # coor is array,[[][][]]
    coor = coor.reshape(-1, 3)
    mass_sum = np.sum(mass)
    mass_center = np.sum(mass.reshape(-1, 1) * coor, 0) / mass_sum
    coor_centered = coor - mass_center

    Ixx = np.sum(mass * (coor_centered[:, 1] ** 2 + coor_centered[:, 2] ** 2))
    Iyy = np.sum(mass * (coor_centered[:, 0] ** 2 + coor_centered[:, 2] ** 2))
    Izz = np.sum(mass * (coor_centered[:, 0] ** 2 + coor_centered[:, 1] ** 2))
    Ixy = -np.sum(mass * coor_centered[:, 0] * coor_centered[:, 1])
    Ixz = -np.sum(mass * coor_centered[:, 0] * coor_centered[:, 2])
    Iyz = -np.sum(mass * coor_centered[:, 1] * coor_centered[:, 2])

    moment_inertial = np.linalg.eigvals(np.matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]))
    ABC_in_MHz = 505379.6963277365/np.array(moment_inertial)
    # ABC_in_MHz = -np.sort(-ABC_in_MHz)
    sum_ABC = np.sqrt(np.sum(ABC_in_MHz**2))

    return sum_ABC


def coordination_generator(mass, coor, unc_for_coor, error_number=1):
    lines = []
    coor = coor.reshape(-1, 3)
    unc_for_coor = unc_for_coor.reshape(-1, 3)
    for m, c, unc in zip(mass, coor, unc_for_coor):
        coor_txt_with_error_list = []
        for i, j in zip(unc, c):
            zc = int(('%.3e'%i)[-1])
            if error_number == 2:
                txt = str(j)[:zc+(4 if str(j).startswith('-') else 3)] + f"({('%.3e'%i)[0] + ('%.3e'%i)[2]})"
            elif error_number == 1:
                txt = str(j)[:zc + (3 if str(j).startswith('-') else 2)] + f"({('%.3e' % i)[0]})"
            coor_txt_with_error_list.append(txt)
        # lines.append('{}\t{:>10.6f}\t{:>10.6f}\t{:>10.6f}\t{:>10.8f}\t{:>10.8f}\t{:>10.8f}'.format(convert_mass_to_atom_str(m), *c, *unc))
        lines.append(
            '{}\t{:>14}\t{:>14}\t{:>14}'.format(convert_mass_to_atom_str(m), *coor_txt_with_error_list))
    return '\n'.join(lines)
def convert_mass_to_atom_str(mass):
    for key, val in mass_table.items():
        if val == mass:
            return key


def fit_plane(point_cloud):
    """
    input
        point_cloud : list of xyz values　numpy.array
    output
        plane_v : (normal vector of the best fit plane)
        com : center of mass
    """

    com = np.sum(point_cloud, axis=0) / len(point_cloud)
    # calculate the center of mass
    q = point_cloud - com
    # move the com to the origin and translate all the points (use numpy broadcasting)
    Q = np.dot(q.T, q)
    # calculate 3x3 matrix. The inner product returns total sum of 3x3 matrix
    la, vectors = np.linalg.eig(Q)
    # Calculate eigenvalues and eigenvectors
    plane_v = vectors.T[np.argmin(la)]
    # Extract the eigenvector of the minimum eigenvalue

    return plane_v, com



def calc_distance_points(c, number_count):
    assert len(number_count) == 2 and number_count[0] > 0 and number_count[1] > 0
    c1 = c[:number_count[0] * 3]
    c2 = c[number_count[0] * 3:]
    c1 = np.mean(np.array(c1).reshape((-1, 3)), axis=0)
    c2 = np.mean(np.array(c2).reshape((-1, 3)), axis=0)
    return np.linalg.norm(c1-c2)

def calc_distance_point_to_plane(c, number_count):
    assert len(number_count) == 2
    if number_count[0] > 0 and number_count[1] < 0:
        coor_avg = c[:number_count[0]*3]
        coor_for_plane = c[number_count[0]*3:]
    elif number_count[0] < 0 and number_count[1] > 0:
        coor_avg = c[-number_count[0]*3:]
        coor_for_plane = c[:-number_count[0]*3]
    else:
        raise
    coor_points = np.array(coor_avg).reshape((-1, 3))
    points_center = np.mean(coor_points, axis=0)
    coor_for_plane = np.array(coor_for_plane).reshape((-1, 3))
    plane_v, com = fit_plane(coor_for_plane)

    # 勾股定理
    # coor_for_plane_center = np.mean(coor_for_plane, axis=0)
    # d1 = np.linalg.norm(coor_for_plane_center-points_center)
    # d2 = abs(np.dot(points_center - com, plane_v))
    # return np.sqrt(d1**2-d2**2)

    return abs(np.dot(points_center - com, plane_v))

def calc_angle_points(c, number_count):
    assert len(number_count) == 3 and number_count[0] > 0 and number_count[1] > 0 and number_count[2] > 0
    c1 = c[:number_count[0] * 3]
    c2 = c[number_count[0] * 3:(number_count[0]+number_count[1]) * 3]
    c3 = c[(number_count[0] + number_count[1]) * 3:]
    c1 = np.mean(np.array(c1).reshape((-1, 3)), axis=0)
    c2 = np.mean(np.array(c2).reshape((-1, 3)), axis=0)
    c3 = np.mean(np.array(c3).reshape((-1, 3)), axis=0)
    v1 = c1 - c2
    v2 = c3 - c2
    a = np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return a / np.pi * 180

def calc_angle_planes(c, number_count):
    assert len(number_count) == 2 and number_count[0] < 0 and number_count[1] < 0
    coor_1 = c[:-number_count[0] * 3]
    coor_2 = c[-number_count[0] * 3:]
    coor_1 = np.array(coor_1).reshape((-1, 3))
    coor_2 = np.array(coor_2).reshape((-1, 3))
    v1, com_1 = fit_plane(coor_1)
    v2, com_2 = fit_plane(coor_2)
    a = np.arccos(np.abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return a / np.pi * 180


# coor_txt = '''C	-1.92531087	 1.90142089	-0.02744284
# C	-2.52058397	 1.06724311	 0.91366471
# C	-1.23181478	 1.35082707	-1.10099109
# H	-3.06264381	 1.49054612	 1.74863139
# H	-0.76113040	 1.99453882	-1.83156779
# C	-2.42596718	-0.31527306	 0.78893720
# C	-1.12899247	-0.02845886	-1.24121481
# H	-2.87568553	-0.98550761	 1.50789138
# H	-0.58487481	-0.47925036	-2.05822764
# C	-1.72847886	-0.83180682	-0.28859717
# F	-1.62263169	-2.17246488	-0.40891205
# H	-2.00134444	 2.97519982	 0.07501169
# C	 1.73108475	-1.45557537	 0.47435540
# C	 2.44026902	-0.88652371	-0.54786692
# S	 2.52720609	 0.81759198	-0.37683798
# C	 1.60797920	 0.78170743	 1.06915706
# C	 1.25185447	-0.49570158	 1.40391713
# H	 1.54979196	-2.51706772	 0.55177747
# H	 2.91357186	-1.37487711	-1.38328213
# H	 1.36972388	 1.69618949	 1.58564851
# H	 0.66135947	-0.73696724	 2.27472727
# '''
coor_txt = '''
C	-2.31596695	-1.46604938	-0.76122832
C	-1.78264462	-1.68776894	 0.50328333
C	-2.42668009	-0.16836827	-1.24840673
H	-2.64118056	-2.30200209	-1.36667125
H	-1.68621011	-2.69670489	 0.88322609
C	-1.36100006	-0.62065531	 1.28661308
H	-2.83490480	 0.00948530	-2.23491549
C	-2.01065650	 0.91022211	-0.47731515
H	-0.93247723	-0.76712778	 2.26808742
C	-1.48721234	 0.65803779	 0.77726821
H	-2.07512673	 1.92732861	-0.83829372
F	-1.07929019	 1.70228500	 1.53308962
C	 2.31475797	-1.46571957	 0.76180697
C	 2.42324247	-0.16822436	 1.24992885
C	 1.78466122	-1.68736558	-0.50409450
H	 2.63902981	-2.30175448	 1.36763964
H	 2.82945811	 0.00937177	 2.23731028
C	 2.00817301	 0.91038078	 0.47833555
H	 1.69015845	-2.69616608	-0.88481223
C	 1.36410359	-0.62028878	-1.28799828
H	 2.07104964	 1.92760293	 0.83927182
C	 1.48794210	 0.65821109	-0.77757613
H	 0.93748914	-0.76673355	-2.27028644
F	 1.08024281	 1.70261565	-1.53350875
'''
m, c = convert_coor_text_to_mass_and_coors(coor_txt)
eps = np.sqrt(np.finfo(float).eps)
derivative = optimize.approx_fprime(c, calc_ABC, eps, m)

# 用三个转动常数，不求和，但计算有问题
# uncertainty = np.array((0.108, 0.123, 0.036))
# c_uncertainty_contribution = derivative * uncertainty[:, np.newaxis]
# c_uncertainty = np.sqrt(np.sum(c_uncertainty_contribution**2, axis=0))

# uncertainty = 0.16759773268
uncertainty = 0.033120990
unc_for_coor = abs(uncertainty / derivative)
coor_in_line = coordination_generator(m, c, unc_for_coor).split('\n')
for i, coor_text_in_line in enumerate(coor_in_line):
    print(i, '\t', coor_text_in_line)
# print(coordination_generator(m, c, unc_for_coor))


# for thiophene-PhF rs
# c = np.array([
#  -1.942,1.889, -0.17,
#  -2.537,1.045, 0.923,
#  -1.269,1.318, -1.114,
#  -2.460,0      , 0.81,
#  -1.325,0      , -1.313,
#  -1.763,-0.78, -0.30,
#  1.758,-1.407, 0.54,
#  2.447,-0.823, -0.613,
# 2.5193,0.811, -0.377,
#  1.660,0.74, 1.07,
#   1.29,-0.41, 1.42,
# ])
#
# unc_for_coor = np.array([
#     0.002, 0.002, 0.02,
#     0.001, 0.003, 0.004,
# 0.008, 0.008, 0.009,
# 0.004, 0, 0.01,
# 0.008, 0, 0.008,
# 0.006, 0.01, 0.03,
# 0.006, 0.007, 0.02,
# 0.001, 0.004, 0.006,
# 0.0006, 0.001, 0.004,
# 0.006, 0.01, 0.01,
# 0.02, 0.08, 0.02,
# ])

# for PhF dimer rs
# c = np.array([
#     -2.30956,     -1.46948000,   -0.757920000000,
#     -1.77205,     -1.70120000,        0.457600000000,
#     -2.41709,     -0.13589000,   -1.254320000000,
#     -1.34773,     -0.65457000,        1.263940000000,
#     -1.99470,          0.91371,  -0.473480000000,
#     -1.48305,          0.68638,       0.735450000000,
#     2.309560,    -1.469480000,       0.757920000000,
#     1.772050,    -1.701200000,  -0.457600000000,
#     2.417090,    -0.135890000,       1.254320000000,
#     1.347730,    -0.654570000,  -1.263940000000,
#     1.994700,         0.913710,       0.473480000000,
#     1.483050,         0.686380,  -0.735450000000,
# ])
#
# unc_for_coor = np.array([
#     7E-4, 1E-3, 2E-3,
#     9E-4, 9E-4, 3E-3,
#     6E-4, 1E-2, 1E-3,
#     1E-3, 2E-3, 1E-3,
#     8E-3, 1E-3, 3E-3,
#     1E-3, 2E-3, 2E-3,
#     7E-4, 1E-3, 2E-3,
#     9E-4, 9E-4, 3E-3,
#     6E-4, 1E-2, 1E-3,
#     1E-3, 2E-3, 1E-3,
#     8E-3, 1E-3, 3E-3,
#     1E-3, 2E-3, 2E-3,
# ])

while True:
    face_num = 0
    number_count = []
    measure_txt = input('measure:')
    if measure_txt == 'q':
        break
    atom_num = measure_txt.split('-')
    coor = []
    ucoor = []
    for i in atom_num:
        if len(ns := i.split(',')) != 1:
            if ns[0][0] == 'f':
                face_num += 1
                ns[0] = ns[0][1:]
                for atom_idx in map(int, ns):
                    coor.extend(c[atom_idx * 3: (atom_idx * 3 + 3)])
                    ucoor.extend(unc_for_coor[atom_idx * 3: (atom_idx * 3 + 3)])
                number_count.append(-len(ns))
            else:
                # center of coordination
                for atom_idx in map(int, ns):
                    coor.extend(c[atom_idx * 3: (atom_idx * 3 + 3)])
                    ucoor.extend(unc_for_coor[atom_idx * 3: (atom_idx * 3 + 3)])
                number_count.append(len(ns))
        else:
            i = int(i)
            coor.extend(c[i * 3: (i * 3 + 3)])
            ucoor.extend(unc_for_coor[i * 3: (i * 3 + 3)])
            number_count.append(1)
    if len(atom_num) == 2:
        if face_num == 0:
            derivative = optimize.approx_fprime(coor, calc_distance_points, eps, number_count)
            result = calc_distance_points(coor, number_count)
        elif face_num == 1:
            derivative = optimize.approx_fprime(coor, calc_distance_point_to_plane, eps, number_count)
            result = calc_distance_point_to_plane(coor, number_count)
        elif face_num == 2:
            derivative = optimize.approx_fprime(coor, calc_angle_planes, eps, number_count)
            result = calc_angle_planes(coor, number_count)
        else:
            raise Exception()
    elif len(atom_num) == 3:
        derivative = optimize.approx_fprime(coor, calc_angle_points, eps, number_count)
        result = calc_angle_points(coor, number_count)
    else:
        raise Exception()
    ucoor = np.array(ucoor)
    e = np.sqrt(np.sum((ucoor * derivative) ** 2))
    print(coor, ucoor)
    print(result, e)