import numpy as np
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from filter_coor_file_with_atom_distance import verify_coor_text
from scipy.spatial.transform import Rotation
from scipy import optimize
from scipy.stats import qmc
from tqdm import tqdm

# http://www.nist.gov/pml/data/comp.cfm
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
re_double_end_line = re.compile(r'\n\s*\n')
re_file_split = re.compile(r'\s{2}\d{1,3}\n')  # crest
# re_file_split = re_double_end_line

process_num = 15
error_percent_discard_threshold = 0.1
ftol = 1E-6  # ftol
jac = '2-point'
rot_method = 'euler'  # euler, quat
not_rot_flag = False
rot_to_trans_scale = 1
algorithm = 'trf'
max_iter_num = 200

class Molecule:
    # must translate first then rotate
    @staticmethod
    def from_text(text, is_active=True):
        mass, coor = text_converter(text)
        return Molecule(mass, coor, 0, is_active)

    # def __init__(self, mass, coor, center_idx, active_flag):
    #     self.mass = mass
    #     self.coor = self.coor0 = coor
    #     self.center_idx = center_idx
    #     self.v0 = coor - coor[center_idx]  # vector from center to other atom, initial value
    #     self.active_flag = active_flag
    #
    # def translate(self, x):
    #     self.coor = self.coor0 + x
    #
    # def rotate(self, q):
    #     if len(q) == 4:
    #         r = Rotation.from_quat(q)
    #     else:
    #         r = Rotation.from_euler('xyz', q)
    #
    #     new_v = r.apply(self.v0)
    #     center_coor = self.coor[self.center_idx]
    #     self.coor = new_v + center_coor
    def __init__(self, mass, coor, _, active_flag):
        self.mass = mass
        self.coor = self.coor0 = coor
        mass_sum = np.sum(mass)
        self.mass_center = self.mass_center0 = np.sum(mass.reshape(-1, 1) * coor, 0) / mass_sum
        self.v0 = coor - self.mass_center0
        self.active_flag = active_flag

    def translate(self, x):
        self.coor = self.coor0 + x
        self.mass_center = self.mass_center0 + x

    def rotate(self, q):
        if len(q) == 4:
            r = Rotation.from_quat(q)
        else:
            r = Rotation.from_euler('xyz', q)

        new_v = r.apply(self.v0)
        self.coor = new_v + self.mass_center


def get_mass_and_coor_vector(motion_inf_list, molecules, uncertainty_analysis_flag=False):
    mass = np.array([])
    coor = np.empty([0, 3])
    rot_param_num = 3 if len(motion_inf_list) % 6 == 0 else 4
    n = 0
    for molecule in molecules:
        if molecule.active_flag:
            motion_trans = motion_inf_list[n:n + 3]
            if not not_rot_flag:
                motion_rot = motion_inf_list[n + 3:n + rot_param_num + 3]
                n += rot_param_num + 3
                molecule.translate(motion_trans)
                molecule.rotate(motion_rot)
            else:
                n += 3
                molecule.translate(motion_trans)

        mass = np.append(mass, molecule.mass)
        coor = np.append(coor, molecule.coor, axis=0)
    principle_axis_coor = convert_to_main_axis(mass, coor)
    if uncertainty_analysis_flag:
        return principle_axis_coor.reshape(-1)
    return mass, principle_axis_coor


def optimize_coor(coor_text, target_ABC, k, isotope_target_ABC=None):
    # coor_text: molecule should be separated by two \n
    # X ...
    # X ...
    #
    # X ...

    # isotope_target_ABC: [(atom index, isotope type, A, B, C), ...]
    # first molecule is not active
    molecules = re.split(re_double_end_line, coor_text)
    molecule_list = []
    flag_first_molecule = True
    for molecule in molecules:
        if not molecule:
            continue

        molecule_list.append(Molecule.from_text(molecule, is_active=not flag_first_molecule))
        if flag_first_molecule:
            flag_first_molecule = False

    # can also use [0, 0, 0, 0, 0, 0, 1] as quat, seems no diff
    if rot_method == 'quat':
        initial_input = [0, 0, 0, 0, 0, 0, 1] * (len(molecule_list) - 1)
    elif rot_method == 'euler':
        if not_rot_flag:
            initial_input = [0, 0, 0] * (len(molecule_list) - 1)
            xscale = [1, 1, 1] * (len(molecule_list) - 1)
        else:
            initial_input = [0, 0, 0, 0, 0, 0] * (len(molecule_list) - 1)
            xscale = [1, 1, 1, rot_to_trans_scale, rot_to_trans_scale, rot_to_trans_scale] * (len(molecule_list) - 1)
            xscale = np.array(xscale)
    else:
        raise ValueError('bad rot method')
    # least square
    result = optimize.least_squares(calc_error, initial_input, args=(target_ABC, molecule_list, isotope_target_ABC),
                                    method=algorithm, jac=jac, ftol=ftol, max_nfev=max_iter_num*len(initial_input), x_scale=xscale)

    # not good
    # result = optimize.minimize(calc_error, initial_input, args=(target_ABC, molecule_list), method='L-BFGS-B', jac='3-point')

    # not good
    # result = optimize.minimize(calc_error, initial_input, args=(target_ABC, molecule_list), method='TNC',
    #                            jac='3-point')

    # not good
    # result = optimize.minimize(calc_error, initial_input, args=(target_ABC, molecule_list), method='trust-constr',
    #                            jac='3-point')

    bounds = [(-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)] * len(molecule_list)
    # simplicial homology global optimization, very slow
    # result = optimize.shgo(lambda x, y=target_ABC, z=molecule_list: calc_error(x, y, z), bounds, options={'ftol': 1.0E-4, 'maxtime': 60})

    # direct, strange
    # result = optimize.direct(calc_error, bounds, args=(target_ABC, molecule_list), maxiter=10000)

    # not good
    # result = optimize.dual_annealing(calc_error, bounds, args=(target_ABC, molecule_list), maxiter=10000)

    opt_coor = coordination_generator(result.x, molecule_list)
    # msg = '' if result.success else result.message
    msg = result.message
    return opt_coor, msg, k



def optimize_one_done(future):
    global sep_idx
    result, msg, k = future.result()
    error = target_ABC - calc_ABC(*text_converter(result))
    p_bar.update(1)
    if any(np.abs(error / target_ABC) > error_percent_discard_threshold) or not verify_coor_text(result, sid=sep_idx):
        # print(error, 'discard')
        return
    sum_result[k] = [error, result, msg]
    # print(k, error)
    # tmp
    # print('iso error:', np.array(iso_abc[0][2:]) - calc_ABC(*text_converter(result.replace('O', '18O'))))


sum_result = {}
p_bar = None
def optimize_coor_file_list(fp, sep_idx, target_ABC, save_fp=None, count=0, isotope_target_ABC=None):
    global p_bar
    with open(fp) as f:
        t = f.read()

    title_repeat_num = 1
    coor_dict = {}
    for i in re.split(re_file_split, t):
        if not i:
            continue
        result = re.findall(re_coor, i)
        if not result:
            continue
        result = ['\t'.join(x) for x in result]
        for n, idx in enumerate(sep_idx):
            result.insert(idx+n, '')
        title = '[%s]' % i.split('\n')[0]
        if coor_dict.get(title):
            title += str(title_repeat_num)
            title_repeat_num += 1
        coor_dict[title] = '\n'.join(result)
        if count:
            if len(coor_dict) == count:
                break

    if not save_fp:
        save_fp = fp + 'optimize_coor.xyz'

    t0 = datetime.now()
    pool = ProcessPoolExecutor(process_num)

    p_bar = tqdm(total=len(coor_dict))
    for k, v in coor_dict.items():
        pool.submit(optimize_coor, v, target_ABC, k, isotope_target_ABC).add_done_callback(optimize_one_done)
    pool.shutdown(wait=True)
    p_bar.close()
    t1 = datetime.now()
    consumed_time = t1 - t0

    if sum_result:
        with open(save_fp, 'w', encoding='utf-8') as f:
            f.write(
                f'Input file: {file_path}\nRotation method: {rot_method}\nftol: {ftol}\njac: {jac}\nTarget ABC: {target_ABC}\nTime used: {str(consumed_time)}\n\n')
            for k, v in sum_result.items():
                error, result, msg = v
                f.write('%s\n%.3e\t%s\n%s\n\n' % (k, np.sum(error ** 2), msg, result))
    else:
        print('no result')

    # plt.plot([x[0] for x in sum_result.values()])
    # plt.show()


def calc_error(motion_inf_list, target_ABC, molecules, isotope_target_ABC_list=None):
    # motion_inf_list = [*7,*7,..]
    mass, coor = get_mass_and_coor_vector(motion_inf_list, molecules)
    ABC = calc_ABC(mass, coor)
    # scale of error effect gtol, large get more precise result, but slow
    ABC_error = (target_ABC - ABC) / target_ABC * 1000
    ABC_error = np.sum(ABC_error**2)

    iso_ABC_error = 0
    if isotope_target_ABC_list:
        for i in isotope_target_ABC_list:
            idx, iso_atom, *iso_tar_ABC = i
            ori_mass = mass[idx]

            mass[idx] = mass_table[iso_atom]
            iso_ABC = calc_ABC(mass, coor)
            single_iso_ABC_error = (iso_tar_ABC - iso_ABC) / iso_tar_ABC * 1000
            iso_ABC_error += np.sum(single_iso_ABC_error**2)

            mass[idx] = ori_mass

    # error = target_ABC - ABC
    error = ABC_error + iso_ABC_error
    return error

def calc_ABC(mass, coor):
    # coor is array,[[][][]]
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
    sorted_ABC = -np.sort(-ABC_in_MHz)

    return sorted_ABC

def text_converter(text):
    result = re.findall(re_coor, text)
    if not result:
        return
    mass = []
    coor = []
    for i in result:
        mass.append(mass_table[i[0]])
        coor.append(list(map(float, i[1:])))
    return np.array(mass), np.array(coor)


def coordination_generator(motion_inf_list, molecules):
    # to principal axis is done in get_mass_and_coor_vector
    mass, coor = get_mass_and_coor_vector(motion_inf_list, molecules)
    # principle_axis_coor = convert_to_main_axis(mass, coor)
    lines = []
    for m, c in zip(mass, coor):
        lines.append('{}\t{:>11.8f}\t{:>11.8f}\t{:>11.8f}'.format(convert_mass_to_atom_str(m), *c))
    return '\n'.join(lines)


def convert_mass_to_atom_str(mass):
    for key, val in mass_table.items():
        if val == mass:
            return key


def convert_to_main_axis(mass, coor):
    mass_sum = np.sum(mass)
    mass_center = np.sum(mass.reshape(-1, 1) * coor, 0) / mass_sum
    coor_centered = coor - mass_center

    Ixx = np.sum(mass * (coor_centered[:, 1] ** 2 + coor_centered[:, 2] ** 2))
    Iyy = np.sum(mass * (coor_centered[:, 0] ** 2 + coor_centered[:, 2] ** 2))
    Izz = np.sum(mass * (coor_centered[:, 0] ** 2 + coor_centered[:, 1] ** 2))
    Ixy = -np.sum(mass * coor_centered[:, 0] * coor_centered[:, 1])
    Ixz = -np.sum(mass * coor_centered[:, 0] * coor_centered[:, 2])
    Iyz = -np.sum(mass * coor_centered[:, 1] * coor_centered[:, 2])

    w, v = np.linalg.eigh(np.matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]))

    for i in range(0, len(mass)):
        coor_centered[i] = np.matmul(v.T, coor_centered[i])

    return coor_centered


if __name__ == '__main__':
    # water hexamer:
    # cage: 2161.8762, 1129.3153, 1066.9627; prism: 1658.224, 1362.000, 1313.124; book: 1879.4748, 1063.9814, 775.0619
    # heptamer: PR1:1304.4355, 937.8844, 919.5236; PR2:1345.1594, 976.8789, 854.4739
    # nonamer: D1:787.2274, 619.5586, 571.8116; S1: 774.7442, 633.5400, 570.6459; S2:776.7668, 626.8284, 563.3916; S3:761.7375,642.6831, 563.0343; D2:763.0778,643.8806, 564.1854
    # octamer: PPD1: 576.5298,555.3607, 497.7572; PPS1: 567.8751,562.6901, 497.454; PPS2: 576.9403,563.5092, 480.954; PPS3: 577.7268,563.2242, 480.586
    # target_ABC = np.array([2648.36020, 1131.68565, 1047.42194])
    # target_ABC = np.array([1299.84034, 460.03769, 433.66443])
    target_ABC = np.array([1488.44069,828.38884,664.21572])
#     t = '''O        -1.074218      -1.941212      -0.789132
# H        -1.012845      -1.783302       0.177397
# H        -0.988378      -2.896211      -0.900355
#
# O         1.066249      -0.295116      -1.792634
# H         0.341818      -0.907633      -1.559982
# H         0.691896       0.584803      -1.602655
#
# O        -2.475159       0.651108      -0.299381
# H        -2.234737       0.226414       0.547956
# H        -2.438795      -0.097031      -0.915923
#
# O        -0.206371       2.106083      -0.736908
# H        -1.103965       1.703779      -0.630504
# H        -0.356985       3.000666      -1.065603
#
# O         0.924739       0.860969       1.620240
# H         1.728534       0.436758       1.246953
# H         0.614203       1.442087       0.901271
#
# O         2.892463      -0.541065       0.248033
# H         2.354801      -0.492972      -0.579200
# H         3.797797      -0.344812      -0.020788
#
# O        -1.114754      -0.813046       1.746943
# H        -0.314458      -0.212311       1.794334
# H        -1.284380      -1.100181       2.652181
# '''
#     x = calc_ABC(*text_converter(t))
#     x = [round(q, 6) for q in x]
#     print(', '.join(map(str, x)))
#     print('\n'.join(map(str, x)))
#     print(target_ABC - x, (target_ABC-x)/target_ABC * 100)
#     print(target_ABC - x, ['%.3e' % q for q in (target_ABC-x)/target_ABC])
    #
    # opt_coor, unc = optimize_coor(t, target_ABC, ignore_error=False)
    # print(opt_coor)
    # print(unc)
    # x = calc_ABC(*text_converter(opt_coor))
    # print(target_ABC - x, (target_ABC - x) / target_ABC * 100)
    #
    # optimize_coor_Quasi_Monte_Carlo_sample(t, target_ABC, loop_num=576000, acceptable_error=10000, save_file_path=r"D:\MW\data\程序优化结构测试\水六聚体\opt_qmc.xyz")
    # exit()



    file_path = r"C:\Users\mike chen\Documents\WeChat Files\wxid_vom3jfqtyo9x22\FileStorage\File\2023-04\change_multistructures.xyz"
    text = '''  C     -3.017483    0.498314    1.411246
  C     -3.131057    0.433686    0.050862
  S     -1.647857    0.817644   -0.722589
  C     -0.858792    1.070554    0.780555
  C     -1.708309    0.867754    1.829831
  H     -3.835621    0.294147    2.087194
  H     -3.999714    0.177316   -0.533293
  H      0.185291    1.332607    0.807764
  H     -1.410341    0.968505    2.863497
  
  C      1.646544    0.009297   -1.895725
  C      2.402502    0.778351   -1.059597
  S      2.807378   -0.069745    0.379211
  C      1.960323   -1.471741   -0.137457
  C      1.394937   -1.287399   -1.366568
  H      1.285541    0.350799   -2.854834
  H      2.743313    1.789628   -1.211506
  H      1.926374   -2.347328    0.490444
  H      0.823010   -2.048198   -1.878480
  
  O     -1.076167   -2.426866    0.879897
  H     -0.510893   -1.881278    0.319742
  H     -1.727930   -1.806813    1.228032'''
    # result = optimize_coor(text, target_ABC)
    # print(result)
    # print('new error:', target_ABC-calc_ABC(*text_converter(result)))
    # exit()
    iso_abc = [
        (18, '18O', 1242.416, 458.95570, 426.62361),
    ]
    sep_idx = [3, 6, 9, 12, 15, 18]
    optimize_coor_file_list(file_path, sep_idx, target_ABC)
    # result, msg, __ = optimize_coor(text, target_ABC, 0, isotope_target_ABC=iso_abc)
    # print(result,'\n', msg)
    # print('parent error:', target_ABC-calc_ABC(*text_converter(result)))
    # print('parent error:', np.array(iso_abc[0][2:]) - calc_ABC(*text_converter(result.replace('O','18O'))))
    exit()

    di = r'D:\MW\data\SO2_H2O\coor\opt_try'
    parm = {
        '1s2w': ([3, 6],                (2089.5204, 1459.8550, 1194.6845)),
        '1s3w': ([3, 6,9],              (2089.5204, 1459.8550, 1194.6845)),
        '1s4w': ([3, 6,9,12],           (2089.5204, 1459.8550, 1194.6845)),
        '1s5w': ([3, 6, 9, 12, 15],     (2089.5204, 1459.8550, 1194.6845)),
        '1s6w': ([3, 6, 9, 12, 15,18],  (2089.5204, 1459.8550, 1194.6845)),
        '2s1w': ([3, 6],                (2089.5204, 1459.8550, 1194.6845)),
        '2s2w': ([3, 6,9],              (2089.5204, 1459.8550, 1194.6845)),
        '2s3w': ([3, 6,9,12],           (2089.5204, 1459.8550, 1194.6845)),
        '2s4w': ([3, 6, 9, 12, 15],     (2089.5204, 1459.8550, 1194.6845)),
        '2s5w': ([3, 6, 9, 12, 15,18],  (2089.5204, 1459.8550, 1194.6845)),
        '2s6w': ([3, 6,9,12,15,18,21],  (2089.5204, 1459.8550, 1194.6845)),
        '3s1w': ([3, 6,9],              (2089.5204, 1459.8550, 1194.6845)),
        '3s2w': ([3, 6,9,12],           (2089.5204, 1459.8550, 1194.6845)),
        '3s3w': ([3, 6, 9, 12, 15],     (2089.5204, 1459.8550, 1194.6845)),
        '3s4w': ([3, 6, 9, 12, 15,18],  (2089.5204, 1459.8550, 1194.6845)),
        '3s5w': ([3, 6,9,12,15,18,21],  (2089.5204, 1459.8550, 1194.6845)),
    }
    import os
    for k, v in parm.items():
        fn = '{}_crest_conformers.xyz'.format(k)
        ffn = os.path.join(di, fn)
        print(ffn)
        optimize_coor_file_list(ffn, v[0], np.array(v[1]))
        sum_result.clear()

