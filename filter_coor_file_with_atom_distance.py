import numpy as np
import re
import os

re_coor = re.compile(r'\s*(\d{0,3}[A-Z][a-z]?)\s+(\S+)\s+(\S+)\s+(\S+)\s*')
re_double_end_line = re.compile(r'\n\s*\n')
# re_file_split = re.compile(r'\s{2}\d{1,3}\n')  # crest
re_file_split = re_double_end_line

# doi: 10.1039/B801115J
covalent_radius = {
    'H': 0.36,
    'C': 0.69,
    'N': 0.71,
    'O': 0.66,
    'F': 0.57,
    'S': 1.05,
}

sep_idx = (14, 17, 20, 23, 26, 29)


def verify_coor_text(coor_text, sid=sep_idx):
    result = re.findall(re_coor, coor_text)
    if not result:
        return None

    fragments = []
    previous_idx = 0
    for idx in sid:
        fragments.append(result[previous_idx:idx])
        previous_idx = idx
    fragments.append(result[idx:])

    for n in range(len(fragments)-1):
        f1 = fragments[n]
        for m in range(n+1, len(fragments)):
            f2 = fragments[m]

            for i in f1:
                a1 = i[0]
                c1 = np.array(list(map(float, i[1:])))
                for j in f2:
                    a2 = j[0]
                    c2 = np.array(list(map(float, j[1:])))

                    distance = np.sqrt(np.sum((c1-c2)**2))
                    if distance < (covalent_radius[a1] + covalent_radius[a2]) * 1.4:
                        return False
    return True

def filter_one_file(file_path):
    with open(file_path) as f:
        t = f.read()

    total_count = 0
    output_text = []
    for sep_coor in re.split(re_file_split, t):
        r = verify_coor_text(sep_coor)
        if isinstance(r, bool):
            if r:
                output_text.append(sep_coor)
            total_count += 1

    print(file_path, f': {len(output_text)} / {total_count}')
    return output_text


if __name__ == '__main__':
    file_path = r"D:\MW\data\程序优化结构测试\苯甲醛-水\6c.xyzoptimize_coor.xyz"
    result = filter_one_file(file_path)
    if result:
        with open(file_path + 'distance_filter.xyz', 'w') as f:
            f.write('\n\n'.join(result))
    else:
        print('no result')