import numpy as np
import pandas as pd

spaceGroupList = ['A*', 'A***', 'A**a', 'A*/*', 'A*/a', 'A*a*', 'A-1', 'A1', 'A11*/a', 'A112', 'A112/a', 'A112/m', 'A11a', 'A2', 'A2/a', 'A2/m', 'A2/n', 'A2122', 'A21am', 'A21ma', 'A222', 'Aa', 'Ab2a', 'Aba*', 'Aba2', 'Abaa', 'Abam',
                  'Abm2', 'Abma', 'Ac2a', 'Acaa', 'Acam', 'Am', 'Ama2', 'Amaa', 'Amam', 'Amm2', 'Amma', 'Ammm', 'An', 'B**b', 'B*a*', 'B-1', 'B1', 'B11*/b', 'B112', 'B112/b', 'B112/m', 'B112/n', 'B11b', 'B2', 'B2/c', 'B21', 'B21/*',
                  'B21/a', 'B21/c', 'B21/d', 'B21/m', 'B2212', 'B222', 'B2ab', 'B2cb', 'B2mb', 'Bb21m', 'Bba*', 'Bba2', 'Bbab', 'Bbam', 'Bbcb', 'Bbcm', 'Bbmb', 'Bbmm', 'Bm21b', 'Bmab', 'Bmam', 'Bmm2', 'Bmmb', 'Bmmm', 'C*', 'C***',
                  'C*/*', 'C*/c', 'C*c*', 'C-1', 'C-4m21', 'C1', 'C112/a', 'C112/b', 'C1121', 'C1121/b', 'C11a', 'C11b', 'C2', 'C2/c', 'C2/c11', 'C2/m', 'C2/m11', 'C2/n', 'C222', 'C2221', 'C2ca', 'C2cb', 'C2cm', 'C2mb', 'C2mm',
                  'C41221', 'Cc', 'Cc**', 'Cc*a', 'Cc2a', 'Cc2b', 'Cc2m', 'Ccc*', 'Ccc2', 'Ccca', 'Cccb', 'Cccm', 'Ccm21', 'Ccma', 'Ccmb', 'Ccmm', 'Cm', 'Cm2a', 'Cm2m', 'Cmc21', 'Cmca', 'Cmcm', 'Cmm2', 'Cmma', 'Cmmm', 'Cn', 'F***',
                  'F*3*', 'F*3c', 'F-1', 'F-43c', 'F-43m', 'F1', 'F2', 'F2/d', 'F2/m', 'F222', 'F23', 'F2dd', 'F2mm', 'F4132', 'F432', 'Fd-3', 'Fd-3c', 'Fd-3m', 'Fd11', 'Fd2d', 'Fd3', 'Fd3c', 'Fd3m', 'Fdd2', 'Fddd', 'Fm-3', 'Fm-3c',
                  'Fm-3m', 'Fm2m', 'Fm3', 'Fm3c', 'Fm3m', 'Fmm2', 'Fmmm', 'I*', 'I***', 'I**a', 'I**d', 'I*/*', 'I*/a', 'I*3', 'I*3*', 'I*c*', 'I-1', 'I-4', 'I-42d', 'I-42m', 'I-43d', 'I-43m', 'I-4c2', 'I-4m2', 'I1', 'I112', 'I112/a',
                  'I112/b', 'I112/m', 'I2', 'I2/a', 'I2/b11', 'I2/c', 'I2/c11', 'I2/m', 'I2/m11', 'I212121', 'I213', 'I222', 'I23', 'I2am', 'I2cb', 'I2cm', 'I2ma', 'I2mm', 'I4', 'I4*d', 'I4/*', 'I4/***', 'I4/*c*', 'I4/m', 'I4/mcm',
                  'I4/mmm', 'I41', 'I41/a', 'I41/acd', 'I41/amd', 'I4122', 'I4132', 'I41cd', 'I41md', 'I422', 'I432', 'I4cm', 'I4mm', 'Ia', 'Ia-3', 'Ia-3d', 'Ia3', 'Ia3d', 'Iba*', 'Iba2', 'Ibam', 'Ibca', 'Ibm2', 'Ibmm', 'Ic', 'Ic2a',
                  'Ic2m', 'Icab', 'Icm2', 'Icma', 'Icmm', 'Im', 'Im**', 'Im-3', 'Im-3m', 'Im2a', 'Im2m', 'Im3', 'Im3m', 'Ima2', 'Imam', 'Imcb', 'Imcm', 'Imm2', 'Imma', 'Immb', 'Immm', 'P*', 'P***', 'P**2', 'P**a', 'P**b', 'P**n',
                  'P*,-3', 'P*/*', 'P*/a', 'P*/c', 'P*/n', 'P*3', 'P*3n', 'P*aa', 'P*ab', 'P*c*', 'P*cb', 'P*cc', 'P*cn', 'P*n*', 'P*nb', 'P-1', 'P-3', 'P-31c', 'P-31m', 'P-3c1', 'P-3m1', 'P-4', 'P-421c', 'P-421m', 'P-42c', 'P-42m',
                  'P-43m', 'P-43n', 'P-4b2', 'P-4c2', 'P-4m2', 'P-4n2', 'P-6', 'P-62c', 'P-62m', 'P-6c2', 'P-6m2', 'P1', 'P112', 'P112/a', 'P112/b', 'P112/m', 'P112/n', 'P1121', 'P1121/a', 'P1121/b', 'P1121/m', 'P1121/n', 'P11a',
                  'P11b', 'P11m', 'P11n', 'P2', 'P2/a', 'P2/c', 'P2/c11', 'P2/m', 'P2/n', 'P21', 'P21/*', 'P21/a', 'P21/b11', 'P21/c', 'P21/c11', 'P21/m', 'P21/m11', 'P21/n', 'P21/n11', 'P2111', 'P2121*', 'P21212', 'P212121', 'P2122',
                  'P21221', 'P213', 'P21ab', 'P21am', 'P21ca', 'P21cn', 'P21ma', 'P21mn', 'P21nb', 'P21nm', 'P2212', 'P22121', 'P222', 'P2221', 'P23', 'P2aa', 'P2an', 'P2cb', 'P2mm', 'P2na', 'P2nn', 'P3', 'P3*1', 'P31', 'P31,2',
                  'P31,221', 'P3112', 'P312', 'P3121', 'P31c', 'P31m', 'P32', 'P321', 'P3212', 'P3221', 'P3c1', 'P3m1', 'P4', 'P4/*', 'P4/***', 'P4/*b*', 'P4/*n*', 'P4/m', 'P4/mbm', 'P4/mcc', 'P4/mmm', 'P4/mnc', 'P4/n', 'P4/nbm',
                  'P4/ncc', 'P4/nmm', 'P4/nnc', 'P41', 'P41,3', 'P41,3212', 'P41,322', 'P41,332', 'P41212', 'P4122', 'P4132', 'P42', 'P42/*', 'P42/m', 'P42/mbc', 'P42/mcm', 'P42/mmc', 'P42/mnm', 'P42/n', 'P42/nbc', 'P42/ncm',
                  'P42/nmc', 'P42/nnm', 'P4212', 'P422', 'P42212', 'P4222', 'P4232', 'P42bc', 'P42cm', 'P42mc', 'P42nm', 'P43', 'P432', 'P43212', 'P4322', 'P4332', 'P4bm', 'P4cc', 'P4mm', 'P4nc', 'P6', 'P6/*', 'P6/***', 'P6/**c',
                  'P6/*cc', 'P6/m', 'P6/mcc', 'P6/mmm', 'P61', 'P61,5', 'P61,522', 'P6122', 'P62', 'P62,4', 'P62,422', 'P622', 'P6222', 'P63', 'P63/*', 'P63/m', 'P63/mcm', 'P63/mmc', 'P6322', 'P63cm', 'P63mc', 'P64', 'P6422', 'P65',
                  'P6522', 'P6cc', 'P6mm', 'Pa', 'Pa-3', 'Pa3', 'Pb**', 'Pb*a', 'Pb-3', 'Pb11', 'Pb21a', 'Pb21m', 'Pb2n', 'Pb3', 'Pba*', 'Pba2', 'Pbaa', 'Pbam', 'Pban', 'Pbc*', 'Pbc21', 'Pbca', 'Pbcb', 'Pbcm', 'Pbcn', 'Pbm2', 'Pbma',
                  'Pbmm', 'Pbmn', 'Pbn*', 'Pbn21', 'Pbna', 'Pbnb', 'Pbnm', 'Pbnn', 'Pc', 'Pc**', 'Pc*a', 'Pc*b', 'Pc*n', 'Pc11', 'Pc21b', 'Pc21n', 'Pc2a', 'Pc2m', 'Pca*', 'Pca21', 'Pcaa', 'Pcab', 'Pcam', 'Pcan', 'Pcc2', 'Pcca', 'Pccb',
                  'Pccm', 'Pccn', 'Pcm21', 'Pcma', 'Pcmb', 'Pcmm', 'Pcmn', 'Pcn*', 'Pcn2', 'Pcna', 'Pcnb', 'Pcnm', 'Pcnn', 'Pm', 'Pm*n', 'Pm-3', 'Pm-3m', 'Pm-3n', 'Pm21b', 'Pm21n', 'Pm3', 'Pm3m', 'Pm3n', 'Pma2', 'Pmaa', 'Pmab', 'Pmam',
                  'Pman', 'Pmc21', 'Pmca', 'Pmcb', 'Pmcm', 'Pmcn', 'Pmm2', 'Pmma', 'Pmmm', 'Pmmn', 'Pmn*', 'Pmn21', 'Pmna', 'Pmnb', 'Pmnm', 'Pmnn', 'Pn', 'Pn**', 'Pn*a', 'Pn*n', 'Pn-3', 'Pn-3m', 'Pn-3n', 'Pn21a', 'Pn21m', 'Pn2b',
                  'Pn2n', 'Pn3', 'Pn3m', 'Pn3n', 'Pna*', 'Pna21', 'Pnaa', 'Pnab', 'Pnam', 'Pnan', 'Pnc2', 'Pnca', 'Pncb', 'Pncm', 'Pncn', 'Pnm21', 'Pnma', 'Pnmb', 'Pnmm', 'Pnmn', 'Pnn*', 'Pnn2', 'Pnna', 'Pnnb', 'Pnnm', 'Pnnn', 'R*',
                  'R**', 'R*c', 'R-3', 'R-3c', 'R-3m', 'R3', 'R32', 'R3c', 'R3m', 'Unknown']
spaceGroupPops = [2, 1, 3, 6, 20, 1, 42, 13, 1, 5, 62, 4, 6, 84, 283, 35, 50, 10, 10, 7, 4, 55, 9, 2, 1088, 1, 10, 65, 26, 5, 3, 22, 5, 196, 1, 6, 38, 5, 4, 13, 1, 1, 28, 2, 1, 19, 193, 17, 1, 29, 1, 1, 34, 2, 13, 65, 6, 1, 27, 1, 2,
                  2, 2, 3, 1, 1, 1, 8, 1, 8, 1, 1, 7, 5, 1, 1, 9, 1, 4, 8, 33, 116, 8, 104, 1, 8, 4, 2, 7, 2, 1, 1, 8898, 88639, 4, 5249, 3, 89, 77, 1935, 9, 66, 11, 2, 4, 2, 11240, 2, 2, 13, 1, 7, 1, 133, 525, 2, 141, 41, 4, 32, 31,
                  368, 2, 5, 1475, 1258, 1041, 11, 80, 221, 9, 10, 10, 1, 21, 114, 59, 1, 3, 13, 3, 37, 131, 133, 3, 66, 64, 119, 146, 172, 3, 26, 20, 23, 51, 3594, 1241, 46, 95, 844, 6, 8, 2, 87, 79, 148, 5, 13, 1, 2, 6, 45, 2, 5, 1,
                  57, 1502, 724, 224, 334, 482, 118, 41, 6, 1, 18, 14, 3, 571, 3980, 2, 532, 3, 424, 2, 78, 149, 253, 208, 1, 13, 5, 1, 6, 306, 1, 6, 1, 3, 825, 156, 301, 272, 4033, 592, 280, 218, 33, 441, 60, 104, 82, 45, 27, 367,
                  142, 128, 19, 9, 5, 625, 455, 317, 2, 1, 33, 7, 8, 4, 1, 7, 3, 31, 1, 160, 180, 4, 5, 14, 42, 139, 5, 9, 10, 72, 284, 2, 161, 241, 19, 2, 1, 6, 3, 5, 23, 7, 28, 3, 1, 2, 1, 1, 1, 1, 1, 3, 9, 7, 283356, 1272, 524, 64,
                  819, 162, 244, 1342, 330, 36, 6, 115, 170, 68, 44, 7, 193, 44, 201, 51, 19, 50, 10926, 2, 10, 15, 4, 11, 198, 206, 802, 20, 462, 2, 14, 1, 4, 193, 201, 4252, 1, 162, 2862, 57673, 86, 11815, 29, 216538, 21, 5298, 1,
                  157711, 28, 15, 5, 4143, 78676, 6, 75, 676, 65, 3, 65, 236, 20, 14, 180, 14, 10, 329, 39, 90, 25, 2, 1, 6, 3, 8, 11, 261, 2, 813, 4, 4, 29, 18, 1026, 388, 26, 818, 110, 25, 810, 114, 8, 63, 1, 1, 1, 1, 53, 82, 114,
                  213, 223, 973, 68, 539, 241, 252, 1004, 16, 10, 2, 2, 2125, 99, 68, 126, 4, 118, 116, 25, 158, 219, 1422, 50, 160, 129, 63, 104, 13, 225, 9, 6, 116, 9, 2, 32, 880, 45, 1934, 99, 59, 6, 35, 6, 132, 33, 3, 2, 4, 1, 60,
                  143, 149, 729, 6, 1, 323, 85, 6, 1, 27, 89, 776, 6, 1230, 80, 462, 202, 38, 168, 60, 52, 709, 268, 15, 4, 124, 780, 230, 1, 4, 1, 1, 78, 11, 4, 3, 6, 185, 5, 319, 104, 12, 438, 35244, 12, 999, 8713, 1, 22, 1, 4, 12,
                  195, 182, 77, 167, 20, 2753, 2, 2, 4, 4, 2, 160, 243, 5, 2, 6, 7740, 6, 1370, 42, 162, 34, 502, 4, 21, 3752, 5, 9, 17, 2, 142, 1, 3, 4, 98, 2, 20, 44, 1, 40, 313, 97, 1, 29, 8, 31, 11, 17, 1, 12, 2, 5, 141, 5, 7, 4,
                  211, 9, 71, 41, 287, 3, 556, 143, 139, 10, 30, 2155, 10, 12, 1, 51, 28, 163, 512, 14, 4, 5, 10, 11, 19, 28, 14124, 129, 103, 463, 20, 138, 120, 5, 7, 7, 11, 10362, 5, 24, 32, 12, 307, 1135, 11, 714, 76, 10, 3, 3,
                  9323, 2037, 703, 1517, 530, 1115, 333, 2953]


def loadSpaceGroups():
    sgDict = pd.DataFrame(columns=['system', 'laue', 'class', 'ce', 'SG'], index=np.arange(0, 230 + 1))
    sgDict.loc[0] = pd.Series({'system': 'n/a', 'laue': 'n/a', 'class': 'n/a', 'ce': 'n/a', 'SG': 'n/a'})  # add an entry for misc / throwaway categories

    # Triclinic
    i = 1
    sgDict.loc[i] = pd.Series({'system': 'triclinic',
                               'laue': '-1',
                               'class': '1',
                               'ce': 'p',
                               'SG': 'P1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'triclinic',
                               'laue': '-1',
                               'class': '-1',
                               'ce': 'p',
                               'SG': 'P-1'
                               })
    i += 1

    # Monoclinic
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2',
                               'ce': 'p',
                               'SG': 'P2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2',
                               'ce': 'p',
                               'SG': 'P21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2',
                               'ce': 'c',
                               'SG': 'C2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'p',
                               'SG': 'Pm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'p',
                               'SG': 'Pc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'c',
                               'SG': 'Cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'c',
                               'SG': 'Cc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P2/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P21/m'  # alt - P21m
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'c',
                               'SG': 'C2/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P2/c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P21/c'  # alt - P21c
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'c',
                               'SG': 'C2/c'
                               })
    i += 1

    # Orthorhombic
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P2221'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P21212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P212121'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'c',
                               'SG': 'C2221'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'c',
                               'SG': 'C222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'f',
                               'SG': 'F222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'i',
                               'SG': 'I222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'i',
                               'SG': 'I212121'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pmm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pmc21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pcc2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pma2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pca21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pnc2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pmn21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pba2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pna21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pnn2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Cmm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Cmc21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Ccc2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Amm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Abm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Ama2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Aba2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'f',
                               'SG': 'Fmm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'f',
                               'SG': 'Fdd2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'i',
                               'SG': 'Imm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'i',
                               'SG': 'Iba2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'i',
                               'SG': 'Ima2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnnn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pccm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pban'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmma'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnna'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmna'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pcca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbam'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pccn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnnm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmmn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbcn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnma'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cccm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmma'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Ccca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'f',
                               'SG': 'Fmmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'f',
                               'SG': 'Fddd'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Immm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Ibam'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Ibca'  # aka Ibcm
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Imma'
                               })
    i += 1

    # Tetragonal
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P41'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P42'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P43'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'i',
                               'SG': 'I4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'i',
                               'SG': 'I41'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '-4',
                               'ce': 'p',
                               'SG': 'P-4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '-4',
                               'ce': 'i',
                               'SG': 'I-4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P4/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P42/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P4/n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P42/n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'i',
                               'SG': 'I4/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'i',
                               'SG': 'I41/a'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P422'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4122'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P41212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P42212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4322'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P43212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'i',
                               'SG': 'I422'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'i',
                               'SG': 'I4122'  # aka I4212
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4mm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4bm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42nm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4cc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4nc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42mc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42bc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I4mm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I4cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I41md'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I41cd'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-42m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-42c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-421m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-421c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4m2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4c2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4b2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4n2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'i',
                               'SG': 'I-4m2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'i',
                               'SG': 'I-4c2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'i',
                               'SG': 'I-42m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'I-42d'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mcc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/nbm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/nnc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mbm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mnc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/nmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/ncc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mmc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/nbc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/nnm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mbc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mnm'  # incorrectly /mcm in source
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/nmc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/ncm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I4/mmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I4/mcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I41/amd'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I41/acd'
                               })
    i += 1

    # Trigonal
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'p',
                               'SG': 'P3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'p',
                               'SG': 'P31'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'p',
                               'SG': 'P32'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'r',
                               'SG': 'R3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '-3',
                               'ce': 'p',
                               'SG': 'P-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '-3',
                               'ce': 'r',
                               'SG': 'R-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '312',
                               'ce': 'p',
                               'SG': 'P312'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'p',
                               'SG': 'P321'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '312',
                               'ce': 'p',
                               'SG': 'P3112'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'p',
                               'SG': 'P3121'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '312',
                               'ce': 'p',
                               'SG': 'P3212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'p',
                               'SG': 'P3221'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'r',
                               'SG': 'R32'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'p',
                               'SG': 'P3m1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '31m',
                               'ce': 'p',
                               'SG': 'P31m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'p',
                               'SG': 'P3c1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '31m',
                               'ce': 'p',
                               'SG': 'P31c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'r',
                               'SG': 'R3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'r',
                               'SG': 'R3c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-31m',
                               'ce': 'p',
                               'SG': 'P-31m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-31m',
                               'ce': 'p',
                               'SG': 'P-31c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'p',
                               'SG': 'P-3m1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'p',
                               'SG': 'P-3c1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'r',
                               'SG': 'R-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'r',
                               'SG': 'R-3c'
                               })
    i += 1

    # Hexagonal
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P6'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P61'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P65'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P62'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P64'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P63'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '-6',
                               'ce': 'p',
                               'SG': 'P-6'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6/m',
                               'ce': 'p',
                               'SG': 'P6/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6/m',
                               'ce': 'p',
                               'SG': 'P63/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P622'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6122'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6522'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6422'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6322'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P6mm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P6cc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P63cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P63mc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-6m2',
                               'ce': 'p',
                               'SG': 'P-6m2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-6m2',
                               'ce': 'p',
                               'SG': 'P-6c2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-62m',
                               'ce': 'p',
                               'SG': 'P-62m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-62m',
                               'ce': 'p',
                               'SG': 'P-62c'  # error in source, missing negative http://pd.chem.ucl.ac.uk/pdnn/symm3/allsgp.htm
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P6/mmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P6/mcc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P63/mcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P63/mmc'
                               })
    i += 1

    # Cubic
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'p',
                               'SG': 'P23'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'f',
                               'SG': 'F23'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'i',
                               'SG': 'I23'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'p',
                               'SG': 'P213'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'i',
                               'SG': 'I213'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'p',
                               'SG': 'Pm-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'p',
                               'SG': 'Pn-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'f',
                               'SG': 'Fm-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'f',
                               'SG': 'Fd-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'i',
                               'SG': 'Im-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'p',
                               'SG': 'Pa-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'i',
                               'SG': 'Ia-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P432'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P4232'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'f',
                               'SG': 'F432'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'f',
                               'SG': 'F4132'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'i',
                               'SG': 'I432'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P4332'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P4132'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'i',
                               'SG': 'I4132'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'p',
                               'SG': 'P-43m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'f',
                               'SG': 'F-43m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'i',
                               'SG': 'I-43m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'p',
                               'SG': 'P-43n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'f',
                               'SG': 'F-43c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'i',
                               'SG': 'I-43d'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pm-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pn-3n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pm-3n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pn-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fm-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fm-3c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fd-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fd-3c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'i',
                               'SG': 'Im-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'i',
                               'SG': 'Ia-3d'
                               })

    return sgDict
