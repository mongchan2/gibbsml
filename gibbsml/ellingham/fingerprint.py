import os # 파일관 관계 참고
import json # json 파일을 사용할것임 
from functools import reduce
from math import gcd # gcd는 왠지 모르겠짐나 사용한다고 한다. 

from ase.io import read 
from mendeleev import element
import numpy as np
from pymatgen.ext.matproj import MPRester

__author__ = "Jose A. Garrido Torres, Alexander Urban"
__email__ = "aurban@atomistic.net"
__date__ = "2021-03-18"
__version__ = "1.0"


class Fingerprint:

    def __init__(self, USER_API_KEY): # 일단 Fingerprint 객체의 경우에는 USER_API_KEY를 인수로 하여 생성자가 실행된다. 
        """
        Parameters
        ----------
        USER_API_KEY: str
            User's API key generated from the Materials Project database.
            See https://materialsproject.org/open
        """

        self.user_api_key = USER_API_KEY
        self.fp = {}  # Dictionary containing the set of features.
        self.implemented_features = ['stoichiometry', 'electronegativity',
                                     'mass', 'volume', 'density',
                                     'bulk modulus', 'shear modulus',
                                     'poisson ratio', 'anisotropy',
                                     'spacegroup', 'ionic character']
        self.selected_features = None
        self.label = None
        # 이 객체 자체가 가지는 property, 특히 implented_feature을 통해서 GPR을 진행한다.  
        
    # MP를 통해서 feature을 추출하는 과정을 보여준다. => 추후에 우리 방법대로 수정할 수 있는 가능성이 존재한다. 
    def extract_mp_features(self, id_mo, id_m1='', id_m2='',
                            id_oxygen='mp-12957',
                            selected_features='all', label=None,
                            mo_energy_correction=True):
        # input으로 material project ID가 필요하다. id_m1과 m2의 경우에는 화학식을 통해서 얻어지는거고
        # selective feature의 경우에는 어떤 feature을 포함할지에 대해 결정 
        """
        Generates a feature set for an oxidation for a given metal
        oxide (AxByOz) from the elements (A and B).

        Parameters
        ----------
        id_mo: str
            Materials Project mp-id for the metal oxide or chemical formula
            of the metal oxide, e.g. 'Al2SiO5' or 'mp-4753'.
        id_m1: str
            (optional) Materials Project mp-id for the metal A, e.g. 'mp-134'.
        id_m2: str
            (optional) Materials Project mp-id for the metal B, e.g. 'mp-149'.
        id_oxygen: str
            Materials project mp-id for oxygen in the gas phase.
        selected_features: list or str
            (option 1): list
                List of selected features to be considered to
                generate the fingerprint. Implemented are: 'stoichiometry',
                'electronegativity', 'mass', 'volume', 'density',
                'bulk modulus', 'shear modulus', 'poisson ratio',
                'anisotropy', 'spacegroup' and 'ionic character'.
            (option 2): str
                'all': Include all implemented features (see option 1).
                'ellingham': Recommended features for building models for
                             predicting Ellingham diagrams. Includes only
                             the following features:
                             'stoichiometry', 'electronegativity',
                             'density', 'bulk modulus', 'ionic character'.

        'label': str
            Defines the label tag for the fingerprint. The user can chose a
            name for the fingerprint for the data entry, e.g. 'Al2SiO5-PBEU'.
        'mo_energy_correction': bool
            If True the algorithm only selects Material Project entries which
            in which energy corrections are available. See:
            https://materialsproject.org/docs/calculations#Total_Energy_Adjustments

        """

        # Implemented features. OK 
        if selected_features == 'all':
            self.selected_features = self.implemented_features
        if selected_features == 'ellingham':
            self.selected_features = ['stoichiometry', 'electronegativity',
                                      'density', 'bulk modulus',
                                      'ionic character']
        else:
            self.selected_features = selected_features

        # Get ID for the oxidized state. 데이터가 없는 경우에 찾는 일종의 방법을 사용한다. 
        print("Getting information for " + id_mo)
        if "mp" not in id_mo:
            id_mo = self._find_id(id_mo)
        with MPRester(self.user_api_key) as m:
            data_mo = m.get_data(id_mo)[0] # data_mo에는 Metal Oxide에 대한 data들이 들어있는 형태임 

        # Set oxygen energy and MO energy corrections.
        with MPRester(self.user_api_key) as m:
            try:
                e_adjus = m.get_entries(id_mo)[0].__dict__['energy_adjustments']
                e_mo_corr = 0
                for e_ad in e_adjus:
                    e_mo_corr += e_ad.value
            except:
                e_adjus = m.get_entries(id_mo)[0].__dict__['correction']
                e_mo_corr = e_adjus
        if mo_energy_correction is False:
            e_mo_corr = 0.0
        data_o2 = m.get_data(id_oxygen)[0]
        e_o2 = 2 * data_o2['energy'] / data_o2['unit_cell_formula']['O']

        # Recognize whether is a unary or binary oxide. ternary 이상인 경우에는 오류 메시지를 출력한다. 
        n_elements = data_mo['nelements'] # Metal Oxide의 화학식 내에 몇 개의 element가 존재하는지 확인한다. 이를 바탕으로 binary인지 uni인지 판별 
        binary_oxide = False
        if n_elements == 3:
            binary_oxide = True
        msg = "Only unary and binary oxides are implemented."
        assert n_elements <= 3, NotImplementedError(msg)

        elements = data_mo['elements'] # 어떤 element가 있는지(ex. O, Mg) elements에 저장한다. 
        element_no_ox = np.array(elements)[~np.isin(elements, 'O')] # np.array의 함수를 통해서 Oxide에서 oxygen을 제외한 다른 파일들을 마련한다. 

        if binary_oxide is True: # binary_oxide인 경우에 
            element_m1, element_m2 = element_no_ox[0], element_no_ox[1] # element_m1에는 Mg, m2에는 Na 뭐 이런식으로 담을 수 있도록 
        else:
            element_m1, element_m2 = element_no_ox[0], element_no_ox[0] # 이 부분은 그대로 진행한다. 

        # Get info for M1 and M2. - 기본적으로 m1과 m2에 대한 정보가 제공되지 않는 경우에 대해서, _find_id method를 사용하여 Material ID를 탐색한다. 
        if "mp" not in id_m1: 
            id_m1 = self._find_id(element_m1)
        if "mp" not in id_m2:
            id_m2 = self._find_id(element_m2)
        # get_data 함수를 통해서 데이터를 확보하는데, 어떤 데이터들을 확보하는지 모르겠네. 구식 API를 사용해서, 이를 현재의 API로 변경하는 과정이 필요할것으로 예상 
        data_m1 = m.get_data(id_m1)[0]
        data_m2 = m.get_data(id_m2)[0]
        
        
        # Pretty formula의 형태로 공식을 받아오고 
        formula_mo = data_mo['pretty_formula']
        formula_m1 = data_m1['pretty_formula']
        formula_m2 = data_m2['pretty_formula']

        # Set a label for the compound if not specified. => 이 코드에서의 Label은, [formula_material_id] 의 format임
        self.label = label
        if self.label is None:
            self.label = formula_mo + '_' + id_mo

        # self.fp의 경우에는 여기 전까지는 빈 디렉토리 상태임 
        self.fp.update({self.label: {}})  # { label_name : {} } , 즉 { HfO2_mp_112 : { } } 이런 형태의 dictionary를 생성한다 
        self.fp[self.label]['target_features'] = {} # HfO2_mp_112에 'target_features', 'features' 라는 키를 추가 
        self.fp[self.label]['features'] = {} # 이후 add features를 통해서 추가해줄거임 

        # ASE Atoms object를 사용하여 포멧 변환을 진행하는 듯 
        # cif(Crystallographic Information File) format으로 작성을 진행할 수 있다. 
        atoms = []
        for i in ['m1', 'm2', 'mo']:
            f_atoms = open('tmp_Atoms.cif', 'w') # write형태로 파일을 open
            f_atoms.write(eval("data_" + i)['cif']) # 쓰는건 알겠는데 뭔소린지는 잘 모르겠음 
            f_atoms.close()
            atoms.append(read('tmp_Atoms.cif')) # atoms라는 오브젝트에 파일 내용을 써준다. 여기서의 read함수는, atom에서 가져온 파일 포멧임 
            os.remove('tmp_Atoms.cif') # write한 파일을 삭제함 

        atoms_m1, atoms_m2, atoms_mo = atoms # 각 atom에 대한 정보들이 atoms에 [m1, m2, mo] 이렇게 담겨있을텐데, m1을 atom_m1에 할당해주는 식으로 각 변수를 선언해주는 문법

        
        # Get formula unit for M1, M2 and MO.
        # 일단 기본 상태로는, Hf2Zr2O4처럼 존재하는 상황임. 따라서 unit formula에 따라서 이를 나누어주는 작업을 진행한다. 
        # 이렇게 하는 이유는 Ellingham Diagram에서 O2 1mole을 기준으로 반응이 일어난다고 가정하기 때문
        
        n_m1, fu_m1 = 2 * (len(atoms_m1),) # a, b = (c,)의 형태의 문법은, a와 b에 각각 c를 넣어주는 형태로 볼 수 있음 
        n_m2, fu_m2 = 2 * (len(atoms_m2),) # 왜 2를 곱하는지는 잘 모르겠지만, 쨋든 뭐 atom_m의 길이의 2 배를 각각 n_m, fu_m 이라는 변수에 넣어준다. 
        fu_mo = self._get_atoms_per_unit_formula(atoms_mo) # mo에 대해서 atom_per_unit_formula를 불러와준다 
        n_m1_in_mo = self._get_number_of_atoms_element(atoms_mo,
                                                       symbol=element_m1)
        n_m1_in_mo /= fu_mo
        n_m2_in_mo = self._get_number_of_atoms_element(atoms_mo,
                                                       symbol=element_m2)
        n_m2_in_mo /= fu_mo
        n_ox_in_mo = self._get_number_of_atoms_element(atoms_mo, symbol='O')
        n_ox_in_mo /= fu_mo

        # self.fp에 raw data라는 항목을 새로 추가하고, 여기에 관련 자료를 담아주는 형태로 진행한다. 
        self.fp[self.label]['raw data'] = {}
        self.fp[self.label]['raw data']['data mo'] = data_mo
        self.fp[self.label]['raw data']['data m1'] = data_m1
        self.fp[self.label]['raw data']['data m2'] = data_m2

        # Formation energy.
        # Get formation energies (a M1 + b M2 + O2 --> c M1xM2yOz).
        # 먼저 각 구조에 대한 에너지를 추출해서 
        e_m1, e_m2 = data_m1['energy'], data_m2['energy']
        e_mo = data_mo['energy']
        # 각 unit에 대한 개수를 구한 이후에 
        x, y, z = n_m1_in_mo, n_m2_in_mo, n_ox_in_mo
        # O2에 대해서 맞춰주는 과정이 포함되기 때문에, 아래와 같은 식을 사용해준다 
        a = (2 / z) * x                     # c * x = a 가 성립해야 한다.
        b = (2 / z) * y                     # c * y = b 가 성립해야 한다. 
        c = 2 / z                           # 반드시 1mol의 O2가 들어가기 때문에, c*z = 2 가 성립한다. 이에 따라서 c = 2 / z
        
        if not binary_oxide: # unary의 경우. 즉, AO2의 형태를 가지는 것들에 대해서이다.
            a /= 2
            b /= 2
            
        dH0 = c * (e_mo + e_mo_corr)/fu_mo
        dH0 -= a * e_m1/fu_m1
        dH0 -= b * e_m2/fu_m2
        dH0 -= e_o2
        dH0 *= 96.485  # eV to kJ/mol.
        self.add_feature(description='formation energy (kJ/mol)', value=dH0)
        # 계산을 굳이 이렇게 해야 하는건가.. 
        balanced_reaction = None
        
        # 이 부분은 이제 reaction에 대한 표시를 하기 위한 문자열 생성하는 부분 
        if not binary_oxide:
            balanced_reaction = str(2 * a) + " " + formula_m1 + " + O2"
            balanced_reaction += " --> "
            balanced_reaction += str(c) + " " + formula_mo

        if binary_oxide:
            balanced_reaction = str(a) + " " + formula_m1 + " + "
            balanced_reaction += str(b) + " " + formula_m2 + " + O2"
            balanced_reaction += " --> "
            balanced_reaction += str(c) + " " + formula_mo

        self.fp[self.label]['balanced reaction'] = balanced_reaction

        
        # -------------------------------------------------------
        '''
        여기부터는 일종의 self.fp[self.label]['features']에 내용을 추가하는 부분이다
        self.fp[self.label]['features']['특정한 description'] = 특정 value의 형식으로 내용을 저장 
        그러면 접근할 때 굉장히 복잡할 것으로 생각하는데, 차라리 구조체같은거 쓰는게 낫지 않을까. 아니면 클래스에서 컴포넌트를 늘리던지 
        element에 쓰는 요소들은 어떤 이유인지 모르겠짐지만 mendeleev라는걸 사용한다. 
        self.fp[self.label]['features']['특정한 description] => 이러한 format들이 계속해서 저장된 이후에, get관련 함수를 통해서 train data에 정보를 전달
        traing_set에서 이 정보들을 바탕으로 GPR에 대한 학습을 진행시킨다. 
        '''
        # -------------------------------------------------------
        # Stoichiometry. self.selected_features의 경우에는 특별한 옵션을 주지 않는 경우 이 아래에 있는 모든 if문에 대한 분기는 yes가 된다. 
        if 'stoichiometry' in self.selected_features:

            # Ratio metal / oxygen:
            ratio_o_m1 = n_m1_in_mo / n_ox_in_mo
            ratio_o_m2 = n_m2_in_mo / n_ox_in_mo
            # Average oxidation state of the metals:
            av_ox_state = (n_ox_in_mo * 2) / (n_m1_in_mo + n_m2_in_mo)
            # Atomic number:
            element_m1, element_m2 = atoms_m1[0].symbol, atoms_m2[0].symbol
            z_m1 = element(element_m1).atomic_number
            z_m2 = element(element_m2).atomic_number

            self.add_feature(description='ratio metal oxygen (mean)',
                             value=np.mean([ratio_o_m1, ratio_o_m2]))
            self.add_feature(description='ratio metal oxygen (var)',
                             value=np.var([ratio_o_m1, ratio_o_m2]))
            self.add_feature(description='average oxidation state',
                             value=av_ox_state)
            self.add_feature(description='atomic number (mean)',
                             value=np.mean([z_m1, z_m2]))
            self.add_feature(description='atomic number (var)',
                             value=np.var([z_m1, z_m2]))

        # Electronegativity (Pauling).
        if 'electronegativity' in self.selected_features:
            elecneg_m1 = element(element_m1).en_pauling
            elecneg_m2 = element(element_m2).en_pauling
            self.add_feature(description='pauling electronegativity (mean)',
                             value=np.mean([elecneg_m1, elecneg_m2]))
            self.add_feature(description='pauling electronegativity (var)',
                             value=np.var([elecneg_m1, elecneg_m2]))

        # Percentage of ionic character (Pauling).
        if 'ionic character' in self.selected_features:
            elnegdif_m1 = (element('O').en_pauling
                           - element(element_m1).en_pauling)
            elnegdif_m2 = (element('O').en_pauling
                           - element(element_m2).en_pauling)
            pio_m1 = 100 * (1 - np.exp(-(1/2 * elnegdif_m1)**2))
            pio_m2 = 100 * (1 - np.exp(-(1/2 * elnegdif_m2)**2))
            self.add_feature(description='% ionic character (mean)',
                             value=np.mean([pio_m1, pio_m2]))
            self.add_feature(description='% ionic character (var)',
                             value=np.var([pio_m1, pio_m2]))

        # Volume.
        if 'volume' in self.selected_features:
            V_m1 = atoms_m1.get_volume()
            V_per_fu_m1 = V_m1 / fu_m1
            V_m2 = atoms_m2.get_volume()
            V_per_fu_m2 = V_m2 / fu_m2
            V_mo = atoms_mo.get_volume()
            V_per_fu_mo = V_mo / fu_mo
            self.add_feature(description='volume per formula unit (mean)',
                             value=np.mean([V_per_fu_m1, V_per_fu_m2]))
            self.add_feature(description='volume per formula unit (var)',
                             value=np.var([V_per_fu_m1, V_per_fu_m2]))
            self.add_feature(description='volume MO per formula unit',
                             value=V_per_fu_mo)
            diff_V_per_fu_m1_mo = V_per_fu_mo - V_per_fu_m1
            diff_V_per_fu_m2_mo = V_per_fu_mo - V_per_fu_m2
            self.add_feature(description='difference volume (MO-M) (mean)',
                             value=np.mean([diff_V_per_fu_m1_mo,
                                            diff_V_per_fu_m2_mo]))
            self.add_feature(description='difference volume (MO-M) (var)',
                             value=np.var([diff_V_per_fu_m1_mo,
                                           diff_V_per_fu_m2_mo]))

        # Mass.
        if 'mass' in self.selected_features:
            mass_m1 = np.average(atoms_m1.get_masses())
            mass_m2 = np.average(atoms_m2.get_masses())
            mass_mo = np.average(atoms_mo.get_masses())
            mass_per_fu_m1 = mass_m1 / fu_m1
            mass_per_fu_m2 = mass_m2 / fu_m2
            mass_per_fu_mo = mass_mo / fu_mo
            self.add_feature(description='mass per formula unit (mean)',
                             value=np.mean([mass_per_fu_m1, mass_per_fu_m2]))
            self.add_feature(description='mass per formula unit (var)',
                             value=np.var([mass_per_fu_m1, mass_per_fu_m2]))
            self.add_feature(description='mass MO per formula unit',
                             value=mass_per_fu_mo)
            diff_mass_per_fu_m1_mo = mass_per_fu_mo - mass_per_fu_m1
            diff_mass_per_fu_m2_mo = mass_per_fu_mo - mass_per_fu_m2
            self.add_feature(description='difference mass (MO-M) (mean)',
                             value=np.mean([diff_mass_per_fu_m1_mo,
                                            diff_mass_per_fu_m2_mo]))
            self.add_feature(description='difference mass (MO-M) (var)',
                             value=np.var([diff_mass_per_fu_m1_mo,
                                           diff_mass_per_fu_m2_mo]))

        # Density.
        if 'density' in self.selected_features:
            dens_m1 = data_m1['density']
            dens_m2 = data_m2['density']
            dens_mo = data_mo['density']
            self.add_feature(description='density (mean)',
                             value=np.mean([dens_m1, dens_m2]))
            self.add_feature(description='density (var)',
                             value=np.var([dens_m1, dens_m2]))
            self.add_feature(description='density MO', value=dens_mo)
            diff_dens_m1_mo = dens_mo - dens_m1
            diff_dens_m2_mo = dens_mo - dens_m2
            self.add_feature(description='difference density (MO-M) (mean)',
                             value=np.mean([diff_dens_m1_mo, diff_dens_m2_mo]))
            self.add_feature(description='difference density (MO-M) (var)',
                             value=np.var([diff_dens_m1_mo, diff_dens_m2_mo]))

        # Bulk modulus.
        if 'bulk modulus' in self.selected_features:
            elas_m1 = data_m1['elasticity']
            elas_m2 = data_m2['elasticity']
            elas_mo = data_mo['elasticity']
            Kv_m1, Kv_m2 = Kv_mo = (elas_m1['K_Voigt'], elas_m2['K_Voigt'])
            if elas_mo:
                Kv_mo = elas_mo['K_Voigt']
            else:
                with MPRester(self.user_api_key) as m:
                    Kv_mo = m.get_data(id_mo, 
                            prop="elastic_moduli", 
                            data_type="pred")[0]['elastic_moduli']['K']
            self.add_feature(description='bulk modulus (mean)',
                             value=np.mean([Kv_m1, Kv_m2]))
            self.add_feature(description='bulk modulus (var)',
                             value=np.var([Kv_m1, Kv_m2]))
            self.add_feature(description='bulk modulus MO', value=Kv_mo)
            diff_Kv_m1_mo = Kv_mo - Kv_m1
            diff_Kv_m2_mo = Kv_mo - Kv_m2
            self.add_feature(
                description='difference bulk modulus (MO-M) (mean)',
                value=np.mean([diff_Kv_m1_mo, diff_Kv_m2_mo]))
            self.add_feature(
                description='difference bulk modulus (MO-M) (var)',
                value=np.var([diff_Kv_m1_mo, diff_Kv_m2_mo]))

        # Shear modulus.
        if 'shear modulus' in self.selected_features:
            elas_m1 = data_m1['elasticity']
            elas_m2 = data_m2['elasticity']
            elas_mo = data_mo['elasticity']
            Gv_m1, Gv_m2 = (elas_m1['G_Voigt'], elas_m2['G_Voigt'])
            if elas_mo:
                Gv_mo = elas_mo['G_Voigt']
            else:
                with MPRester(self.user_api_key) as m:
                    Gv_mo = m.get_data(id_mo, 
                            prop="elastic_moduli", 
                            data_type="pred")[0]['elastic_moduli']['G']
            self.add_feature(description='shear modulus (mean)',
                             value=np.mean([Gv_m1, Gv_m2]))
            self.add_feature(description='shear modulus (var)',
                             value=np.var([Gv_m1, Gv_m2]))
            self.add_feature(description='shear modulus MO', value=Gv_mo)
            diff_Gv_m1_mo = Gv_mo - Gv_m1
            diff_Gv_m2_mo = Gv_mo - Gv_m2
            self.add_feature(
                description='difference shear modulus (MO-M) (mean)',
                value=np.mean([diff_Gv_m1_mo, diff_Gv_m2_mo]))
            self.add_feature(
                description='difference shear modulus (MO-M) (var)',
                value=np.var([diff_Gv_m1_mo, diff_Gv_m2_mo]))

        # Poissons Ratio.
        if 'poisson ratio' in self.selected_features:
            elas_m1 = data_m1['elasticity']
            elas_m2 = data_m2['elasticity']
            elas_mo = data_mo['elasticity']
            pois_m1, pois_m2, pois_mo = (elas_m1['poisson_ratio'],
                                         elas_m2['poisson_ratio'],
                                         elas_mo['poisson_ratio'])
            self.add_feature(description='poisson ratio (mean)',
                             value=np.mean([pois_m1, pois_m2]))
            self.add_feature(description='poisson ratio (var)',
                             value=np.var([pois_m1, pois_m2]))
            self.add_feature(description='poisson ratio MO', value=pois_mo)
            diff_pois_m1_mo = pois_mo - pois_m1
            diff_pois_m2_mo = pois_mo - pois_m2
            self.add_feature(
                description='difference poisson ratio (MO-M) (mean)',
                value=np.mean([diff_pois_m1_mo, diff_pois_m2_mo]))
            self.add_feature(
                description='difference poisson ratio (MO-M) (var)',
                value=np.var([diff_pois_m1_mo, diff_pois_m2_mo]))

        # Universal anisotropy.
        if 'anisotropy' in self.selected_features:
            elas_m1 = data_m1['elasticity']
            elas_m2 = data_m2['elasticity']
            elas_mo = data_mo['elasticity']
            u_ani_m1, u_ani_m2, u_ani_mo = (elas_m1['universal_anisotropy'],
                                            elas_m2['universal_anisotropy'],
                                            elas_mo['universal_anisotropy'])
            el_ani_m1, el_ani_m2, el_ani_mo = (elas_m1['elastic_anisotropy'],
                                               elas_m2['elastic_anisotropy'],
                                               elas_mo['elastic_anisotropy'])
            self.add_feature(description='universal anisotropy (mean)',
                             value=np.mean([u_ani_m1, u_ani_m2]))
            self.add_feature(description='universal anisotropy (var)',
                             value=np.var([u_ani_m1, u_ani_m2]))
            self.add_feature(description='universal anisotropy MO',
                             value=u_ani_mo)
            diff_u_ani_m1_mo = u_ani_mo - u_ani_m1
            diff_u_ani_m2_mo = u_ani_mo - u_ani_m2
            self.add_feature(
                description='difference universal anisotropy (MO-M) (mean)',
                value=np.mean([diff_u_ani_m1_mo, diff_u_ani_m2_mo]))
            self.add_feature(
                description='difference universal anisotropy (MO-M) (var)',
                value=np.var([diff_u_ani_m1_mo, diff_u_ani_m2_mo]))

            # Elastic anisotropy.
            self.add_feature(description='elastic anisotropy (mean)',
                             value=np.mean([el_ani_m1, el_ani_m2]))
            self.add_feature(description='elastic anisotropy (var)',
                             value=np.var([el_ani_m1, el_ani_m2]))
            self.add_feature(description='elastic anisotropy MO',
                             value=el_ani_mo)
            diff_el_ani_m1_mo = el_ani_mo - el_ani_m1
            diff_el_ani_m2_mo = el_ani_mo - el_ani_m2
            self.add_feature(
                description='difference elastic anisotropy (MO-M) (mean)',
                value=np.mean([diff_el_ani_m1_mo, diff_el_ani_m2_mo]))
            self.add_feature(
                description='difference elastic anisotropy (MO-M) (var)',
                value=np.var([diff_el_ani_m1_mo, diff_el_ani_m2_mo]))

        # Spacegroup.
        if 'spacegroup' in self.selected_features:
            spacegroup_m1 = data_m1['spacegroup']['number']
            spacegroup_m2 = data_m2['spacegroup']['number']
            spacegroup_mo = data_mo['spacegroup']['number']
            self.add_feature(description='Spacegroup M1', value=spacegroup_m1)
            self.add_feature(description='Spacegroup M2', value=spacegroup_m2)
            self.add_feature(description='Spacegroup MO', value=spacegroup_mo)

        print("Fingerprint for " + self.label + " completed.")

    # 기본적으로 ID를 명시하지 않은 경우에 대해서, ID를 명시할 수 있도록 하는 변수가 된다. 예를 들어 HfO2면 거기서 Hf => 여러가지 구조가 나올거임  
    def _find_id(self, compound): 
        """ Find Materials Project ID for a given compound.

            Parameters
            ----------
            compound: str
                Compound formula.  Examples: ``'Li2O'``,
                ``'AlLiO2'``, ``'Mg2SiO4'``.
            user_api: str
                Materials Project users API.

            Returns
            -------
            id_compound: str
                Materials Project compound ID.

        """
        # 먼저 MPRester을 사용해서 검색을 진행한다. elasticity,e_above_hull등을 통해서 검색을 진행한다 => ex) Hf의 여러가지 상태가 존재함 
        
        with MPRester(self.user_api_key) as m: 
            info_MOs = m.get_entries(compound, inc_structure='final',
                                        property_data=['elasticity',
                                                       'e_above_hull',
                                                       'Correction'],
                                        sort_by_e_above_hull=True)
        # 불러온 엔트리는 List or Dictionary의 자료 구조를 가질 것이고. 
        # 불러온 모든 엔트리에 대해서 반복 진행. 일단 e_above_hull 이 가장 낮은것이 아래에 배열이 될 것이고, 이를 통해서 에너지를 얻어낸다. 
        # 에너지가 가장 낮고, 그리고 elasticity가 존재하는 material을 하나 불러올 수 있도록 한다. 
            for i in range(len(info_MOs)):
                id_compound = info_MOs[i].__dict__['entry_id']
                elasticity = m.get_data(id_compound)[0]['elasticity']
                if elasticity:
                    break
            if not elasticity:
                id_compound = info_MOs[0].__dict__['entry_id']
        # 해당 id를 return하면서 종료한다. 
        return id_compound
    
    # parameter로 받은 description과 value를 바탕으로, feature에 디렉토리 요소 추가 
    # self.label의 경우에는, MOformula_MOid 의 형태를 가진다. 
    def add_feature(self, description, value):
        """
        Parameters
        ----------
        description: str
            Description of the property to append (e.g. 'Formation energy').
        value: float
            Numerical value for a given property.

        Returns
        -------
        Adds a feature to the fingerprint (stored in self.fp).
        """

        self.fp[self.label]['features'].update({description: value})
    
    # 얘는 위에꺼랑 거의 완전히 똑같다고 할 수 있음 
    def add_target_feature(self, description, value):
        """
        Parameters
        ----------
        description: str
            Description of the property to be appended (e.g. 'Reduction
            temperature').
        value: float
            Numerical value for a given property.

        Returns
        -------
        Adds a target feature to the fingerprint (stored in self.fp).
        Note: In this case the properties and values will be only used for
        training the model (commonly know as train_y).
        """
        self.fp[self.label]['target_features'].update({description: value})

    def _get_number_of_atoms_element(self, atoms, symbol='O'):
        n_atom = 0
        for atom in atoms:
            if atom.symbol == symbol:
                n_atom += 1
        return n_atom

    # atoms라는 list에 있는 각각의 요소에 대해서, atom.symbol에 대해서 추출. 이거를 제대로 알려면 ASE를 알아야할듯 
    # 중복되는 원소를 제거한다. 뭐 HfO2가 들어오면 Hf랑 O만 symbols에 대해서 남는 형태인거지
    # 최대공약수를 통해서, 
    def _get_atoms_per_unit_formula(self, atoms):
        symbols = []
        for atom in atoms:
            symbols.append(atom.symbol)
        a = np.unique(symbols, return_counts=True)[1] # HfO2가있으면, [0]에는 ['Hf', 'O']가, [1]에는 [1 , 2] 가 저장되어 있는 형태이다. 
        return reduce(gcd, a)

    def _get_number_of_species(self, atoms):
        symbols = []
        for atom in atoms:
            symbols.append(atom.symbol)
        a = np.unique(symbols)
        return len(a), a
    
    # feature을 바탕으로, feature_name을 명시하지 않은 경우에 대해서는 feature_name을 명시하는 결과가 나온다. 
    def get_feature_value(self, feature_name):
        species = list(self.fp.keys())
        feature_value = []
        for i in species:
            feature = self.fp[i]['features'][feature_name]
            feature_value.append([feature])
        return feature_value

    def get_features_values(self):
        species = list(self.fp.keys())
        features_names = list(self.fp[species[0]]['features'].keys())
        features_values = []
        for i in species:
            features_i = []
            for j in features_names:
                features_i.append(self.fp[i]['features'][j])
            features_values.append(features_i)
        val_shape = np.shape(features_values)
        features_values = np.reshape(features_values,
                                     (val_shape[0], val_shape[1]))
        return features_values

    def get_target_features_values(self, target_feature='dS0_expt'):
        species = list(self.fp.keys())
        target_features_values = []
        for i in species:
            feature = self.fp[i]['target_features'][target_feature]
            target_features_values.append([feature])
        val_shape = np.shape(target_features_values)
        target_features_values = np.reshape(target_features_values,
                                            (val_shape[0], -1))
        return target_features_values
   
    #   
    def get_labels(self):
        """
        Returns the list of species (labelled), e.g. CaO-mp-2605.
        """
        return list(self.fp.keys())
    
    #  species의 feature을 return 
    def get_features_names(self):
        """
        Returns a list containing the names of the features, e.g. formation
        energy (kJ/mol).
        """
        species = list(self.fp.keys())
        features_names = list(self.fp[species[0]]['features'].keys())
        return features_names
    
    # List of targer feature을 return 
    def get_target_features_names(self):
        """
        Returns the list of target features. These features are
        user-defined and must be included with the add_target_features
        function.
        """
        species = list(self.fp.keys())
        features_names = list(self.fp[species[0]]['target_features'].keys())
        return features_names

    # Fingerprint class를 json 파일로 저장한다.
    def dump_set(self, filename='fingerprint.json'):
        """
        Parameters
        ----------
        filename: str
            Name of the file to save the generated Fingerprint class.

        Returns
        -------
        Saves the whole Fingerprint class into a json file.
        """
        self.label = 0.0  # Remove label to prevent confusion.
        fp_dict = self.__dict__
        del fp_dict['user_api_key']  # Important! Don't store API key.
        with open(filename, 'w') as fp:
            json.dump(fp_dict, fp)

    # json file을 load한다. 
    def load_set(self, filename): 
        """
        Parameters
        ----------
        filename: str
            Name of the file to load (json format).

        Returns
        -------
        Load a json file containing a previously saved Fingerprint (see
        dump_set function).
        """
        with open(filename, 'r') as fp:
            self.__dict__ = json.load(fp)
