import os
from catlearn.regression.gaussian_process import GaussianProcess # GPR 을 통해서 ML을 진행할 것이므로 

from .fingerprint import Fingerprint # 같은 디렉토리 내에 존재하는 Fingerprint 파일을 참고 

__author__ = "Jose A. Garrido Torres, Alexander Urban"
__email__ = "aurban@atomistic.net"
__date__ = "2021-03-18"
__version__ = "1.0"


class Ellingham():

    def __init__(self, USER_API_KEY, id_mo, id_m1='', id_m2='',
                 id_oxygen='mp-12957', p_O2=0.0, p_CO=0.0, p_CO2=0.0):
        """
        Args:
          USER_API_KEY: Personal API key for the Materials Project (MP)
            database (https://materialsproject.org)
          id_mo: Chemical formula or MP ID of the metal oxide
          id_m1, id_m2: MP IDs of the metal species; will be determined
            automatically if not provided
          id_oxygen: MP ID of the O2 entry to be used as reference
          p_O2, p_CO, p_CO2: O2, CO, and CO2 partial pressures

        """
        # 3성분계까지 가능한듯. 일단은 Oxigen bases로 진행되는것을 파악
        
        # API 키를 바탕으로, train_fp를 생성하고, 
        self.user_api_key = USER_API_KEY
        train_fp = Fingerprint(USER_API_KEY=self.user_api_key)
      
        # Training set를 불러온다 
        path_mod = os.path.dirname(__file__)
        train_fp.load_set(path_mod + '/trainingset_ellingham_08June2020.json')
        
        # Test set 생성 
        test_fp = Fingerprint(USER_API_KEY=self.user_api_key)
        
        # training 데이터 추출. 주어진 input value와, 정답 label인 train_y를 매칭하고 생성하는 과정인듯 
        # Get training data
        train_x = train_fp.get_features_values()
        train_y = train_fp.get_target_features_values(
            target_feature='dS0_expt')

        # Build GP model.
        kernel = [
            {'type': 'gaussian', 'width': 1., 'scaling': 1.},
            {'type': 'linear', 'scaling': 1., 'constant': 1.}]

        # Train the GP model. 이전에 만든 input value를 바탕으로 
        gp = GaussianProcess(kernel_list=kernel, regularization=1e-3,
                             regularization_bounds=(1e-5, 1e-1),
                             train_fp=train_x, train_target=train_y,
                             optimize_hyperparameters=False,
                             scale_data=True)
        gp.optimize_hyperparameters(global_opt=False,
                                    algomin='TNC',
                                    eval_jac=True)

        # Get test data.
        test_fp.extract_mp_features(id_mo=id_mo,
                                    id_m1=id_m1,
                                    id_m2=id_m2,
                                    id_oxygen=id_oxygen,
                                    selected_features='ellingham')
        test_x = test_fp.get_features_values()

        # Get predictions.
        prediction = gp.predict(test_fp=test_x, uncertainty=False)
        pred = prediction['prediction'][0][0]

        # Store data.
        label = list(test_fp.fp.keys())[0]
        self.balanced_reaction = test_fp.fp[label]['balanced reaction']
        self.dH0 = test_fp.get_feature_value('formation energy (kJ/mol)')[0][0]
        self.dS0 = pred
    
    # 여기부터는 주어진 데이터를 바탕으로 return 
    def get_dG0(self, T):
        dG0 = self.dH0 + T * self.dS0
        return dG0

    def get_dH0(self):
        return self.dH0

    def get_dS0(self):
        return self.dS0

    def get_balanced_reaction(self):
        return self.balanced_reaction
