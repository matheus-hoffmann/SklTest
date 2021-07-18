import warnings
from skl_regressor_test.utils import *
from skl_regressor_test.skl_regressor_test import *

df, input_data, output_data = read_data("80_19_1.csv")

SklRegressors = SklRegressorTest(m_input=input_data,
                                 m_output=output_data,
                                 m_train_percentage=0.8)

warnings.filterwarnings("ignore")

SklRegressors.set_desired_models(models="all")
SklRegressors.test_all(method="all",
                       n_random_states=2,
                       rkf_cv_n_splits=2,
                       rkf_cv_n_repeats=1,
                       n_rand_iter=1)
SklRegressors.write_log(path="",
                        filename="skl_regressor_test_summary")
