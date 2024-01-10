import pymap3d as pm
import numpy as np
from typing import Tuple
from scipy.constants import speed_of_light
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 261.6261890872212 228.65371205245526
# 36.00307172 -78.93744816
class CBRSUtils:
    BS_PCI20_GPS_COORDINATE = (36.00307128210248, -78.93706070613855)

    """
    A utility class for CBRS project related function.
    """

    @staticmethod
    def gps2local(latitude: float, longitude: float, top_left_latitude: float = 36.005429554748495,
                  top_left_longitude: float = -78.93998431793196, arr_size: int = 128) -> Tuple[float]:
        """
        Convert the GPS coordinate to local coordinate, based on the top_left corner GPS coordinates.
        GPS Coordinate format: Decimal degrees (DD) in float.
        :param latitude:
        :param longitude:
        :param top_left_latitude:
        :param top_left_longitude:
        :return: Local coordinates where (0,0) is the top_left corner.
        """
        # top_left = (36.005429554748495, -78.93998431793196)
        # ell_clrk66 = pm.Ellipsoid('clrk66')
        wgs84 = pm.Ellipsoid.from_name('wgs84')
        res = pm.geodetic2ned(top_left_latitude, top_left_longitude, 1, latitude, longitude, 1, ell=wgs84)
        #print(res)
        return res[0], -res[1]

    @staticmethod
    def local2gps(n, e, lat0=36.005429554748495, lon0=-78.93998431793196):
        """

        :param n:
        :param e:
        :param d:
        :param lat0:
        :param lon0:
        :param h0:
        :return:
        """
        wgs84 = pm.Ellipsoid.from_name('wgs84')
        lat, lon, h = pm.ned2geodetic(n, -1 * e, 0.015, lat0, lon0, 1, ell=wgs84)
        return lat, lon

    @staticmethod
    def uma_los(d3d, d2d, dbp, fc, h_b, h_t):
        """
        Compute the path loss in LOS, Urban, Marco setting.
        :param d3d: 3D distance
        :param d2d: 2D distance
        :param dbp: Breakpoint distance
        :param fc: frequency in GHz
        :param h_b: height of basestation
        :param h_t: height of UT
        :return:
        """
        # 38.901 UMa LOS
        PL1 = 28 + 22 * np.log10(d3d) + 20 * np.log10(fc)
        PL2 = 28 + 40 * np.log10(d3d) + 20 * np.log10(fc) - 9 * np.log10(dbp ** 2 + (h_b - h_t) ** 2)
        # PL = np.zeros((d3d.shape))
        PL = PL2  # Default pathloss
        PL[(np.greater_equal(d2d, 10) & np.less_equal(d2d, dbp))] = PL1[(np.greater_equal(d2d, 10) & np.less_equal(d2d,
                                                                                                                   dbp))]  # Overwrite if distance is greater than 10 meters or smaller than dbp
        return PL

    @staticmethod
    def uma_nlos(d3d, d2d, dbp, fc, h_b, h_t):
        """
        Compute the path loss in NLOS, Urban, Marco setting.
        :param d3d: 3D distance
        :param d2d: 2D distance
        :param dbp: Breakpoint distance
        :param fc: frequency in GHz
        :param h_b: height of basestation
        :param h_t: height of UT
        :return:
        """
        # 38901 UMa NLOS
        PL_nlos = 13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(fc) - 0.6 * (h_t - 1.5)
        PL = np.zeros((d3d.shape))
        PL = np.maximum(CBRSUtils.uma_los(d3d, d2d, dbp, fc, h_b, h_t), PL_nlos)
        return PL

    @staticmethod
    def pathloss_38901(distance, frequency, h_bs=30, h_ut=1.5):
        """
        Simple path loss model for computing RSRP based on distance.
        :param distance: distance between basestation and UE
        :param frequency: frequency in GHz
        :param h_bs: height of basestation
        :param h_ut: height of UT
        :return:
        """
        # Constants
        fc = frequency
        h_b = h_bs  # 30 meters
        h_t = h_ut  # 1.5

        # 2D distance
        d2d = distance

        # 3D distance
        h_e = h_b - h_t  # effective height
        d3d = np.sqrt(d2d ** 2 + h_e ** 2)

        # Breakpoint distance
        dbp = 4 * h_b * h_t * fc * 10e8 / speed_of_light

        loss = CBRSUtils.uma_nlos(d3d, d2d, dbp, fc, h_b, h_t)
        return loss

    @staticmethod
    def pathloss_friis_free_space_model(d, h_b, h_r, f):
        """
        Compute the pathloss in dB, Friis model.
        :param d: is the link distance in km and 
        :param f: is the transmission frequency in MHz
        :return: pathloss in dB
        """
        return 32.45 + 20 * np.log10(d) + 20 * np.log10(f)

    @staticmethod
    def pathloss_ericsson_model(d, h_b, h_r, f):
        """
        Compute the pathloss in dB, Ericsson model.
        :param h_b: is the height of the base station in meters,
        :param h_r: is the height of the receiver in meters,
        :param d: is the link distance in km,
        :param f: is the transmission frequency in MHz, 
        :return: pathloss in dB
        """

        def g(freq):
            return 44.49 * np.log10(freq) - 4.78 * (np.log10(freq) ** 2)

        # for urban environments, (a0, a1, a2, a3) = (36.2, 30.2, 12, 0.1)
        a0 = 36.2
        a1 = 30.2
        a2 = 12
        a3 = 0.1
        line1 = a0 + a1 * np.log10(d) + a2 * np.log10(h_b)
        line2 = + a3 * np.log10(h_b) * np.log10(d)
        line3 = - 3.2 * ((np.log10(11.75 * h_r)) ** 2) + g(f)
        return line1 + line2 + line3

    @staticmethod
    def rmse(y_actual, y_predicted):
        """
        Compute the RMSE between two vectors.
        :param y_actual:
        :param y_predicted:
        :return: RMSE of the two vectors.
        """
        return mean_squared_error(y_actual, y_predicted, squared=False)

    @staticmethod
    def generate_linear_regression_result(prediction, ground_truth, title="", print_result=True):
        """
        :param prediction:  predicted value
        :param ground_truth: ground truth value
        :param title:
        :param print_result:
        :return:
        """
        assert ground_truth.shape == prediction.shape, "ground_truth and prediction shape mismatch "
        print()
        print("++++++++++++++", title, "++++++++++++++")
        X = []
        y = []
        for row in range(prediction.shape[0]):
            for col in range(prediction.shape[1]):
                if ground_truth[row, col] != -160:
                    X.append(prediction[row, col])
                    y.append(ground_truth[row, col])
        X = np.array(X).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.85, random_state = 42)
        reg = LinearRegression().fit(X_train, y_train)

        print("score", reg.score(X, y))
        print("coef", reg.coef_[0])

        print("intercept", reg.intercept_)

        ori_rmse = mean_squared_error(y, X, squared=False)
        print("rmse ", ori_rmse)

        fitted_rmse = mean_squared_error(y, X * reg.coef_ + reg.intercept_, squared=False)
        print("rmse after linear regression ", fitted_rmse)

        ori_mae = mean_absolute_error(y, X)
        fitted_mae = mean_absolute_error(y, X * reg.coef_ + reg.intercept_)
        print("mae ", ori_mae)
        print("mae after linear regression ", fitted_rmse)
        print()
        return reg.coef_[0], reg.intercept_, ori_rmse, fitted_mae




if __name__ == "__main__":
    print(CBRSUtils.gps2local(CBRSUtils.BS_PCI20_GPS_COORDINATE[0], CBRSUtils.BS_PCI20_GPS_COORDINATE[1]))
