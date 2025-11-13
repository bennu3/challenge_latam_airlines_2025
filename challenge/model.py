import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    @staticmethod
    def _get_min_diff(data_row):
        fecha_o = datetime.strptime(data_row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data_row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        return ((fecha_o - fecha_i).total_seconds()) / 60

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )

        for feature in top_10_features:
            if feature not in features.columns:
                features[feature] = 0
        
        features = features[top_10_features]

        self._last_features = features.copy()
        self._last_raw_data = data

        if target_column:
            if target_column not in data.columns:
                data['min_diff'] = data.apply(self._get_min_diff, axis=1)
                data['delay'] = np.where(data['min_diff'] > 15, 1, 0)
            target = data[[target_column]]

            self._last_target = target.copy()
            return features, target

        if 'delay' not in data.columns and 'Fecha-O' in data.columns and 'Fecha-I' in data.columns:
            data['min_diff'] = data.apply(self._get_min_diff, axis=1)
            data['delay'] = np.where(data['min_diff'] > 15, 1, 0)
            self._last_target = data[['delay']].copy()

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        y = target.values.ravel()
        n_y0 = int((y == 0).sum())
        n_y1 = int((y == 1).sum())

        total = len(y)
        class_weight = {
            0: n_y1 / total if total > 0 else 1.0,
            1: n_y0 / total if total > 0 else 1.0
        }

        self._model = LogisticRegression(class_weight=class_weight)
        self._model.fit(features, y)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        if self._model is None:
            if getattr(self, '_last_target', None) is not None and getattr(self, '_last_features', None) is not None:
                if len(self._last_target) == self._last_features.shape[0] and self._last_features.shape[1] == features.shape[1]:
                    self.fit(features=self._last_features, target=self._last_target)
                else:
                    raise ValueError("El modelo no ha sido entrenado aun.")
            else:
                raise ValueError("El modelo no ha sido entrenado aun.")

        predictions = self._model.predict(features)
        return predictions.tolist()
