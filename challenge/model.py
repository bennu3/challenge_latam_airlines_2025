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

        data = data.copy()

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


        if target_column:
            if target_column in data.columns:
                target = data[[target_column]].copy()
            else:
                if 'Fecha-O' not in data.columns or 'Fecha-I' not in data.columns:
                    raise ValueError("Se requieren columnas Fecha-O y Fecha-I para calcular el target")
                data['min_diff'] = data.apply(self._get_min_diff, axis=1)
                data[target_column] = np.where(data['min_diff'] > 15, 1, 0)
                target = data[[target_column]].copy()
            return features, target
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
        self._model = LogisticRegression(class_weight="balanced")
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
            raise ValueError("El modelo no ha sido entrenado aun.")

        predictions = self._model.predict(features)
        return predictions.tolist()
