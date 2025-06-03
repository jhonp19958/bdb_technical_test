import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from catboost import CatBoostRegressor
from typing import Dict, List


class MLSalaryEnvironment(gym.Env):
    def __init__(
        self,
        actions: List[Dict],
        data: pd.DataFrame,
        target_variable: str,
        step_variable: str,
        cat_features: List[int],
    ):
        """
        Class for the environment used to train a ML model to predict salaries.

        Parameters:
        actions (List[Dict]): List of predefined actions to take.
        data (pd.DataFrame): A pandas DataFrame containing the dataset.
        target_variable (str): The name of the target variable column.
        step_variable (str): The name of the column representing years.
        cat_features (List[int]): List of indices of categorical features.
        """
        super(MLSalaryEnvironment, self).__init__()
        self.data = data
        self.step_variable = step_variable
        self.target_variable = target_variable
        self.cat_features = cat_features
        self.years = np.sort(self.data[step_variable].unique())
        self.current_year_idx = 0
        # Depende de la cantidad de acciones disponibles
        self.action_space = spaces.Discrete(len(actions))
        self.actions = actions
        # Una observación es el promedio, la desviación y cantidad de registros
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=float
        )

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.

        Parameters:
        seed: Optional seed for random number generation.
        options: Optional information used for environment registry.

        Returns:
        tuple: A tuple containing the initial observation and an empty dict.
        """
        super().reset(seed=seed)
        self.current_year_idx = 0
        return self.get_observation(), {}

    def get_observation(self):
        """
        Returns the current observation based on the current year index.
        If the current year index is -1, it returns a zero vector.
        If its is valid, it computes the mean, std, and number of records
        for the years up to the current year.

        Returns:
        np.ndarray: An array containing the mean, std, and number of records.
        """
        if self.current_year_idx >= len(self.years):
            return np.zeros(3)
        year = self.years[self.current_year_idx]
        salaries = self.data[self.target_variable][
            self.data[self.step_variable] <= year
        ]

        return np.array([salaries.mean(), salaries.std(), len(salaries)])

    def get_action(self, action_to_execute: Dict):
        """
        Returns the model initialized to execute based on the provided action.

        Parameters:
        action_to_execute (Dict): The model class and its args.

        Returns:
        base_model (object): An instance of the base model initialized.
        """
        base_model = action_to_execute.get("base_model")
        kwargs = action_to_execute.get("kwargs", {})
        if base_model is None:
            raise ValueError("Action must contain a 'base_model' key.")
        elif base_model == CatBoostRegressor:
            kwargs["cat_features"] = self.cat_features
            kwargs["verbose"] = False
        if len(kwargs) == 0:
            return base_model()
        else:
            return base_model(**kwargs)

    def step(self, action: int):
        """
        Executes the action in the environment.

        Parameters:
        action (int): The index of the action to execute.

        Returns:
        tuple
            - observation (np.ndarray): The next observation after taking the action.
            - reward (float): The reward received after taking the action.
            - done (bool): Whether the episode has ended.
            - truncated (bool): Whether the episode was truncated.
            - info (dict) : Additional information, such as the model used for this step.
        """  # noqa: E501
        # De la lista de acciones, obtenemos la acción a ejecutar
        action_to_execute = self.actions[action]
        year = self.years[self.current_year_idx]

        # Filtrar los datos hasta el año actual
        data_year = self.data[self.data[self.step_variable] <= year]
        X = data_year.drop(columns=[self.target_variable, self.step_variable])
        y = data_year[self.target_variable]

        # Inicializar el modelo con los datos filtrados
        model = self.get_action(action_to_execute)

        # Crear OHE para las variables categóricas
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns
        if isinstance(model, CatBoostRegressor):
            # CatBoost no necesita OHE, ya que maneja variables categóricas
            pipeline_model = model
        else:
            if not categorical_cols.empty:
                # La primera acción significa que se está entrenando una RL
                # se elimina la primera categoría para evitar multicolinealidad
                drop = "first" if action == 0 else None
                ohe = OneHotEncoder(
                    drop=drop, sparse_output=False, handle_unknown="ignore"
                )
                # Crear pipeline con OHE si es necesario
                pipeline_model = Pipeline([("ohe", ohe), ("model", model)])
            else:
                # Si no hay variables categóricas, usar el modelo directamente
                pipeline_model = model

        # Realizar validación cruzada para evaluar el modelo
        cv_results = cross_validate(
            pipeline_model, X, y, scoring="neg_root_mean_squared_error", cv=3
        )
        # Menor RMSE es mejor, así que tomamos el negativo
        # (neg_root_mean_squared_error devuelve el RMSE negativo)
        reward = cv_results["test_score"].mean()

        # Entrenar el modelo completo con la acción seleccionada
        pipeline_model.fit(X, y)

        # Actualizar para siguiente step
        self.current_year_idx += 1
        done = self.current_year_idx == len(self.years)
        observation = self.get_observation()
        return (
            observation,
            reward,
            done,
            False,
            {year: pipeline_model},
        )
