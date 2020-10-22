import pandas as pd
import numpy as np
import xgboost
from os import path

def identification_0H(reading):
  dir = path.dirname(path.realpath(__file__))
  model_path = path.join(dir, '../../data/model/Identificacao_0H.json')
  threshold = 0.5

  model = xgboost.Booster()
  model.load_model(model_path)

  df = pd.DataFrame(
    data=[reading],
    columns=['Mes', 'Dia', 'Local', 'Precipitacao', 'PrecSum']
  )

  data = xgboost.DMatrix(data=df)
  result = model.predict(data)

  result = True if result > threshold else False

  return result

def prediction_12h(reading):
  # TODO: Implement 12h prediction model
  return None

def prediction_6h_a(reading):
  # TODO: Implement 6h 0mm prediction model
  return None

def prediction_6h_b(reading):
  # TODO: Implement 6h 1.2mm prediction model
  return None
