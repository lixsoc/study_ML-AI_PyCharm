from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn import preprocessing as ppc

import numpy as np
import pandas as pd

def get_data(file_path: str):
    names = """
    age: continuous.
    workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    fnlwgt: continuous.
    education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    education-num: continuous.
    marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    sex: Female, Male.
    capital-gain: continuous.
    capital-loss: continuous.
    hours-per-week: continuous.
    native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    """

    names.split(':')
    print(names)

    # 1
    names = ['age','workclass','fnlwgt','education','education-num','marital-status',
             'occupation','relationship','race','sex','capital-gain','capital-loss',
             'hours-per-week','native-country','income']
    # 셰퍼레이터가 ','한글자에서 ', '두글자가 되면 패턴으로 바뀌어서 파이썬으로 엔진을 바꿔줘야 함.
    adult = pd.read_csv(file_path, header=None, names=names, sep=', ', engine='python')

    # 2
    # print(adult.values[:5, 0])
    # print(adult.values[:5, 1])
    # # =
    # print(adult['age'].values)

    # 각 컬럼의 데이터 타입
    adult.info

    x = [
        adult['age'].values, adult['fnlwgt'].values,
        adult['education-num'].values, adult['capital-gain'].values,
        adult['capital-loss'].values, adult['hours-per-week'].values
    ]
    x = np.int32(x)
    x = np.transpose(x)

    y = np.int32(adult['income'].values == '<=50K')

    return x, y

