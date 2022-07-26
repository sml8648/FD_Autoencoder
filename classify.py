import os, sys
p = os.path.abspath('..')
sys.path.insert(1,p)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from model.AutoEncoder import Model_selector
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import shap

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import lines

class classify:

    def __init__(self, date, model_name='./CheckPoint/2022-07-18 Autoencoder_deep_layer'):

        self.model_list = ["... list of columns "]

        base_set = set(self.model_list)

        df = pd.read_pickle('../Data/'+str(date)+'.pkl')

        # Model load
        state_dict = torch.load(model_name)

        self.Dt = df['dt'].reset_index(drop=True)
        self.custNo = df['custNo'].reset_index(drop=True)
        self.final_df = df.drop(['dt','custNo'],axis=1)
        self.final_df = self.final_df.reset_index(drop=True)

        # Model select
        self.model = Model_selector(
                                state_dict['model']['encoder.0.weight'].size(1),
                                state_dict['config'].model_name,
                                state_dict['config'].activation,
                                state_dict['config'].noise)

        self.model.load_state_dict(state_dict['model'])
        # columns을 맞춰주기 위한 부가적인 작업....

        current_set = set(list(self.final_df.columns))

        # 없는 columns은 추가해줌
        for each in (base_set - current_set):
            self.final_df[each] = 0

        # 필요없는 columns은 제거해줌
        for each in (current_set - base_set):
            self.final_df.drop([each],axis=1,inplace=True)

        # columns 배열 재배치
        self.final_df = self.final_df[self.model_list]

    def make_df(self):

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(self.final_df)
        data_tensor = torch.from_numpy(scaled_data).float()

        Loss_function = nn.MSELoss()

        tensor_loss = []

        for each in data_tensor:
            result = self.model(each)
            each_loss = Loss_function(result, each)

            each_loss = np.log(each_loss.detach().numpy())

            tensor_loss.append(each_loss)

        events_df = pd.DataFrame({
            'log_mse':tensor_loss,
        })

        return events_df

    def draw_graph(self,events_df,threshold):

        plt.figure(figsize=(20,10))
        plot = sns.lineplot(x=(events_df.index), y=(events_df.log_mse))
        line = lines.Line2D(
            xdata=np.arange(0, len(events_df)),
            ydata=np.full(len(events_df), threshold),
            color='#CC2B5E',
            linewidth=1.5,
            linestyle='dashed'
        )

        plot.add_artist(line)
        plt.show()

    def main(self,threshold,only_target=False):

        # Return data from model
        result = self.make_df()

        # Filter based on threshold
        target_list = self.custNo.loc[list(result[result.log_mse > threshold].index.values)]

        # Make new df
        self.final_df['Label'] = 0

        self.final_df.loc[list(target_list.index), 'Label'] = 1
        self.final_df = self.final_df.fillna(0)

        # Split df and label
        Label = self.final_df.Label.astype('int32')
        self.final_df = self.final_df.iloc[:,:-1]

        if only_target:
            return target_list

        else:
            return self.final_df, Label

    def fetch_top_k(self,threshold,k=10):

        """
        threshold를 기준으로 각각의 거래건을 1 or 0로 라벨링함

        라벨링한 데이터를 기준으로 lgbm으로 학습을 시킴

        학습을 시킨 후 shap을 통과 시켜서 어떠한 feature가

        threshold를 넘게했는지에 대한 value를 리턴하는 함
        """

        df, Label = self.main(threshold=threshold)

        Dt = self.Dt
        custNo = self.custNo

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        lgb = LGBMClassifier(n_estimators=200)
        lgb.fit(scaled_data, Label)

        explainer = shap.TreeExplainer(lgb)
        shap_values = explainer.shap_values(scaled_data)

        target_list = list(Label[(Label == 1)].index)

        test_result = shap_values[1][target_list,:]

        tmp_df = {
            'custNo': [],
            'Dt': [],
            'F1' : [],
            'F1_value' : [],
            'F2' : [],
            'F2_value' : [],
            'F3' : [],
            'F3_value' : [],
            'F4' : [],
            'F4_value' : [],
            'F5' : [],
            'F5_value' : [],
            'F6' : [],
            'F6_value' : [],
            'F7' : [],
            'F7_value' : [],
            'F8' : [],
            'F8_value' : [],
            'F9' : [],
            'F9_value' : [],
            'F10'  :[],
            'F10_value' : []
        }

        for cn, dt, each in zip(custNo[target_list], Dt[target_list],test_result):

            tmp_df['custNo'].append(cn)
            tmp_df['Dt'].append(dt)

            # Filter top 10 important value
            each_df = pd.DataFrame(each, index=df.columns).sort_values(by=[0],ascending=False).head(k)

            for idx, (index, value) in enumerate(zip(each_df.index, each_df.values)):
                
                column_1 = 'F'+str(idx+1)
                tmp_df[column_1].append(index)

                columns_2 = 'F'+str(idx+1)+'_value'
                tmp_df[columns_2].append(value[0])

        Total_df = pd.DataFrame(tmp_df)

        return Total_df

if __name__ == '__main__':

    clf = classify('220401')
