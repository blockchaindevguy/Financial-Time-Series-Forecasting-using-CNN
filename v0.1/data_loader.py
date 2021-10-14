import os
import pandas as pd
from utils import create_labels, download_financial_data
from technical_indicators import calculate_technical_indicators
import re
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from operator import itemgetter


class DataLoader:
    def __init__(self, symbol, data_folder="historical_data", output_folder="data", auxiliary_symbols=[]):
        self.symbol = symbol
        self.data_folder = data_folder
        self.data_path = data_folder+"/"+symbol+"/"+symbol+".csv"
        self.output_folder = output_folder
        self.start_col = 'open'
        self.end_col = 'eom_26'
        self.download_stock_data()
        self.df = self.create_dataframe()
        self.auxiliary_symbols = auxiliary_symbols
        if len(self.auxiliary_symbols) > 0:
            self.download_auxiliary_data()
            self.add_auxiliary_data_to_dataframe()
        self.feat_idx = self.feature_selection()
        self.one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
        self.one_hot_enc.fit(self.df['labels'].values.reshape(-1, 1))
        self.batch_start_date = self.df.head(1).iloc[0]["timestamp"]
        self.test_duration_years = 1
        print("{} has data for {} to {}".format(data_folder, self.batch_start_date,
                                                                 self.df.tail(1).iloc[0]['timestamp']))

    def create_dataframe(self):   # TODO rewrite
        if not os.path.exists(os.path.join(self.output_folder, "df_" + self.symbol+".csv")):
            df = pd.read_csv(self.data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            intervals = range(6, 27)  # 21
            calculate_technical_indicators(df, 'open', intervals)
            print("Saving dataframe...")
            df.to_csv(os.path.join(self.output_folder, "df_" + self.symbol+".csv"), index=False)
        else:
            print("Technical indicators already calculated. Loading...")
            df = pd.read_csv(os.path.join(self.output_folder, "df_" + self.symbol+".csv"))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)

        prev_len = len(df)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("Dropped {0} nan rows before label calculation".format(prev_len - len(df)))

        if 'labels' not in df.columns:
            df['labels'] = create_labels(df, 'close')
            prev_len = len(df)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            print("Dropped {0} nan rows after label calculation".format(prev_len - len(df)))
            if 'dividend_amount' in df.columns:
                df.drop(columns=['dividend_amount', 'split_coefficient'], inplace=True)
            df.to_csv(os.path.join(self.output_folder, "df_" + self.symbol + ".csv"), index=False)
        else:
            print("labels already calculated")

        print("Number of Technical indicator columns for train/test are {}".format(len(list(df.columns)[7:])))
        return df

    def download_stock_data(self):
        print("path to {} data: {}".format(self.symbol, self.data_path))
        parent_folder = os.sep.join(self.data_path.split(os.sep)[:-1])
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)
        if not os.path.exists(self.data_path):
            print('Downloading historical data of {}.'.format(self.symbol))
            download_financial_data(self.symbol, self.data_path)
        else:
            print("Historical data {} is already available on disk. Therefore not downloading.".format(
                self.symbol))

    def download_auxiliary_data(self):
        for symbol in self.auxiliary_symbols:
            symbol_path = self.data_path[:-11]+symbol+"/"+symbol+".csv"
            print("path to {} data: {}".format(symbol, symbol_path))
            parent_folder = os.sep.join(symbol_path.split(os.sep)[:-1])
            if not os.path.exists(parent_folder):
                os.makedirs(parent_folder)


            if not os.path.exists(symbol_path):
                print('Downloading auxiliary feature data')
                download_financial_data(symbol, symbol_path)
            else:
                print("Auxiliary data {} is already available on disk. Therefore not downloading.".format(symbol))

    def add_auxiliary_data_to_dataframe(self):
        for aux in self.auxiliary_symbols:
            aux_path = os.path.join(self.data_folder, aux, aux + ".csv")
            aux_output_path = os.path.join(self.output_folder, "df_" + aux + '.csv')

            if os.path.exists(aux_path):
                print('Merging dataframes from {} and auxiliary {}'.format(self.symbol, aux))
                if not os.path.exists(aux_output_path):
                    df = pd.read_csv(aux_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    intervals = range(6, 27)  # 21
                    calculate_technical_indicators(df, 'open', intervals)
                    df.to_csv(aux_output_path)
                else:
                    print("Technical indicators for auxiliary data {} already calculated. Will load from disk.".format(
                        aux))
                    df = pd.read_csv(aux_output_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                merged = self.df.merge(df, how='left', on='timestamp', suffixes=('', '_'+aux))
                if 'Unnamed: 0' in merged.columns:
                    merged = merged.drop(axis=1, columns=['Unnamed: 0'])
                self.df = merged.dropna()
            else:
                print("Auxiliary data {} is not available on disk. Therefore, it will be skipped.".format(aux))

        # reorder df so that labels and volume delta are at the end
        cols = self.df.columns.tolist()
        cols = [c for c in cols if c != "labels"]
        cols.append("labels")
        self.df = self.df[cols]
        output_path = os.path.join(self.output_folder, "df_" + self.symbol + '_aux.csv')
        self.df.to_csv(output_path)
        print("df_{}_aux written to disk.".format(self.symbol))

    def feature_selection(self):   # TODO rewrite

        def df_by_time_frame(start_date=None, years=5):  # TODO rewrite
            if not start_date:
                start_date = self.df.head(1).iloc[0]["timestamp"]

            end_date = start_date + pd.offsets.DateOffset(years=years)
            df_batch = self.df[(self.df["timestamp"] >= start_date) & (self.df["timestamp"] <= end_date)]
            return df_batch

        df_batch = df_by_time_frame(None, 10)
        list_features = list(df_batch.loc[:, self.start_col:self.end_col].columns)
        mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
        x_train = mm_scaler.fit_transform(df_batch.loc[:, self.start_col:self.end_col].values)
        y_train = df_batch['labels'].values
        num_features = 225  # should be a perfect square
        topk = 350
        select_k_best = SelectKBest(f_classif, k=topk)
        select_k_best.fit(x_train, y_train)
        selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)

        select_k_best = SelectKBest(mutual_info_classif, k=topk)
        select_k_best.fit(x_train, y_train)
        selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)

        common = list(set(selected_features_anova).intersection(selected_features_mic))
        print("common selected featues:" + str(len(common)) + ", " + str(common))
        if len(common) < num_features:
            raise Exception(
                'number of common features found {} < {} required features. Increase "topK"'.format(len(common),
                                                                                                    num_features))
        feat_idx = []
        for c in common:
            feat_idx.append(list_features.index(c))
        feat_idx = sorted(feat_idx[0:225])
        print(str(feat_idx))
        return feat_idx