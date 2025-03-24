import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

class DecisionTree:

    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = self.import_file()
        self.transformed_data = self.transform(self.raw_data)


    def import_file(self):
        df = pd.read_excel(self.filepath, header=(0,1))
        df.rename(columns=str.strip, inplace=True)
        return df
    
    def transform(self, df):
        df2 = pd.DataFrame()

        df2.insert(0, 'WORK LIFE BALANCE', df['EMPLOYMENT STATUS']['WORK LIFE BALANCE'])
        df2.insert(1, 'TIME TO READ', df['WORKING STYLE']['WS1'])
        df2.insert(2, 'TIME TO WATCH TV', df['WORKING STYLE']['WS2'])
        df2.insert(3, 'SPEND TIME WITH FAMILY', df['WORKING STYLE']['WS12'])
        df2.insert(4, 'SOCIAL ACTIVITIES', df['WORKING STYLE']['WS22'])
        df2.insert(5, 'RELAX FOR 2HRS', df['WORKING STYLE']['WS26'])
        df2.insert(6, 'TIME FOR SLEEP 8HRS', df['WORKING STYLE']['WS27'])
        df2.insert(7, 'FLEXIBLE WORKING HOURS', df['INSTITUTIONAL EFFORTS']['IE1']) #Binary
        df2.insert(8, 'STRESS AND FRUSTRATION', df['WORK LIFE BALANCE']['WLB12'])

        return df2

    def train(self):
        '''train the model'''
        y = self.transformed_data.iloc[:, 0]
        X = self.transformed_data.iloc[:, 1:]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

        model = DecisionTreeClassifier(max_depth=5)
        model.fit(X_train, y_train)

        pred = model.predict(X_train)
        print(accuracy_score(y_train,pred))

        return model 

    def plot(self, model):
        '''Plot the tree'''
        fig, ax = plt.subplots(figsize=(5,5))
        tree.plot_tree(model, feature_names=self.transformed_data.columns, class_names=['Satisfied','Not satisfied'], filled=True)
        return fig

    def __str__(self):
        return self.transformed_data.to_string()