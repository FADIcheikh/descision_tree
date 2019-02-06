# coding: utf-8
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import collections

df = pd.read_csv("D:\Data_Minig\seance7_decTree_cart\\breast.csv",sep =';',header = 0)
#split dataset into explicatives vars and target
explicative =df.drop(['classe'],axis=1)
names = explicative.columns
target =df['classe']
#set 25% for test
X_train, X_test, y_train, y_test = train_test_split(explicative,target, test_size=0.25, random_state=0)
#training
dtree=DecisionTreeClassifier(criterion='gini')
dtree.fit(X_train,y_train)
#score =  94% accurancy
print dtree.score(X_test, y_test)
#test
pred =dtree.predict([[2,1,1,1,2,1,1,1,1]])
print pred
#Visualisation

dot_data = tree.export_graphviz(dtree,
                                feature_names=names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]

graph.write_png('tree.png')
