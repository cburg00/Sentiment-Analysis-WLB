from datasets import load_dataset
from textblob.classifiers import NaiveBayesClassifier

dataset = load_dataset('kmrmanish/Employees_Reviews_Dataset', split='train')
data_dict = dataset.to_dict()
i = 0
while i < len(data_dict['work_life_balance']):
    if data_dict['work_life_balance'][i] is None:
        for column in data_dict:
            data_dict[column].pop(i)
    else:
        i += 1
text = list(map(' '.join, zip(list(map(str, data_dict['Likes'])), list(map(str, data_dict['Dislikes'])))))
train = list(zip(text, list(map(lambda score: 'pos' if score > 3 else 'neg', data_dict['work_life_balance']))))
cl = NaiveBayesClassifier(train)
print(cl.classify("I love working here"))