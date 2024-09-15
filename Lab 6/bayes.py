# %% [markdown]
# ### Apply Naive Bayes classifier on one of the files in senseval dataset (for example, for the word interest). Use 90% of the phrases for training and 10% for testing the classifier. The phrases should be taken in a random order (shuffle the phrases before training and testing). For the testing set print the predicitons of the classifier and the correct labels from the corpus and also print "true" if they are the same, and "false" if they are different. In the end, print the accuracy of the classifier. Upload your implementation (.py) + a file with the output (.txt).

# %%
import nltk
# nltk.download('senseval')
from nltk.corpus import senseval
from nltk.classify import NaiveBayesClassifier, accuracy
import random

instances = list(senseval.instances('interest.pos'))
random.shuffle(instances)

split = int(len(instances) * 0.9)
train_set, test_set = instances[:split], instances[split:]

def features(instance):
    return dict((word[0], True) for word in instance.context)
    # return dict((word[0], True) for word in instance.context if word[1][0] in ['N', 'V', 'J', 'R'])

train_set = [(features(instance), instance.senses[0]) for instance in train_set]
test_set = [(features(instance), instance.senses[0]) for instance in test_set]

classifier = NaiveBayesClassifier.train(train_set)
predictions = [classifier.classify(features) for features, _ in test_set]

results = [(pred, actual, pred == actual) for pred, actual in zip(predictions, [senses for _, senses in test_set])]

for pred, actual, match in results:
    print(f'Predicted: {pred}, Actual: {actual}, Match: {match}')

print(f'Accuracy: {round(accuracy(classifier, test_set) * 100, 2)}%')


with open('bayes.txt', 'w') as f:
    for pred, actual, match in results:
        f.write(f'Predicted: {pred}, Actual: {actual}, Match: {match}\n')
    f.write(f'Accuracy: {round(accuracy(classifier, test_set) * 100, 2)}%\n')


