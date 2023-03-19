import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import string
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error


#reads train csv file
phrases = pd.read_csv("test.csv")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

#Text Preprocessing

#makes the phrases lowercase
phrases["Text"] = phrases["Text"].str.lower()

#remove special characters 
phrases["Text"] = phrases["Text"].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))

#removes punctuation
phrases["Text"] = phrases["Text"].str.replace('[^A-Za-z0-9]+',' ', regex = True)

#remove links
phrases["Text"] = phrases["Text"].str.replace("http\S+", "", regex = True)

#removes digits
phrases["Text"] = phrases["Text"].apply(lambda x: re.sub('W*dw*','',x))

#remove stopwords
stopwords = stopwords.words('english')
phrases["Text"] = phrases["Text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

#Stemmers each word in the phrase
phrases['Text'] = phrases['Text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

#Lemmatize each word in the phrase
phrases['Text'] = phrases['Text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Tokenize each phrase
phrases["Text"] = phrases["Text"].apply(nltk.word_tokenize)

# Linguistic Feature

#Bag of Words Method
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(phrases["Text"].apply(lambda x: ' '.join(x)))
#print(bag_of_words.toarray())

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, phrases["Sentiment"], test_size=0.3, random_state=42)

# train logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
svc = SVC(probability=True)
svc.fit(X_train, y_train)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# evaluate model on testing set
accuracy = lr.score(X_test, y_test)
print("Accuracy using Bag of Words, Logistic Regression Model: {:.2f}%".format(accuracy * 100))
accuracy = svc.score(X_test, y_test)
print("Accuracy using Bag of Words, SVC Model: {:.2f}%".format(accuracy * 100))
accuracy = rfc.score(X_test, y_test)
print("Accuracy using Bag of Words, RFC Model: {:.2f}%".format(accuracy * 100))
# Create TF-IDF vectorizer object
vectorizer = TfidfVectorizer()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(vectorizer.fit_transform(phrases["Text"].apply(lambda x: ' '.join(x))), phrases["Sentiment"], test_size=0.85, random_state=42)

# Logistic regression model
lc = LogisticRegression()
svc = SVC(probability=True)
rfc = RandomForestClassifier()

# Train the model on the training data
lc.fit(X_train, y_train)
svc.fit(X_train, y_train)
rfc.fit(X_train, y_train)

# Evaluate the model on the testing data
accuracy = lr.score(X_test, y_test)
print("Accuracy using TF-IDF, Logistic Regression Model: {:.2f}%".format(accuracy * 100))
accuracy = svc.score(X_test, y_test)
print("Accuracy using TF-IDF, SVC Model: {:.2f}%".format(accuracy * 100))
accuracy = rfc.score(X_test, y_test)
print("Accuracy using TF-IDF, RFC Model: {:.2f}%".format(accuracy * 100))


print(phrases.head())
