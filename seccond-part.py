import time
s = time.time()
from numpy import nan
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

Data = pd.read_csv('train_sentiment.csv')
Data['rating'].replace('|', nan, inplace=True)
Data['rating'] = pd.to_numeric(Data['rating'], errors='coerce')
Data['rating'].fillna(Data['rating'].mean() ,inplace=True)
Data.loc[Data['rating'] >= 3.5, 'rating'] = 6
Data.loc[Data['rating'] < 3.5,'rating'] = -1
Data['cleaned_review'] = Data['review'].apply(preprocess_text)
vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
Data['vector'] = Data['cleaned_review'].apply(lambda x: vectorizer.encode(x))
X_train, X_test, y_train, y_test = train_test_split(Data.drop('rating',axis=1), Data['rating'], test_size=0.2, random_state=42)
log_reg = LogisticRegression()
rnd_for = RandomForestClassifier()
knn = KNeighborsClassifier()
log_reg.fit(list(X_train['vector']), y_train)
print('LogisticRegression score:',f1_score(y_test, log_reg.predict(list(X_test['vector'])), average='weighted'))
rnd_for_hyp = {'n_estimators': [10, 50, 100, 200],'max_depth': [5, 10,15]}
rnd_for_grid = GridSearchCV(rnd_for, rnd_for_hyp, cv=5,scoring='f1_weighted',verbose=4)
rnd_fit_model = rnd_for_grid.fit(list(X_train['vector']), y_train)
print('Best parameters for RandomForestRegressor: ', rnd_fit_model.best_params_)
print('Best score for RandomForestRegressor: ', rnd_fit_model.best_score_)
knn_hyp = {'n_neighbors': [3, 5, 7, 9, 11]}
knn_grid = GridSearchCV(knn, knn_hyp, cv=5, scoring='f1_weighted', verbose=4)
knn_fit_model = knn_grid.fit(list(X_train['vector']), y_train)
print('Best parameters for KNeighborsClassifier: ', knn_fit_model.best_params_)
print('Best score for KNeighborsClassifier: ', knn_fit_model.best_score_)
print(time.time()-s)