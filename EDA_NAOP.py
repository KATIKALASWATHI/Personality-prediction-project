import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##importing data
data=pd.read_csv('file:///C:/Users/Swathi/Desktop/NAOP project/NAOP_new.csv', encoding='latin1')
data.shape
data.head()
stud_posts=data.iloc[:,1:3]
stud_posts.head()
stud_posts.groupby(['Type']).count()

My_category=['A','B','C','D']
color=['blue','green','yellow','pink']
plt.figure(figsize=(10,4))
plt.title('Data in Category Wise')
stud_posts.Type.value_counts().plot(kind='bar', color=color)

#checking null values
stud_posts.isnull().sum()
#droping the duplicate rows
stud_posts=stud_posts.drop_duplicates(subset='Posts')
stud_posts.shape



#character count and word count
stud_posts['Char_count']=stud_posts['Posts'].str.len()
stud_posts['word_count']=stud_posts['Posts'].apply(lambda x: len(str(x).split(" ")))
avg_count=stud_posts['word_count'].groupby(stud_posts['Type']).mean()
avg_count1=pd.DataFrame(avg_count)
avg_count1['category']=['cat_A','cat_B','cat_C','cat_D']

#avgerage word_count plot
plt.figure(figsize=(12,5))
sns.barplot(data=avg_count1, x="category", y="word_count")


import re
# Remove ||| from dataset
stud_posts['Posts'] =stud_posts['Posts'].apply(lambda x:(re.sub("[]|||[]", " ", x)))
stud_posts['Posts'] = stud_posts['Posts'].apply(lambda x:(re.sub("/r/[0-9A-Za-z]", "", x)))
import string
punc=string.punctuation
stud_posts['Posts']=stud_posts['Posts'].apply(lambda x:" ".join( w for w in x.split() if w not in punc))
#removing url's
#stud_posts['Posts']=stud_posts['Posts'].apply(lambda x:(re.sub(r"http\S+", "", x)))
#removing punctuations and numbers
stud_posts['Posts']=stud_posts['Posts'].apply(lambda x:(re.sub('[^A-Za-z]', ' ', x)))
#stud_posts['Posts']=stud_posts['Posts'].str.replace('\d+', '')
#stud_posts['Posts']=stud_posts['Posts'].apply(lambda x:(re.sub('[^\w\s]', ' ', x)))
stud_posts['Posts']=stud_posts['Posts'].apply(lambda x:" ".join(x.lower() for x in x.split()))
stud_posts['Posts']=stud_posts['Posts'].str.strip()


#removing stopwords
with open("C:\\Users\\Swathi\\Desktop\\myproject\stop.txt", "r") as sw:
    stop_words=sw.read()

stop = stop_words.split("\n")
from nltk.corpus import stopwords
stop=stopwords.words('english')
stud_posts['Posts']=stud_posts['Posts'].apply(lambda x:" ".join(w for w in x.split() if w not in stop))




#lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()
stud_posts['Posts']=stud_posts['Posts'].apply(lambda x:" ".join([wnl.lemmatize(w) for w in x.split()]))

category_A=[]
category_B=[]
category_C=[]
category_D=[]

for i in range(0, len(stud_posts['Type'])):
    if stud_posts['Type'][i]=='A':
        category_A.append(stud_posts['Posts'][i])
    elif stud_posts['Type'][i]=='B':
        category_B.append(stud_posts['Posts'][i])
    elif stud_posts['Type'][i]=='C':
        category_C.append(stud_posts['Posts'][i])
    else:
        category_D.append(stud_posts['Posts'][i])

print(len(category_A));print(len(category_B));print(len(category_C));print(len(category_D))
##word cloud generation
from wordcloud import WordCloud
def wordcloud(data):
    category=' '.join(data)
    wd=WordCloud(
                    background_color='white',
                    width=1800,
                    height=1400
                   ).generate(category)
    fig = plt.figure(figsize = (10, 15))
    plt.axis('off')
    plt.imshow(wd)
    
wordcloud(category_A)
wordcloud(category_B)
wordcloud(category_C)
wordcloud(category_D)



##top 200 words
from nltk import FreqDist
def freq_words(x,terms=200):
    all_words=x.split()
    fdist=FreqDist(all_words)
    df=pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    return (df.nlargest(columns='count', n=terms))
    
    d=df.nlargest(columns='count', n=terms)
    plt.figure(figsize=(15,5))
    ax=sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel="count")
    plt.show()

d=freq_words(str(category_D))
c=freq_words(str(category_C))
b=freq_words(str(category_B))

a=freq_words(str(category_A))

#d.to_csv('topwords_D.csv', header=False)
#c.to_csv('topwords_C.csv', header=False)
#b.to_csv('topwords_B.csv', header=False)
#a.to_csv('topwords_A.csv', header=False)
#top 200 trigrams
from textblob import TextBlob
import collections
def Ngrams(category):
    for i in category_C:
        grams=TextBlob(i).ngrams(2)
        print(pd.DataFrame(grams))
    
counts = collections.Counter()
def Ngrams(category):
    for i in category:
        words=i.split()
        counts.update(nltk.bigrams(words))
        common_bigrams =counts.most_common(20)
    return common_bigrams

cat_D=pd.DataFrame(Ngrams(category_D))
cat_C=pd.DataFrame(Ngrams(category_C))
cat_B=pd.DataFrame(Ngrams(category_B))
cat_A=pd.DataFrame(Ngrams(category_A))

-----------------------------------------------------------------------------------
###Sentiment score
from textblob import TextBlob
def sentiment_score(text):
    score=text.apply(lambda x: TextBlob(x).polarity)
    return score
stud_posts['sentiment_score']=sentiment_score(stud_posts['Posts'])
stud_posts.head()
------------------------------------------------------------------------

####TFIDF matrix
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
X=tfidf.fit_transform(stud_posts['Posts'])
Y=stud_posts['Type']

import pickle
pickle.dump(tfidf,open('Transform.pkl', 'wb'))

###### Data splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X, Y, test_size=0.3, random_state=42)
print(x_train.shape);print(x_test.shape);print(y_train.shape);print(y_test.shape)
------------------------------------------------------------------------------------------

###Converting imbalanced data into balance

from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
sm = SMOTE() 
x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel()) 
print(x_train_res.shape);print(y_train_res.shape)

print("After OverSampling, counts of label 'A': {}".format(sum(y_train_res == 'A'))) 
print("After OverSampling, counts of label 'B': {}".format(sum(y_train_res == 'B'))) 
print("After OverSampling, counts of label 'C': {}".format(sum(y_train_res == 'C'))) 
print("After OverSampling, counts of label 'D': {}".format(sum(y_train_res == 'D'))) 
--------------------------------------------------------------------------------------


#########Modelbuilding
####Naive Bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

model1=MultinomialNB()
model1.fit(x_train_res, y_train_res)
y_pred1=model1.predict(x_test)
y_pred1
print(model1.score(x_test, y_test))
print(classification_report(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))


####SVM
from sklearn.svm import SVC
model2=SVC()
model2.fit(x_train_res, y_train_res)
y_pred2=model2.predict(x_test)
y_pred2

print(accuracy_score(y_test, y_pred2))
print(classification_report(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))

#####Logestic regression
from sklearn.linear_model import LogisticRegression
model3=LogisticRegression()
model3.fit(x_train_res,y_train_res)

y_pred3=model3.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(classification_report(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))

pickle.dump(model3,open('Model.pkl','wb'))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model4=DecisionTreeClassifier(criterion='entropy')
model4.fit(x_train_res, y_train_res)
y_pred4=model4.predict(x_test)
y_pred4

print(accuracy_score(y_test, y_pred4))
print(classification_report(y_test, y_pred4))
print(confusion_matrix(y_test, y_pred4))


##Random_forest
from sklearn.ensemble import RandomForestClassifier
model5=RandomForestClassifier(n_estimators=50)
model5.fit(x_train_res, y_train_res)
y_pred5=model5.predict(x_test)

print(accuracy_score(y_test,y_pred5))
print(confusion_matrix(y_test, y_pred5))
print(classification_report(y_test,y_pred5))

##Neural_networks
from sklearn.neural_network import MLPClassifier
model6=MLPClassifier(hidden_layer_sizes=(5,5))
model6.fit(x_train_res, y_train_res)
y_pred6=model6.predict(x_test)

print(accuracy_score(y_test, y_pred6))
print(confusion_matrix(y_test, y_pred6))
print(classification_report(y_test, y_pred6))

####Bagging Classifier
from sklearn.ensemble import BaggingClassifier

model7=BaggingClassifier(DecisionTreeClassifier(criterion='entropy'))
model7.fit(x_train_res,y_train_res)

y_pred7=model7.predict(x_test)


print(accuracy_score(y_test,y_pred7))
print(confusion_matrix(y_test,y_pred7))
print(classification_report(y_test,y_pred7))


-------------------------------------------------------------------
















