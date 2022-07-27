#Підключення потрібних бібліотек
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('Dataset-CR.csv', header=0)
print('Количество образцов:',df.shape[0])
pd.options.display.max_columns = None
#display(df)
df.head(78)

df = df.loc[df['Вік'] < 85  ]
df

#Візуалізація отриманих данних
df['Вік'].astype(int).plot.hist()

# Окрема оцінка погашення кретиту вчасно по часссу та віку клієнта
plt.figure(figsize = (10, 10))
# TARGET: 0, невчасна оплата
sns.kdeplot(df.loc[df['Статус займу'] == 0, 'Вік'], label = 'Статус займу = 0')
# TARGET: 1, вчасно/раніше погасив
sns.kdeplot(df.loc[df['Статус займу'] == 1, 'Вік'], label = 'Статус займу = 1')
plt.xlabel('Вік (в роках)')
plt.ylim(0, 0.12)
plt.ylabel('Плотність')
plt.legend()
plt.title('Оцінка плотності ядра')

data = df.replace(['RENT', 'MORTGAGE','OWN','OTHER'],[1,2,3,4], regex=False)
data= data.replace(['EDUCATION', 'MEDICAL','VENTURE','PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'],[1,2,3,4, 5, 6], regex=False)
data= data.replace(['A', 'B','C','D', 'E', 'F', 'G'],[1,2,3,4, 5, 6, 7], regex=False)
data

fig, ax = plt.subplots(figsize=(5, 5))
x = data['Вік']
y = data['Тип кредиту']
ax.scatter(x, y)

data['YEARS_BINNED'] = pd.cut(data['Вік'], bins =[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85] )
age_groups = data.groupby('YEARS_BINNED').mean()
age_groups

plt.figure(figsize = (10,4))
plt.bar(age_groups.index.astype(str), 100 * age_groups['Статус займу'])
plt.xticks(rotation = 45)
plt.xlabel('Вікова група')
plt.ylabel('Відмова від оплати(%)')
plt.title('Невиплата по віковим групам')

#Функція розрахунку WoE та IV
def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    
    dset = dset.sort_values(by='WoE')
    
    return dset, iv

for col in data.columns:
    if col == 'Статус займу': continue
    else:
        print('WoE and IV for column: {}'.format(col))
        df, iv = calculate_woe_iv(data, col, 'Статус займу')
        print(df)
        print('IV score: {:.2f}\n'.format(iv))
        print('\n')

#Візуалізація матриці
ext_data_corr=data.corr()
plt.figure(figsize = (10,7))
sns.heatmap(ext_data_corr, cmap = plt.cm.RdYlBu_r, vmin = 0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')

#підключення модулів для створення моделі прогнозу
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
#import ml_metrics, string, re, pylab as pl
#перевірка коефіцієнту інфляції дисперсії

#Поділ на тренувальну та тестову вибірку
X = data[[  'Вік',  'Тип кредиту', 'Дохід', 'Кредитний рейтинг','Нерухомість','Стаж(р.)']]
y = data.loc[:, data.columns == "Статус займу"].values.flatten()
#y - ext_data[['Target']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0, stratify=y)
print("Оригинальные размеры данных: ", X_train.shape, X_test.shape)

# модель лінійної регресії
regr = linear_model.LinearRegression()

# тренування моделі
regr.fit(X_train, y_train)

# прогноз на тренувальній вибірці
y_pred = regr.predict(X_test)
# коефіцієнти регресії
print('Coefficients: \n', regr.coef_)
# середньоквадратична похибка
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, 
                           y_pred))
# Коефіцієнт детермінації
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

#створення логічної регресії та РОС кривої
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_model=sm.Logit(y,X)
result=logit_model.fit()
#відображення отриманих результатів
print(result.summary2())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#Відображення результатів точності прогнозування
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#Відображення результатів точності прогнозування(матриця правильності)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
#RoC-крива для моделы логічної регресії
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

print(classification_report(y_test, y_pred))

#RoC-крива для моделы логічної регресії
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
