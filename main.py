import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics



# Adatok vizsgálata:
def check_dataset(dataset, head=5):
    print("Head_________________")
    print(dataset.head(head))
    # dataset méretét adja meg (tudple-t ad vissza), azaz mennyi oszlop és sor van a táblázatban.
    print("Shape_________________")
    print(dataset.shape)
    print("Types_________________")
    print(dataset.dtypes)
    # Mennyi 0-ás érték van a táblázatban:
    print(("NA___________________"))
    print(dataset.isnull().sum())
    #Összefoglaló stratégia: átlag, szórás, minimum, maximum, 25%-os, 50%-os (medián), 75%-os kvartilis. A .T metódus a táblázatot transzponálja, így az oszlopok helyett sorokban jelennek meg a statisztikák.
    print("Quartiles_____________")
    print(dataset.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


advertising = pd.read_csv('Advertising (1).csv')
dataset = advertising.copy()
dataset.head()

# Táblázat beállítása
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x:"%.4f" % x)
dataset.drop('Unnamed: 0', axis=1, inplace=True)
print(dataset.head())

check_dataset(dataset)

sns.pairplot(dataset, kind="reg")
sns.heatmap(dataset.corr(), vmin=-1, vmax=1, cmap='Blues', annot=True)

X = dataset[['TV', 'Radio', 'Newspaper']]
Y = dataset[['Sales']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(x_train)

print(f'x_train : {x_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'x_test : {x_test.shape}')
print(f'y_test : {y_test.shape}')

regressor = LinearRegression()
regressor.fit(x_train, y_train)

print(regressor)
# Előrejelzés
pred = regressor.predict(x_test)
# Actual vs Predicted
pred = pred.ravel()
y_test = y_test.values.ravel()
ac = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
ac.head(10)

# Accuracy of Model
print(f'Accuracy : {regressor.score(x_test,y_test)*100} %')

print('Mean Absolute Error:',
      metrics.mean_absolute_error(y_test, pred))
