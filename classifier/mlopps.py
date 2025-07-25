
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
class DataLoader:
  def __init__(self):
    self.X, self.y = load_iris(return_X_y=True)
  def split(self, test_size=0.3, random_state=42):
    return train_test_split(self.X, self.y, test_size=test_size,random_state=random_state)
class Preprocessor:
  def __init__(self):
    self.scaler = StandardScaler()
  def fit_transform(self, X_train):
      return self.scaler.fit_transform(X_train)
  def transform(self, X_test):
    return self.scaler.transform(X_test)
class MLModel:
  def __init__(self):
    self.model = DecisionTreeClassifier()
  def train(self, X_train, y_train):
    self.model.fit(X_train, y_train)
  def predict(self, X_test):
    return self.model.predict(X_test)
class Evaluator:
  def __init__(self, y_true, y_pred):
    self.y_true = y_true
    self.y_pred = y_pred

  def report(self):
    print(" Classification Report:\n")
    print(classification_report(self.y_true, self.y_pred))
class MLApplication:
  def __init__(self):
    self.loader = DataLoader()
    self.preprocessor = Preprocessor()
    self.model = MLModel()
  def run(self):
    X_train, X_test, y_train, y_test = self.loader.split()
    X_train_scaled = self.preprocessor.fit_transform(X_train)
    X_test_scaled = self.preprocessor.transform(X_test)
    self.model.train(X_train_scaled, y_train)
    y_pred = self.model.predict(X_test_scaled)
    evaluator = Evaluator(y_test, y_pred)
    evaluator.report()
  