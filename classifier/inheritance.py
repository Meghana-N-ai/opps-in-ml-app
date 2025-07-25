from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Base class
class MLComponent:
    def __init__(self):
        print(f"{self.__class__.__name__} initialized.")

# DataLoader class
class DataLoader(MLComponent):
    def __init__(self):
        super().__init__()
        self.X, self.y = load_iris(return_X_y=True)
    
    def split(self, test_size=0.3, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

# Preprocessor class
class Preprocessor(MLComponent):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def fit_transform(self, X_train):
        return self.scaler.fit_transform(X_train)

    def transform(self, X_test):
        return self.scaler.transform(X_test)

# MLModel class
class MLModel(MLComponent):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

# Evaluator class
class Evaluator(MLComponent):
    def __init__(self, y_true, y_pred):
        super().__init__()
        self.y_true = y_true
        self.y_pred = y_pred

    def report(self):
        print("Classification Report:\n")
        print(classification_report(self.y_true, self.y_pred))

# MLApplication class
class MLApplication(MLComponent):
    def __init__(self):
        super().__init__()
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
