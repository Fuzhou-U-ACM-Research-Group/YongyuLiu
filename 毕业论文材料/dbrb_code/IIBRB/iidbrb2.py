from DBRB_opt.dbrb import DBRBClassifier, DBRBRegressor
from DBRB.incomplete_inference_belief_base2 import IIBB2


class IIDBRB2:
    def __init__(self, A, D, is_classify=False):
        self.brb = None
        self.iibb = None
        self.is_classify = is_classify

        if is_classify:
            self.brb = DBRBClassifier(A, D)
        else:
            self.brb = DBRBRegressor(A, D)

    def fit(self, X, y):
        self.brb.fit(X, y)
        self.iibb = IIBB2(X, self.brb.A)
        # self.iibb = IIBB(X, self.brb.A)
        return self

    def predict(self, X):
        data = self.iibb.resolve(X)
        if self.is_classify:
            return self.brb.classify(data)
        else:
            return self.brb.predict(data)


class IIDBRBClassifier2(IIDBRB2):
    def __init__(self, A, D):
        super().__init__(A, D, True)

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)


class IIDBRBRegressor2(IIDBRB2):
    def __init__(self, A, D):
        super().__init__(A, D, False)

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)
