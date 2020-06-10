import imblearn
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks,NearMiss

class ClassImbalance:
    def __init__(self):
        print("ClassImbalance")
        pass

    def smote_tomek(self, x_train,y_train):
        oversample = BorderlineSMOTE(sampling_strategy = 0.5, random_state = 0, k_neighbors = 5, m_neighbors=10, n_jobs=-1, kind = 'borderline-1')
        X, y = oversample.fit_resample(x_train, y_train)
        
        tom_lin = TomekLinks(sampling_strategy='majority', n_jobs = -1)
        X, y = tom_lin.fit_resample(X, y)
        # print(len([i for i in y_train.values if i==1]))
        # print(len([i for i in y.values if i==1]))
        # print(len(y_train))
        # print(len(y))
        return X,y    

    def easy_ensemble_clasiffication(self, classifier):
        
        easy_ensemble = imblearn.ensemble.EasyEnsembleClassifier(n_estimators=3, base_estimator=classifier, sampling_strategy='majority', n_jobs=-1)
        return(easy_ensemble)

    def near_miss (self, x_train, y_train):
        
        nm = NearMiss (sampling_strategy = 'majority',n_neighbors = 5,version=1)
        X, y = nm.fit_resample(x_train, y_train)
        
        return X, y