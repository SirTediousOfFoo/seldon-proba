import joblib
from seldon_core.user_model import SeldonResponse

class ProbniModel(object):
    last_predict = None
    
    def __init__(self):
        with open("svm-novelty-detector", "rb") as model_file:
            self._lr_model = joblib.load(model_file)
        
    def predict(self, X, feature_names):
        last_predict = self._lr_model.predict(X)
        runtime_metrics = [{"type": "GAUGE", "key": "prediction", "value": last_predict}]
        return SeldonResponse(data=last_predict, metrics=runtime_metrics, tags=runtime_tags)      
    
    def metrics(self):
        return [{"type": "GAUGE", "key": "predict_val", "value": last_predict}]
