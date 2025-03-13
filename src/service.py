import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

# Load the saved model
model_ref = bentoml.sklearn.get("my_rf_model:latest")
model_runner = model_ref.to_runner()

# Create a service
svc = bentoml.Service("my_rf_classifier", runners=[model_runner])

# Define an API endpoint
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_array: np.ndarray) -> np.ndarray:
    
    # Make prediction
    result = model_runner.predict.run(input_array)
    
    return result
