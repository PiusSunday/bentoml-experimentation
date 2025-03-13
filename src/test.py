import bentoml

iris_model = bentoml.sklearn.get("my_rf_model:latest").to_runner()

iris_model.init_local()

print(iris_model.predict.run([[5.9, 3.0, 5.1, 1.8]]))
