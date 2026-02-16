import datarobot as dr

# login
dr.Client(token="Njk4ZmRhZThmMmRiN2MwMWU4NWU5YWEyOmRkQ2hlSDlEVXBxdElxcjR2RHFoMHVNbWlDdjcyWGtHUHFOQ3dMWVdZYU09", endpoint="https://app.datarobot.com/api/v2")

# register custom model
model_pkg = dr.CustomModel.create(
    name="ModelClassificationModelViaSciKitLearn.joblib- My Sklearn Pipeline 2026",
    target_type="Binary",
    description="RandomForest pipeline with scaler. ModelClassificationModelViaSciKitLearn.joblib",
    language="python3",
)

# upload the file
model_pkg.upload_files("ModelClassificationModelViaSciKitLearn.joblib")