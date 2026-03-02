# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  


# Install dependencies (installed in the venv directory)
pip install --upgrade pip
pip install scikit-learn xgboost torch pytorch-lightning matplotlib awkward uproot

# Split SM and LIV samples
python split_sm_liv.py

# Training model
python training_model.py