# Conditional VAE

## dataset : QM9

Based on the Aspuru-Guzik group's Selfies repository, add conditional functionality to the basic VAE.

- Utilize a larger dataset if necessary, such as zinc dataset.
- Refer to the calculate_properties function in utils.py to customize desired conditions.
- Added Tensorboard to visualize various metrics such as Reconstruction, Diversity, Validity, and loss. you can see the actual results (generated smiles / selfies) from the validation set.
- Model weights are automatically saved (see test.ipynb).
