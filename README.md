# Conditional VAE

## dataset : QM9

Based on the Aspuru-Guzik group's Selfies repository, add conditional functionality to the basic VAE.

- Utilize a larger dataset if necessary, such as qm9.
- Refer to the calculate_properties function in utils.py to customize desired conditions.
- Utilize Tensorboard to visualize various metrics such as Reconstruction, Diversity, Validity, and loss. Verify real results (generated smiles / selfies) from the validation set.
- Update scores and adjust model weights accordingly for storage (See test.ipynb for reference).
