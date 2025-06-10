# ChemBerta inside Pytorch DeepSet architecture

## Preprocessing
- The REFPROP XR0 file was split into train, val, and test csv.
- Three smiles pairs were forced into the test set:
  ```python
    frozenset(['O=C=O', 'FC(F)F']),                        # CO₂ / R32
    frozenset(['FC=CC(F)(F)F', 'C=C(F)C(F)(F)F']),         # R1234ze(E) / R1234yf
    frozenset(['C=C(F)C(F)(F)F', 'FCC(F)(F)F'])            # R1234yf / R132a ```
  
## model
- ChemBerta-DeepSet to make predictions of beta and gamma coefficients.