# Project Description

- `XGBoost2Res.ipynb` demonstrates the training process.
- `optuna-xgb.py` demonstrates the hyperparameter tuning process, and `optuna-xgb.db` stores related information.
- `***.csv` is the final result. F1(macro) score reached 0.52 on a K5-fold test. Accuracy reached 0.81, which is significantly high on this superficial data. For comparison, the normal Random Forest reached 0.31-0.36 F1(macro) respectively.

## Cons

- Didn't use feature selection. Only 15 features have been used.

## Disclaimer

This repository is solely an archive of the DSAA1001 Project's XGBoost solution and framework, unrelated to the DSAA1001 course itself. I bear no liability for any code contained herein. This code is licensed under the MIT license.

---

Also has an RNN solution using GRU with the same accuracy. Contact me for more information.
