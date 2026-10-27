# Responsible AI Use Log — Activity 16

## AI Tool Used

Perplexity was used as a step-by-step programming assistant. The student executed the code locally in VS Code using the project's Python virtual environment.

## Summary of Technical Requests

The following items summarize the technical assistance requested during the activity. They are professional summaries, not verbatim quotations.

- Assist with loading and inspecting the Team 3 EV battery dataset using pandas.
- Configure a reproducible 80/20 train/test split with random_state=42.
- Implement five linear regression models with progressively larger predictor subsets.
- Calculate training R2, testing R2, adjusted training R2, test MAE, and test RMSE.
- Export the model comparison table and generate residual diagnostic plots for the two strongest candidates.
- Support the interpretation of model performance, residual patterns, and methodological limitations.
- Assist with drafting the technical model-selection defense and documenting responsible AI use.

## AI Code and Suggestions

The assistant proposed code to:

- Load the EV battery CSV from the project's data folder.
- Inspect dimensions, data types, missing values, and duplicate rows.
- Create one shared 80/20 train/test split using random_state=42.
- Define five progressively larger predictor combinations.
- Train five LinearRegression models.
- Calculate training R2, testing R2, adjusted training R2, test MAE, and test RMSE.
- Save the comparison table as a CSV.
- Generate residual diagnostic plots for the two candidates with the lowest test RMSE.
- Draft a technical defense based on the reported metrics and plots.

Core training and prediction snippet:

```python
model = LinearRegression()
model.fit(X_train_selected, y_train)

y_train_pred = model.predict(X_train_selected)
y_test_pred = model.predict(X_test_selected)
```

## Human Review and Execution Evidence

The student ran the script locally and shared the execution output.

The output showed:

- 680 records and 7 numerical columns.
- No missing values and no fully duplicated rows.
- 544 training records and 136 testing records.
- Successful alignment and split-overlap checks.
- Five model configurations containing 1, 2, 3, 4, and 5 predictors.
- A completed comparison table and saved diagnostic figures.

The student opened the diagnostic images in VS Code and shared screenshots. The assistant helped interpret them. This was an assisted review, not an independent statistical audit.

## Clarifications and Limitations

The CSV location was clarified before execution: it was stored in data, while the Python script was stored in the project root.

The assignment lists six available predictors but requests models with one to five predictors. The implemented comparison used five predictors and excluded cell_voltage_v. Therefore, Model 5 is not an all-six-predictor model.

The predictor order was an initial engineering proposal, not an empirically established ranking of individual importance.

Adjusted R2 was calculated on the training set and explicitly labeled. MAE and RMSE were calculated on the test set.

The diagnostic interpretation was visual. No formal assumption tests or error-independence analysis were performed.

The same test set was used to compare candidates and select the preferred model. No additional untouched holdout set or cross-validation was used, so performance after model selection still needs independent validation.

No sensor costs, maintenance costs, or financial return on investment were measured.

No execution error was reported in the shared successful runs. The limitations above are methodological qualifications, not fabricated bugs.

## Final Model Choice

Model 5 was selected as the best-performing candidate in this experiment.

It achieved:

- Test R2: 0.9835
- Adjusted training R2: 0.9811
- Test MAE: 1.1355 C
- Test RMSE: 1.3936 C

The residual plots provided visual support for the choice, but the results do not guarantee performance under different operating conditions or in production.