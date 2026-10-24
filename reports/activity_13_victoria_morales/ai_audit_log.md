# Responsible AI Audit Log — Activity 13

Student: Victoria Morales Cabrera
Team: 3
AI tool: Perplexity

These prompts summarize my conversation with the AI.

## 1. Repository setup
Prompt: Help me start the activity step by step.
AI help: Suggested checking the folder, repository, and current branch.
My check: I ran the Git commands and confirmed that I was in the Team 3 repository with no pending changes.

## 2. Personal branch
Prompt: Should I create my own branch, like in Activity 10?
AI help: Changed its initial suggestion and helped me create a personal branch.
My check: I confirmed that my branch was feature/activity-13-pruning-victoria-morales. This name uses a personal suffix that is not specified in the activity sheet. Instructor approval of the name is still pending.

## 3. Python libraries
Prompt: Help me fix the missing sklearn error.
AI help: Explained how to create a virtual environment and install pandas and scikit-learn.
My check: I activated the environment, checked the installation output, and successfully ran the code.

## 4. Dataset and split
Prompt: Help me check and split the dataset.
AI help: Provided code to inspect the data and create a stratified 75/25 split.
My check: I confirmed 700 rows, four numeric predictors, and no missing values. The split had 525 training records and 175 test records.

## 5. Pruning experiment
Prompt: Help me train the trees and calculate the results.
AI help: Provided code for alpha values of 0.001, 0.015, and 0.080. It also included 0.000 because the instructions show two different values for the first candidate.
My check: I ran the script and reviewed the output. Alpha 0.015 had the lowest test error and total cost. I have not checked every calculation by hand.

## 6. Technical interpretation
Prompt: Explain my results in simple English.
AI help: Drafted an explanation using my results.
My check: My final review of the wording is still pending. I have not confirmed the meaning of class 1 in a data dictionary, so the statement about thermal runaway risk remains conditional.