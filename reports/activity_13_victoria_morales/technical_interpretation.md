# Technical Interpretation — Team 3

In this experiment, α = 0.015 gave the lowest test error (9.14%) and the lowest total cost (0.196429). The tree had seven leaves and a maximum depth of four. Among the tested models, it provided the best balance between tree size and prediction performance.

The model with α = 0.001 had 59 leaves and a depth of 12. Its training error was 0.00%, but its test error was 20.57%. This gap suggests overfitting: the tree learned details from the training data that were less useful for predicting new records. The additional model with α = 0.000 produced the same results.

When α increased to 0.080, the tree became a single leaf with a depth of zero. Its test error increased to 27.43%. This model always predicted the majority class instead of separating different cases. If class 1 represents thermal runaway risk, it would miss those risky cases and could prevent timely alerts for the client.

Therefore, moderate pruning was the best option for this experiment. However, these results do not guarantee the same performance on future data.