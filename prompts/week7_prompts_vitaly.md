Scan the repo and create a highly detailed diagram that explains the input of M1, M2 and M3 and the output of each components. I need to see exatcly what features each models use, create it highly detaled and not abstract at all

Imorove the diagram so it will be more human readable. In the botton of the MD file explain each Feature for extended Description

For the data points, if being used to train the model make sure there is Time Alignment & InterpolationIf. In example if some datasets are sampled at different rates of 1 month and 1 week, or misaligned, align them to a common timestamp using interpolation. Also, Use Forward Filling to propagate the last known value forward until the next timestamp observation on test data or on when we want to model to eval

In the "m1_m2_m3_feature_io" file, explain which feature are being used for the same models (in example m1 and m2),  explain how they are being used differently or what is the rational to use it again. Suggest room for future improvements as well

In the same file, try to to simplify the text even further. Evaluate if the file answers "summarize onto the macro variable choices, and how does that implement to the model?" and explain each feature rational and why it was chosen. Explain which features are not being used at all and which features we can further discover in the future

I want to add a diagram in the top of the page to show exatcly all the features that are being used to eah of the models. Make sure that the features are described only by their name in the diagram itself but the diagram needs to include all of them

Good, I want to further improve it and put the features in segments. In example, what is considered Macro, Trend, or for M2, what are the dynamic or the external factors. I want people to understand immediatly what is this feature related to

Read the code and check what is the time interval between each data point. In example, it could be every week or every 4-week or 1 month. If some of the data is weekly and some monthly, check if that data is being interpolated or has forward filling

Add the conversation we had exactly, including the models into a new file named "week7_promts_vitaly" inside the prompts directory