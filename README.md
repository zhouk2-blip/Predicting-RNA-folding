trainandvalidate.py is the script to train our model
model_conv_attn.py is the model architecture
Datavisualization.py is the script to plot the 3d structures using trained model and give predictions
DATASET.py is the script to preprocess the data and do padding and masking every batch
best_model1.pth is the model that we used to present results in final writeup

In dataset, the test and validation contain the same IDS, but the coordinates differ:
Validation labels: have full or partial true (x,y,z)
Test labels: do not exist — you predict everything.
The validation_sequences_new.normalized.csv is just for visual comparison between predictions.csv. The DATASET.py will do normalization on dataset.
