# Required libraries
import os 
import pandas as pd

# Insert the path to the test folder to read the images
TEST_DIR = "./Test 2/Test 2"
# Insert the path to the folder which contains the prediction files
PRED_PATH = "./files/"

# Load the list of predictions from the different classifiers 
# in dictionaries whose keys are the test filenames and whose values
# are the test labels
pred_list = []
for pred_file in os.listdir(PRED_PATH):
    pred_df = pd.read_csv(PRED_PATH+pred_file)
    pred_list.append(dict(zip(list(pred_df["id.jpg"].values), list(pred_df["label"].values))))

# Get test labels from the test images
test_labels = [file[:file.index("_")] for file in os.listdir(TEST_DIR)]
test_dict = dict(zip(os.listdir(TEST_DIR), test_labels))
# Delete weird key
del test_dict[".DS_Store"]

###################################### MOST VOTED ENSEMBLE ######################################
# Iterate over the test images to choose the most voted class between the classifiers
ensemble_preds = []
# Calculate the accuracy over the test dataset
accuracy = 0
for test_img in test_dict:
    try:
        all_preds = [pred_dict[test_img] for pred_dict in pred_list]
        most_voted_class = max(set(all_preds), key = all_preds.count)
        ensemble_preds.append(most_voted_class)
        accuracy += 1 if int(test_dict[test_img]) == int(most_voted_class) else 0
    except:
        pass
    
print("Accuracy: ", (accuracy/len(test_dict))*100)
# Create submission file
submission_df = pd.DataFrame({'id.jpg': list(test_dict.keys()), 'label': ensemble_preds})
submission_df.to_csv("39_ensemble.csv", index=False)

###################################### WEIGHTED ENSEMBLE ######################################
# Weights for each classifier depending on its score
WEIGHTS = [0.1, 0.4, 0.3, 0.2, 0.5]
# Iterate over the test images to choose the most voted class between the classifiers
ensemble_preds = []
# Calculate the accuracy over the test dataset
accuracy = 0
for test_img in test_dict:
    try:
        all_preds = [pred_dict[test_img] for pred_dict in pred_list]
        weighted_preds = dict.fromkeys(all_preds, 0)
        for i in range(0, len(all_preds)):
            weighted_preds[all_preds[i]] += WEIGHTS[i]
        
        weighted_class = max(weighted_preds, key=weighted_preds.get)
        ensemble_preds.append(weighted_class)
        accuracy += 1 if int(test_dict[test_img]) == int(weighted_class) else 0
    except:
        pass
    
print("Accuracy: ", (accuracy/len(test_dict))*100)
# Create submission file
submission_df = pd.DataFrame({'id.jpg': list(test_dict.keys()), 'label': ensemble_preds})
submission_df.to_csv("weighted_ensemble.csv", index=False)