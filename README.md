# decision_tree_classifier
A decision tree classifier built from scratch utilizing pseudo-code provided from "Fundamental Algorithms in Machine Learning Systems" lecture. The decision tree classifier is written in Python, and utilizes Gini Index, Information Gain, and Gain Ratio to produce decision trees for each metric. An accuracy score is also computed for each decision tree, reflecting each decision tree's performance for each metric.

Algorithm also utilizes the pandas library for interpreting datasets in .csv file format. Repository includes source code, all datasts used for testing, and a report about the classifier and its effectiveness.

Important takeaways:
  - Collaborated with a classmate using Google Colab for efficient work.
  - Datasets used for testing are all found from the UC Irvine Machine Learning Repository. The datasets used for this project include Car Evaluation, Nursery, Lymphography, Hayes-Roth, Tic-Tac-Toe Endgame, Lenses, Balance Scale, and MONK's Problems.
  - Conducted data preprocessing to clarify feature labels and reposition the classes column to the last column in each dataset. This step was crucial for accurate decision tree creation, preventing issues arising from identical names representing
    different features.
  - Noted variations in accuracies due to the pandas library's 70/30 split for training and test data. Despite this, the decision tree classifier should be deterministic with consistent results when using the same training and testing data.
  - Source code incorporates a small dataset within the main function for initial algorithm testing, originating from our class lecture. It successfully generates a 100% accurate decision tree for Information Gain and Gain Ratio metrics, while yielding
    a 71.4% accuracy for the Gini Index. This outcome aligns with expectations, given that the Gini Index tends to create leaf nodes by deducing the majority class from a small subset within a branch, making it more susceptible to such discrepancies
    compared to the other metrics.
