"""
Team 9:
Gorav Kumar
Adith Shetty
Pratik Patil
Yash Megendale
Yug Patel
"""

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

def load_data_training(fname):
    """
    Load in training data with labels
    
    Inputs:
        fname               <str> Name of file with path
    
    Outputs:
        text_list           <list> List of reviews
        labels_list         <list> List of labels
    """
    f = open(fname, encoding="utf8", errors='ignore')
    text_list = []
    labels_list = []
    for line in f:
        split_line = line.strip('\n').split('\t')
        text_list.append(split_line[0].lower())
        labels_list.append(split_line[1])
        
    f.close()
    return text_list, labels_list

def load_data_testing(fname):
    """
    Load in testing data without labels
    
    Inputs:
        fname               <str> Name of file with path
    
    Outputs:
        text_list           <list> List of reviews
    """
    f = open(fname, encoding="utf8", errors='ignore')
    text_list = []
    for line in f:
        line = line.strip('\n').lower()
        text_list.append(line)
        
    f.close()
    return text_list


def run_model(vectorizer, model, training_text, training_labels, testing_text):
    """
    Trains and test a model given the data, model and count vectorizer
    
    Inputs:
        vectorizer          <sklearn> Count Vectorizer to use for text
        model               <sklearn> Classification Model
        training_text       <list> Text for training
        training_labels     <list> Labels for training
        testing_text        <list> Text to testing
        
    Outputs:
        pred                <list> Predicted labels for testing set
    """
    vectorizer.fit(training_text)

    counts_train = vectorizer.transform(training_text)
    counts_test = vectorizer.transform(testing_text)
    
    model.fit(counts_train, training_labels)
    pred = model.predict(counts_test)
    return pred

def export_pred(pred, path):
    """
    Load in testing data without labels
    
    Inputs:
        pred                <str> Predicted labels for testing set
        path                <str> Directory to output files to
    
    Outputs:
    """
    f = open(path + 'pred_labels.txt','w')
    for label in pred:
        f.write(label+ '\n')
    f.close()

if __name__ == '__main__':
    # Path of data 
    path = 'C:/Users/gorav/Documents/College/Spring 2020/BIA 660/Final Project/Submission/' ### NEEDS TO BE CHANGED BY PROFESSOR/TA
    
    # Load in training data
    training_text, training_labels = load_data_training(path + 'training.txt')
    
    # Load in testing data
    testing_text = load_data_testing(path + 'testing.txt') ### NEEDS TO BE CHANGED BY PROFESSOR/TA
    
    # Create Vectortizer for data
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), min_df=1, use_idf=True, 
                                 smooth_idf=True, stop_words = stopwords.words('english'))
    
    # Create Model for data
    model1 = MultinomialNB()
    model2 = LinearSVC()
    model3 = LogisticRegression(solver='liblinear')
    predictors=[('nb',model1),('svc',model2),('lr',model3)]
    VT=VotingClassifier(predictors)
    
    # Fit Model and get predictions
    pred = run_model(vectorizer, VT, training_text, training_labels, testing_text)
    
    # Output predictions to .txt file
    export_pred(pred, path)