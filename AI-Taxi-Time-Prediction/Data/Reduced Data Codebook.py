#Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA

#Load the features.csv file
flight_data = pd.read_csv("C:/Users/youse/Desktop/QMPLUS DATA/features.csv")

#The type feature is mapped into numerical values (arrival>0, departure>1)
flight_data['type'] = flight_data['type'].map({'arrival': 0, 'departure': 1})

#Drop non-numerical features (rwy and id) and drop the target (taxi_time)
PCA_Input_features = flight_data.drop(columns=['rwy', 'id', 'taxi_time'])

#Standradise the features by scaling them to have a mean of 0 and standard deviation of 1
x = StandardScaler().fit_transform(PCA_Input_features)

#Compute the covariance matrix 
Covariance_Matrix = np.cov(x.T)

#Caclulate the eigenvalues and eigenvectors of the covariance matrix
Eigenvalues, Eigenvectors = np.linalg.eig(Covariance_Matrix)

#Sort in descending order 
Sorting_Index = Eigenvalues.argsort()[::-1]
Eigenvalues = Eigenvalues[Sorting_Index]
Eigenvectors = Eigenvectors[:, Sorting_Index]

#Selecting eigenvectors based on the first top m largest eigenvalues
m = 12
Top_Eigenvectors = Eigenvectors[:, :m]

#Calculate the contribution of each feature by summing the absolute values of the eigenvectors 
feature_contribution = []
for i in range(PCA_Input_features.shape[1]):
    contribution = np.sum(np.abs(Top_Eigenvectors[i, :]))
    feature_contribution.append(contribution)

#Sorting the features based on thier contribution score from highest to lowest
sorted_feature_contributions = sorted(enumerate(feature_contribution), key=lambda x: x[1], reverse=True)
print(sorted_feature_contributions)

#Select the top n features based on thier contribution score
n = 15
top_n_features = [PCA_Input_features.columns[i] for i, _ in sorted_feature_contributions[:n]]
print(f"\nThe top {n} features are: {', '.join(top_n_features)}")


#Create a new Data Frame with the selected feature and save it as a csv file along with the target taxi time
Selected_features_with_taxi_time = flight_data[top_n_features + ['taxi_time']]
Selected_features_with_taxi_time.to_csv("C:/Users/youse/Desktop/New Folder/Selected_features_with_taxi_time.csv", index=False)

#Perform PCA with 12 principal components 
PCA_flights = PCA(n_components=12)
principal_components = PCA_flights.fit_transform(x)

#Calculate the variance for each principal component
explained_variance = PCA_flights.explained_variance_ratio_ * 100
print("Explained Variance Ratio for each PC:")
for i, var in enumerate(explained_variance, 1):
    print(f"PC{i}: {var:.2f}%")

total_explained_variance = np.sum(explained_variance)
print(f"\nTotal Explained Variance: {total_explained_variance:.2f}%\n")

#Create a dataframe to hold the feature loadings on each principal component
Feature_loadings = pd.DataFrame(
    PCA_flights.components_,
    columns=PCA_Input_features.columns,
    index=[f'PC{i+1}' for i in range(PCA_flights.n_components_)]
)

#Create a dataframe with the principal components and the target taxi time
principal_components_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(12)])
principal_components_df['taxi_time'] = flight_data['taxi_time'].values
#Save the transformed PCA data as a CSV file
principal_components_df.to_csv("C:/Users/youse/Desktop/New Folder/PCA_flight_data.csv", index=False)


#Plot a heatmap with feature loadings
plt.figure(figsize=(12, 8))
sns.heatmap(Feature_loadings.T, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Contribution to Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Features')
plt.show()

#Plot a bar chart to display the variance for each principal component 
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance per Principal Component')
plt.show()



#This function splits the data into training and testing sets for the selected features dataset
def Training_Testing_Split_Selected_features(Selected_features_Data_Split):

    #calculate precentiles for taxi time
    percentile_5 = Selected_features_Data_Split["taxi_time"].quantile(0.05)
    percentile_95 = Selected_features_Data_Split["taxi_time"].quantile(0.95)

    #filter the data that is outside the 5th and 95th precentile. For this case the 5th precentile value was approximated to 1 minuts and 95th precentile was approximated to 50 minuts
    Selected_features_Data_Split = Selected_features_Data_Split[(Selected_features_Data_Split["taxi_time"] >= 1) & (Selected_features_Data_Split["taxi_time"] <= 50)]

    #Use an 80/20 split for training and testing
    train_percentage = 0.8

    #Split the data
    train, test = np.split(
        Selected_features_Data_Split.sample(frac=1, random_state=42),
        [int(train_percentage * len(Selected_features_Data_Split))]
    )

    print(len(train))
    print(len(test))

    #Save the training and testing sets
    train.to_csv('C:/Users/youse/Desktop/New Folder/Selected Features Training 80%.csv', index=False)
    test.to_csv('C:/Users/youse/Desktop/New Folder/Selected Features Testing 20%.csv', index=False)


#This function splits the data into training and testing sets for the transformed PCA dataset
def Training_Testing_Split_Transformed_PCA_data(PCA_Data_Split):

    #calculate precentiles for taxi time
    percentile_5 = PCA_Data_Split["taxi_time"].quantile(0.05)
    percentile_95 = PCA_Data_Split["taxi_time"].quantile(0.95)

    #filter the data that is outside the 5th and 95th precentile. For this case the 5th precentile value was approximated to 1 minuts and 95th precentile was approximated to 50 minuts
    PCA_Data_Split = PCA_Data_Split[(PCA_Data_Split["taxi_time"] >= 1) & (PCA_Data_Split["taxi_time"] <= 50)]

    #Use an 80/20 split for training and testing
    train_percentage = 0.8

    #Split the data
    train, test = np.split(
        PCA_Data_Split.sample(frac=1, random_state=42),
        [int(train_percentage * len(PCA_Data_Split))]
    )

    print(len(train))
    print(len(test))

    #Save the training and testing sets
    train.to_csv('C:/Users/youse/Desktop/New Folder/PCA transformed Data Training 80%.csv', index=False)
    test.to_csv('C:/Users/youse/Desktop/New Folder/PCA transformed Data Testing 20%.csv', index=False)



Training_Testing_Split_Selected_features(Selected_features_with_taxi_time)
Training_Testing_Split_Transformed_PCA_data(principal_components_df)




