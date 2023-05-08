# Ex-06-Feature-Transformation

## AIM:

To read the given data and perform Feature Transformation process and save the data to a file.
## EXPLANATION:

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
## ALGORITHM:

STEP 1 Read the given Data

STEP 2 Clean the Data Set using Data Cleaning Process

STEP 3 Apply Feature Transformation techniques to all the features of the data set

STEP 4 Save the data to the file

## PROGRAM:
```
DEVELOPED BY: HARISH RAGAVENDRA S
REGISTER NUMBER: 212222230045
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
df=pd.read_csv("/content/Data_to_Transform.csv")
df
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
```
## OUTPUT:
![Screenshot from 2023-05-08 21-03-08](https://user-images.githubusercontent.com/114852180/236868149-a7a2c182-afae-41d0-a7d2-f7af22665cd3.png)
![Screenshot from 2023-05-08 21-03-12](https://user-images.githubusercontent.com/114852180/236868158-58fa7d15-4270-4e5f-9dc0-97e6b45256c0.png)
![Screenshot from 2023-05-08 21-03-21](https://user-images.githubusercontent.com/114852180/236868225-04c53ffb-43f7-49ef-bf76-a2c2a431a335.png)
![Screenshot from 2023-05-08 21-03-28](https://user-images.githubusercontent.com/114852180/236868238-e8969edd-92aa-49e9-83c9-150af7d38687.png)
![Screenshot from 2023-05-08 21-02-33](https://user-images.githubusercontent.com/114852180/236868260-18ca1186-c2c0-46c8-9c32-4363f42a3d4d.png)
![Screenshot from 2023-05-08 21-02-40](https://user-images.githubusercontent.com/114852180/236868274-67be6897-2c1f-41a6-8e2e-7ddc19faef51.png)
![Screenshot from 2023-05-08 21-02-47](https://user-images.githubusercontent.com/114852180/236868280-f7d664c9-5b2c-42f8-ba1c-adaab66367d8.png)
![Screenshot from 2023-05-08 21-02-51](https://user-images.githubusercontent.com/114852180/236868287-37c50a09-a6a0-48ad-b53e-4516dc5fec6e.png)
![Screenshot from 2023-05-08 21-02-55](https://user-images.githubusercontent.com/114852180/236868297-8574f088-a5dc-4347-b250-7e0b6acc9e68.png)
![Screenshot from 2023-05-08 21-02-58](https://user-images.githubusercontent.com/114852180/236868305-92ecbbe8-de9c-48d8-8d09-4069a0c166fb.png)

## RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully
