Step 1 | Import Libraries
 Library such as pandas,seaborn,matplotlib,linearRegression
RandomForestRegressor, GradientBoostingRegressor 

Step 2 | Read Dataset



Step 3 | Dataset Overview
  Dataset Basic Information
Number of Entries: The dataset consists of 28242 entries, ranging from index 0 to 28241.
Columns: There are 8 columns in the dataset.
Data Types:
Most of the columns (6 out of 8) are of the int64 & float64 data type.
Only the Item and Area columns are of the object data type.
Missing Values: There doesn't appear to be any missing values in the dataset as each column has 28242 non-null entries.

Step 3.2 | Summary Statistics for Numerical Variables
Numerical Features:
average_rain_fall_mm_per_year: The average rainfall per year is approximately 1149, with the least rainfall being 51 and the most 3240.
pesticides_tonnes: The mean pesticides used in tonnes is a whopping 37077 tonnes, with minimum as little as 0.04 and maximum as huge as 367778 tonnes.
hg/ha_yield: The average crop production yield is 77053.3. Ranging from 50 all the way to 501412 hectograms per hectare.


Step 4 | EDA
inferences:
rainfall: Most rainfall is between 0-1000 and very few is around 3000
pesticides_tonnes: The majority of used pesticides is little to zero
avg_temp: Most average temperatures is around 25. *hg/ha yield: There is a vast majority of yield production around 0.




![crop](https://github.com/Jonathan-libesa/Crop-Yield-Prediction/assets/75207695/1dbb93f5-c2b4-44fe-9ee6-834d4cdfd0fe)





Inferences:
Group 0: Australia produced the largest amount of yield harvesting potatoes while Angola has the least yield harvesting sorghum,soybeans and maize.
Group 1: Egypt was producing most yield in this group growing sweet potatoes and potatoes while Ecuador was struggling in harvesting wheat.
Group 2: France and Germany are at the top of yield production both harvesting potatoes while Honduras wasn't in luck harvesting wheat
Group 3: India soared in growing Cassava and Japan was at the top growing potatoes while Madagascar struggled to grow soybeans and sorghum.
Group 4: Morocco and Mexico shined in yield production of potatoes while Niger failed to have high productions of wheat
Group 5: Pakistan was struggling in growing sorghum while South Africa Spain, and Saudi Arabia shined in growing Potatoes
Group 6: Tajikistan failed to harvest large amounts of soybeans while United Kingdom and Turkey produced a myriad of potatoes Collectively:
Top producing countries: United Kingdom, France, Germany, Australia, and Japan
Top produced Item: potatoes
Least producing countries: Zimbabwe, Azerbaijan, Angola, Niger, Tajikistan
Least produced item: sorghum and soybeans

![rain](https://github.com/Jonathan-libesa/Crop-Yield-Prediction/assets/75207695/a7b607d6-015e-4b6d-89eb-06f02d0ffeaf)

