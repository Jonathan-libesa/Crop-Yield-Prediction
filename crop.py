import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error,r2_score
class colorss:
    yellows=['#ffffd4','#fee391','#fec44f','#fe9929','#d95f0e','#993404','#a70000','#ff5252','#ff7b7b','#ffbaba']
    greens=['#ffffd4','#fee391','#fec44f','#fe9929','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#005a32']
cmaps=['flare','icefire','bwr_r','Accent','Spectral','RdGy','afmhot_r','afmhot','inferno','seismic','vlag','vlag_r']
# Read the dataset into a pandas DataFrame
yield_data = pd.read_csv( r"C:\Users\cash\Documents\SEMSTER 4.1 NOTES\yield.csv")
temp_data = pd.read_csv( r"C:\Users\cash\Documents\SEMSTER 4.1 NOTES\temp.csv")
rainfall_data = pd.read_csv( r"C:\Users\cash\Documents\SEMSTER 4.1 NOTES\rainfall.csv")
pesticides_data = pd.read_csv( r"C:\Users\cash\Documents\SEMSTER 4.1 NOTES\pesticides.csv")
df = pd.read_csv( r"C:\Users\cash\Documents\SEMSTER 4.1 NOTES\yield_df.csv")

yield_data.head(5)
rainfall_data.head(5)

print(yield_data)
print(rainfall_data)
print(pesticides_data)
print(df)
print(temp_data)

#Dataset Basic Information
df.info()

#Summary Statistics for Numerical Variables
df.describe().T

#Summary Statistics for Categorical Variables
df.describe(include='object')


df.drop("Unnamed: 0", axis=1,inplace=True)
df

# remove countries with less than 100 record
country_counts =df['Area'].value_counts()
countries_to_drop = country_counts[country_counts < 100].index.tolist()
df_filtered = df[~df['Area'].isin(countries_to_drop)]
df = df_filtered.reset_index(drop=True)

df


datacorr=df.copy()



sns.set(palette='BrBG')
df.hist(figsize=(5,10));

df2=df[df['Item']=='Yams']
df2.groupby('Year')['hg/ha_yield'].mean().plot(color='brown')



palette = sns.color_palette('tab20', 21,as_cmap=True)
num_plots = 7
areas_per_plot = 10

# Get unique areas 
unique_areas = sorted(df['Area'].unique())

# Split into chunks
area_chunks = [unique_areas[i:i+areas_per_plot] for i in range(0, len(unique_areas), areas_per_plot)]
area_chunks[-2] = unique_areas[-11:] 
fig, axs = plt.subplots(ncols=num_plots, figsize=(30, 10))
j=0
for i, ax in enumerate(axs):

    plot_df = df[df['Area'].isin(area_chunks[i])]
    for i, area in enumerate(plot_df['Area'].unique()):
       data = plot_df[plot_df['Area'] == area]
       ax.hist(data['hg/ha_yield'], facecolor=palette(i), label=area)

    ax.legend()
    j+=1
   
plt.show()

#CHECKING CONTRIES WITH TOP RAINFALL AND LEAST RAINFALL
for i in range(0,7):
    plot_df = df[df['Area'].isin(area_chunks[i])]
    plot_df.groupby(['Area'])['average_rain_fall_mm_per_year'].mean().plot(kind='bar',rot=0,color=colorss.greens)
    plt.xticks(rotation=90)
    plt.show()

#CHECKING COUNTRIES WITH LEAST PESTICIDE AND HIGH PESTICIDE
for i in range(0,7):
    plot_df = df[df['Area'].isin(area_chunks[i])]
    plot_df.groupby(['Area'])['pesticides_tonnes'].mean().plot(kind='bar',rot=0,color=colorss.yellows)
    plt.xticks(rotation=90)
    plt.show()


#Checking if  pesticides affect yield production?
for i in range(0,7):
    plot_df = df[df['Area'].isin(area_chunks[i])]
    plot_df.groupby('Area')[['pesticides_tonnes', 'hg/ha_yield']].mean().plot(kind='bar',rot=0,color=colorss.yellows[-6:])
    plt.xticks(rotation=90)
    plt.show()

#Top pesticides and Least pesticides used on items: Plantains and others
sns.barplot(data=df, x = df.Item, y = df['pesticides_tonnes'],palette='BrBG')
plt.xticks(rotation=90)
plt.show()

#Top producing items: potatoes, cassava Least producing items: soybeans, sorghum, wheat, maize, and rice, paddy
a4_dims = (16.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot(x="Item",y="hg/ha_yield",palette="BrBG",data=df,ax=ax)

#Cassava, Yams, and Plantains and others need more than 15 avg temperature to grow all other items can grow at any temperature range
sns.scatterplot(x = 'Item', y = 'avg_temp', data = df,size=10,color='y')
plt.xticks(rotation=90);


col = ['Year', 'average_rain_fall_mm_per_year','Pesticies Value', 'Avg_Temp', 'Area', 'Item', 'Yield Value']
df = df[col]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
