import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress non-critical user warning
pd.options.display.show_dimensions = False  # Remove pandas dataframe dimensions from output display


# Read in the data as a pandas dataframe
def dataImport(fileName):
    data = pd.read_excel('clinical_dataset.xlsx')
    return data


# Calculate all summary stats for the selected data to 2dp
def dataSummary(columnName, customName):
    mean = round(df[columnName].mean(), 2)
    min = round(df[columnName].min(), 2)
    max = round(df[columnName].max(), 2)
    median = round(df[columnName].median(), 2)
    # mode = df[columnName].mode()
    std = round(df[columnName].std(), 2)

    print('------------------', columnName, '------------------')
    print('The mean of', customName, 'is', mean)
    print('The minimum of', customName, 'is', min)
    print('The maximum of', customName, 'is', max)
    print('The median of', customName, 'is', median)
    print('The standard deviation of', customName, 'is', std)
    print('-------------------', '-' * len(columnName), '-------------------',
          sep='')  # Ensures the dashes are the same length

    # print(mode)  # Mode kinda weird
    return


# Convert list to string
def listToString(s):
    deli = " / "
    return (deli.join(s))


# Find and display the properties of the selected dataset
def dataProperties(dataset):
    print('The size of the dataset is', len(dataset), 'rows by', len(dataset.columns), 'columns.')
    print('Meaning there are', dataset.size, 'total elements in the dataset.')
    print('Their are', len(dataset.columns), 'features in the dataset, which are:',
          listToString(dataset.columns.values))
    print('----------------------------------------------')
    return


# Count the number of missing values in the selected dataset
def dataMissing(dataset):
    print('The number of missing values in the dataset is:', sum(dataset.isnull().values.ravel()))
    print('----------------------------------------------')
    return


# Normalise the data using a min-max normalisation method
def dataNormalisation(dataset):
    statusDropped = dataset.drop(['Status'], axis=1)
    normalised = (statusDropped - statusDropped.min()) / (statusDropped.max() - statusDropped.min())
    return normalised


# Plot a boxplot and density plot for the selected dataset
def dataPlotting(dataset):
    # Box plot creation
    sns.boxplot(dataset['Status'], dataset['Age'], palette=[sns.xkcd_rgb["blue"], sns.xkcd_rgb["light red"]],
                saturation=0.5, width=0.4)
    plt.suptitle('')  # Remove the default boxplot title
    plt.title('A box plot to show the comparison of ages\nbetween cancerous and healthy patients')
    plt.xlabel('Status')
    plt.ylabel('Age')
    plt.show()

    # Density plot creation
    sns.kdeplot(dataset[dataset['Status'] == 'healthy']['BMI'], shade=True, color='b', label='healthy')
    sns.kdeplot(dataset[dataset['Status'] == 'cancerous']['BMI'], shade=True, color='r', label='cancerous')
    plt.legend()
    plt.title('A density plot to show the comparison of BMI\nbetween cancerous and healthy patients')
    plt.show()


if __name__ == '__main__':
    df = dataImport('clinical_dataset.xlsx')
    print('------------ The original dataset ------------')
    print(df)
    print('----------------------------------------------')

    dataSummary('Age', 'age')
    dataSummary('BMI', 'BMI')
    dataSummary('Glucose', 'glucose')
    dataSummary('Insulin', 'insulin')
    dataSummary('HOMA', 'HOMA')
    dataSummary('Leptin', 'leptin')
    dataSummary('Adiponectin', 'adiponectin')
    dataSummary('Resistin', 'resistin')
    dataSummary('MCP.1', 'MCP.1')

    dataProperties(df)
    dataMissing(df)
    print('----------- The normalised dataset -----------')
    print(dataNormalisation(df))
    print('----------------------------------------------')
    dataPlotting(df)
