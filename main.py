import numpy as np
import pandas as pd

df = pd.read_excel('clinical_dataset.xlsx')


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
    print('-------------------------------------------')


if __name__ == '__main__':
    print(df)

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
