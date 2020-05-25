#******************************************************************************
# Capital One Auto Loan Payment Anomaly Detection System
#
#UT Design Authors:
#    Andre Brown
#    Ken Smith
#    Vanessa Webb
#    Stephanie Zang
#    
#Date: 
#    December 1, 2017
#    
#Version: 
#    1.0
#
#Description:
#    The payment anomaly detection system uses rolling window analysis to 
#    predict the one-step out-of-sample volume and range of volume, based on 
#    the input time frequency and window size, for each of the three dispositions 
#    of each of the six services and for the total volume of each of the six 
#    services. 
#    
#    Using rolling windows allowed us to make predictions of volume count for a 
#    certain point in time based off of the mean average of the volumes of past 
#    consecutive points in time. For example, if the data was resampled to be 
#    bucketed by a time period of 15 minutes, and the window size was four, then 
#    the prediction of volume of API calls for the next 15 minutes would be 
#    taken from the average of the past four 15-minute periods, or the volume of 
#    the past hour. The rolling mean for each time unit was then used to 
#    calculate each rolling standard deviation, which was used to calculate the 
#    expected minimum and maximum volume for each time unit based on a given 
#    margin of error. This range of volume gives the one-step-ahead prediction.
#
#    The calculated range of volume for each time unit is then compared to 
#    actual observation. If the actual observation falls outside of the 
#    estimated range, then it is marked as a potential anomaly. 
#
#Limitations: 
#    The limitation of our anomaly detection system is that it is a model that 
#    requires the API call logs to be saved as a CSV file before we can import 
#    and parse the information in the program. In order to make one-step-ahead 
#    predictions continuously, the user would have to export the API call log 
#    from Splunk into this program at regular intervals. Therefore, our model 
#    would need to be modified in the future so that it could directly 
#    integrate with the API call event logging system.
#
#Input files:
#    "History data 7-25 to 10-14 all payments.csv" Contains data provided by Capital One.
#    "potential_anomalies.csv" Contains a list of data points assumed to be anomalies.
#    
#Output files:
#    "performance_metrics.csv" Contains various information on the performance of the program after the program has run.
#    "next_predictions.csv" Contains a list of prediction values made by the program after the program has run.
    
#******************************************************************************

import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean
import statistics
from pandas import read_csv
import time
import os
from sklearn.metrics import confusion_matrix

#Ignores any warnings
warnings.filterwarnings('ignore')

# get the start time to calculate how long it takes to run the model
start_time = time.clock()


# serviceNamesList determines the number and ordering of the service subsets that 
# will later be partitioned from the overall dataset. 
serviceNamesList=('loans-autoloans-payment-amounts-app', 
          'loans-autoloans-payment-dates-app',
          'loans-autoloans-paymentinstructions', 
          'loans-autoloans-paymentinstructions-app', 
          'loans-autoloans-paymentplan-app', 
          'loans-autoloans-paymentplanoptions-app')


#******************************************************************************

# Read specific columns from a CSV file into a dataframe, parse the _time.
# Change entries with a disposition of "NONE" to "FAILURE".

def readDataset(path, col_names):
        
    # read CSV into a DataFrame
    entireDataset = read_csv(path, usecols=col_names, engine='python')
    
    # parse _time column to remove extraneous characters
    # the original timestamp format is: 2017-07-25T00:00:00.000-0500
    # parse so that it looks like:      2017-07-25 00:00:00
    entireDataset['_time'] = entireDataset['_time'].astype(str) # change type to string
    entireDataset['_time'] = entireDataset['_time'].str.replace('T',' ')
    entireDataset['_time'] = entireDataset['_time'].apply(lambda x: x[:19])
    
    # convert _time column to datetime format
    entireDataset['_time'] = pd.to_datetime(entireDataset['_time'])
    
    # since we've observed that call events categorized as "none" are actually failures, 
    # change those with "none" to "failure"
    entireDataset.loc[entireDataset['Disposition']=='NONE', 'Disposition'] = 'FAILURE'

    return entireDataset




#******************************************************************************

#Method that converts the input dataframe into bucketed rows of the prescribed frequency
    #originalDF: dataframe to be converted
    #frequency: 'T' for minutes, 'H' for hours, 'D' for days
    #showInfo: shows head and tail of the dataset for each step of the conversion process
    #infoLength: length of head and tail for showInfo
    
def formatDF(originalDF, frequency = 'T', showInfo = False, infoLength = 10):
    df = originalDF.copy()
    
    #Makes 1 row for each datetime index and adds together any overlapping 
    #values (missing rows still present)
    dfGrouped = df.groupby('_time').sum()
    
    #Makes a minutely datetime index with the same range as dfGrouped, but 
    #without any missing rows
    firstDate = str(list(dfGrouped.index)[0])
    lastDate = str(list(dfGrouped.index)[-1])
    dateRangeMin = pd.date_range(start = firstDate, end = lastDate, freq = 'T')
    
    #Assigns values from dfGrouped to a dataframe without any missing rows, 
    #and fills missing values with 0 
    finalDF = dfGrouped.copy().reindex(dateRangeMin,fill_value=0)
    
    #If frequency is not default(minutely), re-buckets data to designated frequency
    if(frequency != 'T'):
        finalDF = finalDF.copy().resample(frequency).sum()
    
    # Since the '____Anomaly' columns are only supposed to have 0 or 1 to 
    # represent being anomalous, if the value in the columns are greater than 
    # 1 due to rows with the same timestamp being summed together, change the 
    # value back to 1.
    finalDF.loc[finalDF['successAnomaly'] > 1, 'successAnomaly'] = 1 
    finalDF.loc[finalDF['policyAnomaly'] > 1, 'policyAnomaly'] = 1 
    finalDF.loc[finalDF['failureAnomaly'] > 1, 'failureAnomaly'] = 1 
    finalDF.loc[finalDF['allAnomaly'] > 1, 'allAnomaly'] = 1 
    
    #Shows conversion process    
    if(showInfo):
        outLength=infoLength
        print('\noriginal head')
        print(df.head(outLength))
        print('\noriginal tail')
        print(df.tail(outLength))
        print('\nhead after grouping')
        print(dfGrouped.head(outLength))
        print('\ntail after grouping')
        print(dfGrouped.tail(outLength))
        print('\nconverted head')
        print(finalDF.head(outLength))
        print('\nconverted tail')
        print(finalDF.tail(outLength))
        
    return finalDF


#******************************************************************************
#Method that removes certain columns so dataframe can be plotted clearly.
    
    #completeDF: dataframe to be converted. Should be a dataframe produced by formatDF function
    #column: The one column that won't be removed. 'Success' by default, can also be 
        #'Failure', 'Policy', or 'All'
        
def getOneColumn(completeDF, column='SUCCESS'):
    oneColumnDF = completeDF.copy()

    if(column == 'ALL'):
        oneColumnDF['ALL'] = oneColumnDF.sum(axis=1)
        
    oneColumnDF = oneColumnDF[[column]]
    
    return oneColumnDF

#******************************************************************************
      
# Obtains rolling means and rolling standard deviations for each time interval, 
# and returns a dataframe with mean, min, and max columns.
    
def RollingWindow(DataFrame, window, frequency, stdMarginMultiplier):
    if not window:
        return 'No window value was assigned.'
    
    series = DataFrame.T.squeeze()
    
    predictions = list()
    stddev = list()
    
    # walk-forward validation
    history = [x for x in series[:window]] # get the first window

    
    remainder = len(series)-window # length of series minus window size
    for i in range(remainder):
        # make prediction
        yhat = int(round(mean(history[-window:])))# get the mean rounded
        predictions.append(yhat)
        sd = int(round(statistics.stdev(history[-window:]))) # get the standard deviation rounded
        stddev.append(sd)
        #Add the next actual observation to history
        history.append(series[i+window])
        
    # prediction for one future (out-of-sample) datapoint
    yhat = int(round(mean(history[-window:]))) # get the mean rounded
    predictions.append(yhat)
    sd = int(round(statistics.stdev(history[-window:]))) # get the standard deviation rounded
    stddev.append(sd)

    
    # add timestamps back on
    
    td = pd.Timedelta(frequency)
    lastDate = series.index[-1] + td # time of the one-step-ahead future prediction
    
    # if the user enters a window size >= length of dataset, then only the 
    # one-step-ahead future prediction will be returned
    if (window < len(series)):
        firstDate = series.index[window]
        predictionsLen = len(series) - window + 1
    else:
        firstDate = lastDate
        predictionsLen = 1
    
    # calculate max/min 
    stddev = np.array(stddev) * stdMarginMultiplier
    stdMin = np.array(predictions) - stddev
    stdMax = np.array(predictions) + stddev
    
    # convert mean, max, and min numpy arrays to dataframes
    stdMin = pd.DataFrame(stdMin.reshape(predictionsLen,1), columns = ['Min'])
    num = stdMin._get_numeric_data()#If min values are negative, convert them to 0
    num[num < 0] = 0
    
    stdMax = pd.DataFrame(stdMax.reshape(predictionsLen,1), columns = ['Max'])
    
    predictions = pd.DataFrame(np.array(predictions).reshape(predictionsLen,1), columns = ['Mean'])
    
    # create complete predictions dataframe with mean, min, and max columns
    predictions = pd.concat([predictions, stdMin, stdMax], axis=1)
    
    #reset index
    dateRangeMin = pd.date_range(start=firstDate, end=lastDate, freq=frequency)    
    predictions = predictions.set_index(dateRangeMin)
    
    return  predictions



#******************************************************************************

#Method for plotting rolling mean, rolling standard deviation, min/max values, 
#and anomaly points of any converted dataframe.
    
    #timeseriesDF: dataframe to be plotted
    #windowSize: size of the window for the rolling mean and standard deviation. Set to 12 by default
    #max/minMultiplier: Multiplies standard deviation by the given number to increase min/max range

# Returns a DataFrame of predicted anomalies.
    
def plotDF(service_name, disposition, timeseriesDF, predictions, window, 
           frequency, stdMarginMultiplier, showGraphs):
    
    #Gets rolling mean (predicted volume), min, and max as a dataframe with datetime index
    meanDF = predictions['Mean'].copy()
    minDF = predictions['Min'].copy()
    maxDF = predictions['Max'].copy()
    
    #Makes a dataframe with Actual values, Min values, and Max values as columns
    actualMinMaxDF = pd.concat([timeseriesDF, minDF, maxDF], axis=1)
    actualMinMaxDF.columns = ['Actual', 'Min', 'Max']
    
    #Creates 'isAnomaly' column for actualMinMaxDF
    #If actual value is outside min/max, isAnomaly = actual value.
    #If actual value is inside min/max range, isAnomaly = 0
    actualMinMaxDF['isAnomaly'] = actualMinMaxDF.apply(lambda x: x['Actual']  
                  if (x['Actual'] < x['Min'] or x['Actual'] > x['Max']) else np.NaN, axis=1)
    
    
    #Makes a dataframe with datetime index and only the isAnomaly column
    anomalyDF = actualMinMaxDF.copy()
    del anomalyDF['Actual']
    del anomalyDF['Min']
    del anomalyDF['Max']
    
    
    #Gets a plot of the actual values
    if (showGraphs == True):
        plotDF = timeseriesDF.plot(style='-k', 
            title = service_name + " - " + disposition + "\n" + 
            "Rolling Mean, Min/Max Range, and Potential Anomalies")
    
        rollingMean = meanDF.plot(style='-b', ax=plotDF)#Plot for rolling mean

#        minPlot = minDF.plot(style='-c', ax = plotDF)#Plot for min values
#        maxPlot = maxDF.plot(style='-m', ax = plotDF)#Plot for max values
    
        # fill area between Max and Min
        plotDF.fill_between(maxDF.index, maxDF.values, minDF.values, facecolor='magenta', alpha=0.3)
        
        #Plots everything (ax parameter chains all the plots together to be shown in 1 figure)
        anomalyPlot = anomalyDF.plot(style='xr', ax=plotDF)
        
        plotDF.legend(["Actual Vol", "Expected Vol", "Potential Anomaly", "Accepted Vol Range"])
        plt.show()
 

    # Prepare a series of predicted anomalies to be used in the calculation of 
    # performance metrics. For a given timestamp, 1 means that there is at least 
    # one anomaly, 0 means that there are none.
    predictedAnomalies = anomalyDF.copy()
    predictedAnomalies[predictedAnomalies > 0] = 1 #If there are anomalies, change to 1
    predictedAnomalies[np.isnan(predictedAnomalies)] = 0 #If there aren't anomalies, change to 0
    predictedAnomalies = predictedAnomalies.astype(int)
    
    return predictedAnomalies




#******************************************************************************

# Runs the test for the prescribed confidence level, bucketing frequency, and window size.
# Calculates and saves performance metrics to a DataFrame.
    
def runTest(serviceList, metricsReportPath, nextPredictionsPath, 
            frequency, window, stdMarginMultiplier, showGraphs):   
    

    
    categoryList = ['ALL','SUCCESS','POLICY','FAILURE']
    anomalyList = ['allAnomaly', 'successAnomaly', 'policyAnomaly', 'failureAnomaly']

    metrics_col_names = ['Confidence','Frequency','WindowSize',
                 'Service','Category','TP','TN','FP','FN',
                 'Accuracy','TruePositiveRate','FalsePositiveRate',
                 'Precision','Prevalence','FScore']
    
    predictions_col_names = ['Confidence','Frequency','WindowSize',
                 'Service','Category','StartTime','EndTime',
                 'Mean','Min','Max']
    
    # Create an empty DataFrame with the column names for the performance 
    # metrics report.
    metricsDF = pd.DataFrame(columns=metrics_col_names)
    
    # Create an empty DataFrame with the column names for the one-step-ahead 
    # predictions file.
    oneStepPredictionsDF = pd.DataFrame(columns=predictions_col_names)
    
    
    #############################################################
    # this is currently hardcoded
    confidence = 0.997
    #############################################################
    

    # for each of the services
    for i in range(len(serviceList)):
        
        # get the service name
        service_name = serviceList[i]['Service'].iloc[0]
        
        # bucket the service dataset by the frequency
        bucketedDF = formatDF(serviceList[i], frequency)
        
        
        # for each of the 4 categories
        for j in range(len(categoryList)):
            
            # remove unneeded columns
            oneColDF = getOneColumn(bucketedDF, categoryList[j])
            
            # perform rolling window, returns a DataFrame with predictions for 
            # mean, min, max.
            predictions = RollingWindow(oneColDF, window, frequency, stdMarginMultiplier)
            
            #*************************************
            # Add the one-step-ahead prediction to the list of predictions.
            
            # get span of time for the one-step-ahead expected volume range
            td = pd.Timedelta(frequency)
            startTime = predictions.index[-1]
            endTime = startTime + td
            
            predMean = predictions['Mean'].iloc[-1]
            predMin = predictions['Min'].iloc[-1]
            predMax = predictions['Max'].iloc[-1]
            
            # create a Series with performance metrics
            row = pd.Series([confidence, frequency, window, service_name, categoryList[j], 
                                 startTime, endTime, predMean, predMin, predMax], 
                            index=predictions_col_names)
            
            #append the Series to oneStepPredictionsDF
            oneStepPredictionsDF = oneStepPredictionsDF.append(row, ignore_index=True) 
            
            
            #*************************************
            # Return a DataFrame of predicted anomalies.
            # Also plot the data on a graph if 'showGraphs' is True.
            predictedAnomalies = plotDF(service_name, categoryList[j], 
                                        oneColDF, predictions, 
                                        window, frequency, stdMarginMultiplier, 
                                        showGraphs)
            

            #*************************************
            # set up confusion matrix
            
            # get actual anomalies
            y_true = bucketedDF[anomalyList[j]].astype(int)
            
            # Get all predicted anomalies except the last datapoint which 
            # doesn't have a prediction.
            y_pred = predictedAnomalies[:-1] 
            
            # Run confusion matrix
            # Returns true negative, false positive, false negative, true positive
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            
            #*************************************
            #Calculate performance metrics
        
            
            # total, actual yes, actual no, predicted yes
            total = tn + fp + fn + tp
            actual_yes = fn + tp
            actual_no = tn + fp
            predicted_yes = tp + fp
            
            # Accuracy 
            accuracy = (tp+tn)/total
            
            # True Positive Rate
            truePositiveRate = tp/actual_yes
            
            # False Positive Rate 
            falsePositiveRate = fp/actual_no
            
            # Precision 
            precision = tp/predicted_yes
            
            # Prevalence
            prevalence = actual_yes/total
            
            # F-score 
            fscore = 2*((truePositiveRate*precision)/(truePositiveRate+precision))
            
   
            # create a Series with performance metrics
            row = pd.Series([confidence, frequency, window, service_name, categoryList[j], 
                                        tp, tn, fp, fn, accuracy, truePositiveRate, 
                                        falsePositiveRate, precision, prevalence, fscore], 
                            index=metrics_col_names)
            
            #append Series to metricsDF
            metricsDF = metricsDF.append(row, ignore_index=True)
            
    #end of outer for loop


    #write one-step-ahead predictions to a csv file      

    # if file does not exist, write the header row
    if not os.path.isfile(nextPredictionsPath):
        oneStepPredictionsDF.to_csv(nextPredictionsPath, header=predictions_col_names)
    else: 
        oneStepPredictionsDF.to_csv(nextPredictionsPath, mode='a', header=False)        
    
    #write performance metrics to a csv file      

    # if file does not exist, write the header row
    if not os.path.isfile(metricsReportPath):
        metricsDF.to_csv(metricsReportPath, header=metrics_col_names)
    else:
        metricsDF.to_csv(metricsReportPath, mode='a', header=False)  
         
# end of runTest 
        
        
        
        
#******************************************************************************
#******************************************************************************
        
#Read the input dataset and partition it into the individual services. Create 
#one-hot columns for each of the three dispositions and multiply by the volume 
#count. Create columns to mark datapoints that are anomalous as specified in 
#the 'potential_anomalies.csv' file. To run the model and output performance 
#metrics and one-step-ahead predictions files, call the 'runTest' function 
#by inputting the desired time frequency and window size.

        
        
# Read the entire dataset into a DataFrame.
path = "History data 7-25 to 10-14 all payments.csv"   
 
col_names = ['_time','Service','Disposition','count']

entireDataset = readDataset(path, col_names)

# add services to a list of DataFrames
serviceList = []

# Separates out 'SUCCESS', 'POLICY', and 'FAILURE' from the Disposition column 
# into their own columns. Creates a list of the desired services with each 
# service having the columns 'Service', 'SUCCESS', 'POLICY', and 'FAILURE'.
for x in range(len(serviceNamesList)):
    # partition the dataset by the API names
    srvc = entireDataset.loc[entireDataset['Service']==serviceNamesList[x]]
    
    # set _time as the index
    srvc.set_index('_time', inplace=True) 
    
    # convert Disposition column to one-hot
    dum = pd.get_dummies(srvc['Disposition']) 
    
    # multiply each column by count
    if 'POLICY' in dum.columns:
        dum['POLICY'] = dum['POLICY'] * srvc['count']
    if 'SUCCESS' in dum.columns:
        dum['SUCCESS'] = dum['SUCCESS'] * srvc['count']
    if 'FAILURE' in dum.columns:
        dum['FAILURE'] = dum['FAILURE'] * srvc['count']
    if 'NONE' in dum.columns:
        dum['NONE'] = dum['NONE'] * srvc['count']
    
    # create a DataFrame with the Service, SUCCESS, POLICY, and FAILURE columns
    srvc = pd.concat([srvc['Service'], dum], axis=1)
    
    serviceList.append(srvc)
# end of for loop
  



# Add columns for marking actual anomalies. 'allAnomaly' is used to mark that 
# there is an anomaly regardless of disposition.
for x in range(len(serviceList)):
    serviceList[x]['successAnomaly'] = 0
    serviceList[x]['policyAnomaly'] = 0
    serviceList[x]['failureAnomaly'] = 0
    serviceList[x]['allAnomaly'] = 0


# Read the list of anomalies into a DataFrame.
anomaliesDataset = readDataset("potential_anomalies.csv", col_names)
anomaliesDataset.set_index('_time', inplace=True) 


# Group anomalies dataset by 'Service'
groupedAnomalies = anomaliesDataset.groupby('Service')


# For each row in the list of anomalies, set the corresponding row (in one of the 
# services) as anomalous by setting the 'allAnomaly' column and one of the 
# 'successAnomaly', 'policyAnomaly', and 'failureAnomaly' columns to 1.
for name, group in groupedAnomalies:

    for x in range(len(serviceList)): # for each service 
        
        #Get the name of the service
        service_name = serviceList[x]['Service'].iloc[0]
        
        # if the group's service name matches the current service 
        if name == service_name: 
            
            # Process each row in the group, set the '____Anomaly' column to 
            # true depending on the disposition and also set 'allAnomaly'.      
            for row in group.itertuples():

                if row.Disposition == 'SUCCESS':
                    serviceList[x].loc[row.Index, 'successAnomaly'] = 1
                if row.Disposition == 'POLICY':
                    serviceList[x].loc[row.Index, 'policyAnomaly'] = 1
                if row.Disposition == 'FAILURE':
                    serviceList[x].loc[row.Index, 'failureAnomaly'] = 1
                
                # set 'allAnomaly' column
                serviceList[x].loc[row.Index, 'allAnomaly'] = 1    



# Run tests on different time bucketing frequencies, window sizes, and 
# standard deviation margins. Set 'showGraphs' to True to view 24 graphs per
# test.
            
##NOTE 68-95-99 rule: Assuming data is normally distributed,
##When min/maxMultiplier=1, 68% of data should be within range
##When min/maxMultiplier=2, 95% of data should be within range
##When min/maxMultiplier=3, 99.7% of data should be within range          
            
            
# the csv file to write the performance metrics report to      
metricsReportPath = "performance_metrics.csv"

# the csv file to write the next predictions to      
nextPredictionsPath = "next_predictions.csv"   


#Run test to resample data in the specified time frequency, perform rollling 
#window analysis, create mean, min, and max predictions, create a list of 
#anomalies detected based off of the predictions, run performance metrics, and 
#output performance metrics and one-step-ahead predictions to CSV files. 
#Optionally, show 24 graphs for each of the four categories for each of the six 
#services.
runTest(serviceList, metricsReportPath, nextPredictionsPath, 
        frequency='15Min', window=4, stdMarginMultiplier=3, 
        showGraphs=True)


print("\nTime to finish model and run tests: %s seconds ---" % 
      (time.clock() - start_time))



