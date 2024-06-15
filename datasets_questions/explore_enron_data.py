#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
import numpy as np

enron_data = joblib.load(open(
    "C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\final_project\\final_project_dataset.pkl",
    "rb"))

# POI = []
# for i in enron_data:
#     if enron_data[i]["poi"]:
#         POI.append(i)
#
# for i in POI:
#     print(i)
#
# print(len(POI))

# if_found = False
# for i in enron_data:
#     if str(i) == 'COLWELL WESLEY':
#         print('Found: ' + str(i))
#         if_found = True
#         break
#
# if not if_found:
#     print('Not Found')


# POI = 'James Prentice'
# POI_upper = POI.upper()

# POI_salary = []
# for i in enron_data:
#     if enron_data[i]["salary"] != "NaN":
#         POI_salary.append(i)
#
#
# POI_email = []
# for i in enron_data:
#     if enron_data[i]['email_address']!="NaN":
#         POI_email.append(i)
#
# print(len(POI_salary))
# print(len(POI_email))

# total_payments = []
# for i in enron_data:
#     if enron_data[i]['total_payments'] == 'NaN':
#         total_payments.append(i)
#
# # print(len(enron_data) - len(total_payments))
# print(len(enron_data) + 10)
# print(len(total_payments) + 10)
# print(f'{((len(total_payments) + 10) / (len(enron_data) + 10)) * 100}%')

# POI = []
# POI_total_payments = []
# for i in enron_data:
#     if enron_data[i]['total_payments'] == 'NaN' and enron_data[i]['poi']:
#         POI_total_payments.append(i)
#     if enron_data[i]['poi']:
#         POI.append(i)
#
# print(len(POI_total_payments))
# print(len(POI) + 10)
# print(f'{((len(POI_total_payments)) / len(enron_data)) * 100}%')
# print(enron_data['ALLEN PHILLIP K'])

data = []
salary_over = []
j = 0
for i in enron_data:
    if enron_data[i]['exercised_stock_options'] == 'NaN':
        data.append([enron_data[i]['salary'], 0])
    elif enron_data[i]['salary'] == 'NaN':
        data.append([0, enron_data[i]['exercised_stock_options']])
    elif enron_data[i]['salary'] == 'NaN' and enron_data[i]['exercised_stock_options'] == 'NaN':
        data.append([0, 0])
    else:
        data.append([enron_data[i]['salary'], enron_data[i]['exercised_stock_options']])
    if enron_data[i]['salary'] != 'NaN' and enron_data[i]['salary'] >= 200000:
        salary_over.append(enron_data[i]['salary'])

data.append([200000, 1000000])

data_np = np.array(data)
print(data_np)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_np)
print(scaled_data)

value = float(7.47172158e-03)
value2 = 3.15198925e-03
print(value)
print(value2)
print(len(salary_over))
