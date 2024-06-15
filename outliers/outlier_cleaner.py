#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    x = 0
    for net_worth in net_worths:
        error = net_worth - predictions[x]
        if abs(error) <= 81:
            cleaned_data.append((ages[x], net_worth, error))
        x += 1

    print(cleaned_data)
    print(len(cleaned_data))
    return cleaned_data
