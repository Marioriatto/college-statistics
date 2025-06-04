def variance(sample : list):
    mean = (sum(sample)/len(sample))
    return { "realVariance" : [abs(i - mean) for i in sample], "estimatedVariance" : (sum([( i - mean ) ** 2 for i in sample]) / ( len(sample) - 1 ))}
