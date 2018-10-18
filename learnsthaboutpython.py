import json
import numpy as np


def myfunc(n):
    '''
    type: n - a positive integer
    rtype: a numpy array
    '''
    ## TODO
    a = 0.5*np.arange(n)
    chan = np.arange(0, n, 2)
    le = np.arange(1, n, 2)
    a[le] = -1
    shape = chan.shape[0]
    min = 2 - shape * 0.5
    arr = np.arange(2, min, -0.5)
    print(arr)
    j = 0
    for i in chan:
        print(arr[j])
        a[i] = arr[j]
        j += 1
    return a
    ## --- end TODO ---


print(myfunc(12))


# with open("sub.json") as data:
#     data_elec = json.load(data)
#     for i in data_elec:
#         print(i['review_body'])
#     with open("sub2.json", 'w') as writ:
#         json.dump(data_elec, writ)
