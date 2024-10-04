import shelve

import dill
dill.load_session('globalsave.pkl')


'''filename = 'shelve.out'

my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()'''

print(mor)