import shelve
import dill

mor=5
m=66
t="morm"

                            #pip install dill --user
filename = 'globalsave.pkl'
dill.dump_session(filename)


'''filename = 'shelve.out'
my_shelf = shelve.open(filename, 'n')  # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()'''