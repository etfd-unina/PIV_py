"""  MainProfile.py """
from  piv import piv,PIVInput,readImagePIV, output
print('start')
flagProf=False
if flagProf:
  import cProfile

#import timeit as tt
#tt.timeit(lambda:'a=a.reshape(IWa.shape)',number=100000 )
# in caso di codici lunghi e indentati creare una stringa ed eseguirla
#starttime = tt.default_timer()
# a=a.reshape(IWa.shape)
#print("The time difference is :", tt.default_timer() - starttime)

#from vprof import runner non mi funziona lanciare vprof -c h main.py
# al posto di h si può usare pmh (p profile, m memory e h heatmap) ogni lettera lancia di nuovo tutto
#import matplotlib.pyplot as plt

#runner.run(PIV2, 'cmhp', args=(), host='localhost', port=8000)
S=PIVInput()


for i in S.nImg:
  S.setImages(readImagePIV(S,i)  )
  if flagProf:
    cProfile.run('D=PIV(S)')
  else:
    D=piv(S)
  output(S,D,i)
#plt.imshow(D.U)
#plt.contourf(Imgs[1])
#plt.show()
