"""  main.py """
from  piv import PIV,PIVInput,ReadImage_PIV, Output
print('start')
flagProf=False
if flagProf:
  import cProfile

#from vprof import runner non mi funziona lanciare vprof -c h main.py
# al posto di h si può usare pmh (p profile, m memory e h heatmap) ogni lettera lancia di nuovo tutto
#import matplotlib.pyplot as plt

#runner.run(PIV2, 'cmhp', args=(), host='localhost', port=8000)
S=PIVInput()


for i in S.NImg:
  S.setImages(ReadImage_PIV(S,i)  )
  if flagProf:
    cProfile.run('D=PIV(S)')
  else:
    D=PIV(S)
  Output(S,D,i)
#plt.imshow(D.U)
#plt.contourf(Imgs[1])
#plt.show()
