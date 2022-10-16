"""  main.py """
import matplotlib.pyplot as plt
from  piv import piv,PIVInput,readImagePIV, output



S=PIVInput()


for i in S.nImg:
  S.setImages(readImagePIV(S,i)  )
  D=piv(S)
  output(S,D,i)
plt.imshow(D.V)
#plt.contourf(Imgs[1])
plt.show()
