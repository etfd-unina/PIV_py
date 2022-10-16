"""#pylint: disable=W0311"""
import os
from math import ceil
import timeit as tt
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.signal import convolve2d



FormatDouble=np.float64

class PIVInput:
  ''' input data for PIV'''
  def setImages(self, img):
    ''' set the images in the input class '''
    self.imgA= img[0]
    self.imgB= img[1]

  imgA= np.zeros((1,1))                           # Input images
  imgB= np.zeros((1,1))
  Wa=np.array([32, 16],dtype=np.intc)             # Interrogation window linear dimension
  Wb=np.array([16,  8],dtype=np.intc)             # Weighted Average window linear dimension
  Ov=np.array([16,  8 ],dtype=np.intc)             # Overlap
  nIt=2                                           # Number of final iterations
  univTh=2                                        # Universal median threshold
  univEps=0.1                                     # Universal median acceptable fluctuation level
  lV=10
  limV=np.array([-lV, lV,-lV, lV],dtype=np.intc)  # Limits for searching the maximum [Min  Max (u) Min Max (v)]
  flagWinA=0                                      # 2 Blackmann else TH
  flagWinB=2                                      # 2 Blackmann else TH
  flagOutPut=1                                    # 0 no output 1 text 2 plot 3 plot and text
  imgInterp=3                                     # Spline order for image interpolation
  velInterp=1                                     # Spline order for dense predictor interpolation

  # file names and option for input and output
  imgName=os.path.join( "./Img/MultiTest_")
  ext='.png'                                      # Extension
  nDig=4                                          # Number of digits
  nImg=range(0,1)               # image numbers (ex. 1:101)
  ROI=np.array([0, 64, 0, 96],dtype=np.intc)  # Region of Interest [Min  Max (row) Min Max (col)]
  #ROI=np.array([0, 1184, 0, 2080],dtype=np.intc)  # Region of Interest [Min  Max (row) Min Max (col)]
  outName='./Out/MultiTest_2IT_'

class PIVData:
  ''' Internal data for PIV'''
  Wa=32                       # window size
  Wb=32                       # filtering window size
  Ov=8                        # overlap
  xStart=np.zeros((1,1),dtype=np.intc)          # Starting coordinate of the IW
  yStart=np.zeros((1,1),dtype=np.intc)          # Starting coordinate of the IW
  X=np.zeros((1,1),dtype=np.intc)               # 2d grid
  Y=np.zeros((1,1),dtype=np.intc)               # 2d grid
  H=32                        # grid size
  W=32                        # grid size
  U=0*X                       # Displacement
  V=0*X                       # Displacement
  Fc=0*X                      # Local CC coefficient
  info=0*X                   # info on vector

def createGridandAlloc(S,data,it):
  ''' Build the grid and allocate memory '''
  data.Wa=S.Wa[min(it,S.Wa.size-1)]                      # set window size
  data.Wb=S.Wb[min(it,S.Wb.size-1)]                      # set filtering window size
  data.Ov=S.Ov[min(it,S.Ov.size-1)]                      # set overlap
  hImg,wImg=S.imgA.shape           # Image size
  data.xStart=np.arange(0,wImg-data.Wa+1,data.Ov,dtype=np.intc)          # Starting coordinate of the IW
  data.yStart=np.arange(0,hImg-data.Wa+1,data.Ov,dtype=np.intc)          # Starting coordinate of the IW
  data.X,data.Y=np.meshgrid(data.xStart+data.Wa/2,data.yStart+data.Wa/2)  # 2d grid
  data.H,data.W=data.X.shape                # grid size
  data.U=0*data.X    # allocating memory
  data.V=0*data.X
  data.Fc=0*data.X
  data.info=0*data.X
  return data
def blackman(Wa):
  ''' Blackman weighting window'''
  win = np.arange(1,Wa+1,dtype=FormatDouble)/(Wa+1)
  return 0.42 - 0.5*np.cos(2*np.pi*win) + 0.08*np.cos(4*np.pi*win)
def initNormCrossCorrWin(Wa):
  ''' Evaluate the normalized Cross Correlation '''
  win=blackman(Wa)[np.newaxis]
  win=win.T@win
  fftWin=np.fft.rfftn(win)
  return (win , fftWin)

def normCrossCorrWin(IWa,IWb,Win,FW):
  ''' Evaluate the normalized Cross Correlation '''
  dum=1/IWa.size
  #ma=IWa.flatten().mean()
  ma=IWa.ravel().sum()*dum#faster than mean
  IWa=IWa-ma
  mb=IWb.ravel().sum()*dum#faster than meanb
  IWb=IWb-mb
  IWa=IWa/np.sqrt( (Win*(IWa**2)).ravel().sum())
  num=np.fft.irfftn(np.fft.rfftn(IWb)*np.conj(np.fft.rfftn(IWa*Win)))
  den=np.sqrt(np.fft.irfftn(np.fft.rfftn(IWb*IWb)*np.conj(FW)))
  return num/den

def normCrossCorr(IWa,IWb):
  ''' Evaluate the normalized Cross Correlation '''
  dum=1/IWa.size
  ma=IWa.ravel().sum()*dum#faster than mean
  IWa=IWa-ma
  mb=IWb.ravel().sum()*dum#faster than meanb
  IWb=IWb-mb
  den=np.sqrt(np.square(IWa).ravel().sum()*np.square(IWb).ravel().sum())
  # cross-correlation theorem
  return  np.fft.irfftn(np.fft.rfftn(IWb)*np.conj(np.fft.rfftn(IWa)))/den


def gaussInt(a):
  ''' Three point Gaussian interpolation
  If at least one of the values is negative perform a standard parabolic interpolation
  '''
  flag=False # simple parabolic interpolation
  if np.all(a>0):  #perform Gaussian interpolation
    a=np.log(a)
    flag=True


  d2=2*a[1] -a[0]-a[2]
  if d2 <=0: # this is not a maximum
    return (0, np.exp(a[1])) if flag else  (0, a[1])
  d=a[2]-a[0]
  dx=(a[2]-a[0])/(2*d2)
  if np.abs(dx)>0.5: # this is not a maximum
    return (0, np.exp(a[1])) if flag else  (0, a[1])

  val=a[1]+dx*d*0.25
  if flag: #Gaussian interpolation
    val=np.exp(val)
  return dx, val

def dispFromCC(R):
  ''' Finds the CC map maximum and then the sub pixel displacement '''
  wa=R.shape[0]
  limit=np.floor(wa/4).astype(int)           # One quarter rule plus one
  #  Limiting to a smaller (known) value can increase the robustness
  #RS=R[Limit:-Limit#Wa+1,Limit:-Limit%Wa+1]                 # Smaller CC Map
  ii=np.arange(-limit,limit+1)   # Index of the points to be extracted
  #RS=R[-Limit+1:+Limit+1,-Limit:+Limit+1]                 # Smaller CC Map
  rS=R[np.ix_(ii,ii)]
  #I=RS.argmax()           # Search for maximum
  ijm = np.unravel_index(rS.argmax(), rS.shape)# Search for maximum position
  fc=rS[ijm]
  im=ii[ijm[0]]                  # Position of the maximum in the full CC map
  jm=ii[ijm[1]]

  dy, valY=gaussInt(    R[np.arange(im-1,im+2)%wa,jm])
  v=im+dy
  dx, valX=gaussInt(    R[im,np.arange(jm-1,jm+2)%wa])
  u=jm+dx
  #if(Valx*Valy>100):    print(Valx*Valy)
  fc=np.sqrt(valX*valY)
  return (u, v,fc)


def dispFromCorr(S,D,it):
  ''' DispFromCorr Evaluate the displacement field by using a Cross Correlation approach
   Used in the predictor and corrector step
  '''
  if S.flagWinA==2: # weighting windows call NormCrossCorrWin
    (win , fftWin)=initNormCrossCorrWin(D.Wa)
    def funCC(IWa,IWb):
      return normCrossCorrWin(IWa,IWb,win,fftWin)
  else: # standard normalized cross correlation
    funCC=normCrossCorr

  for i in range(0,D.yStart.size):
    ie=D.yStart[i]+D.Wa                         # final index for Window extraction
    for j in range(0,D.xStart.size):
      je=D.xStart[j]+D.Wa                       # final index for Window extraction
      if it==0:  # First iteration -> extract the IWs from raw images
        r=funCC(S.imgA[D.yStart[i]:ie,D.xStart[j]:je],S.imgB[D.yStart[i]:ie,D.xStart[j]:je]) # Normalized CC
      else: # following iterations -> extract the IWs from deformed images
        r=funCC(D.imgA[D.yStart[i]:ie,D.xStart[j]:je],D.imgB[D.yStart[i]:ie,D.xStart[j]:je]) # Normalized CC
      (D.U[i,j],D.V[i,j],D.Fc[i,j])=dispFromCC(r) # Displacement vector
  return D
# Replace the outliers
def substituteOutliers(IndGood,X,Y,U):
  ''' Replace the outliers '''
  F=LinearNDInterpolator(list(zip(X[IndGood], Y[IndGood])),U[IndGood])# Interpolating the "good" vectors
  U[np.logical_not(IndGood)] = F(X[np.logical_not(IndGood)],Y[np.logical_not(IndGood)])
  indFinite=np.isfinite(U) # in some case return nan switch to nearest neighborhood
  if not np.all(indFinite):
    F=NearestNDInterpolator(list(zip(X[indFinite], Y[indFinite])),U[indFinite])
    U[np.logical_not(indFinite)] = F(X[np.logical_not(indFinite)],Y[np.logical_not(indFinite)])

def  medianWS(U,V,info,ker,univEps,univTh):
  ''' Universal outlier detector Westerweel and Scarano (2005)
   find outliers and substitute them with the median value
  '''
  h,w=U.shape
  univTh2=univTh*univTh
  for i in range(0,h):
    iMin,iMax=max(0,i-ker),min(h-1,i+ker) #Index for extraction of neighboring points
    #ii=max([1,i-ker]):min([H,i+ker]) #Index for extraction of neighboring points
    for j in  range(0,w):
      jMin,jMax=max(0,j-ker),min(w-1,j+ker) #Index for extraction of neighboring points
      #jj=max([1,j-ker]):min([W,j+ker])# Index for extraction of neighboring points
      nEl=(jMax-jMin+1)*(iMax-iMin+1)-1

      unb=np.empty(nEl)
      vnb=np.empty(nEl)
      # Extraction
      c=0
      for i1 in range(iMin,iMax+1):
        for j1 in range(jMin,jMax+1):
          if(i1==i and j1==j):
            continue # jump the central point
          unb[c]=U[i1,j1]
          vnb[c]=V[i1,j1]
          c=c+1
      mu=np.median(unb)
      rm=np.median(np.fabs(unb-mu))
      erru=abs(U[i,j]-mu)/(rm+univEps)

      mv=np.median(vnb)
      rm=np.median(np.fabs(vnb-mv))
      errv=abs(V[i,j]-mv)/(rm+univEps)
      if (erru**2+errv**2)>univTh2:                    # WS Check
        info[i,j]=0                  # wrong vector
        U[i,j]=mu                     # Substitution
        V[i,j]=mv                     # Substitution
  return U,V,info
def validation(S,D):
   ''' Perform the validation step
   First limit the displacement accordingly with S.limV
   The substitution of the outliers is made by using substituteOutliers
   medianSW perform the outlier detection and substitution
   '''
   indGood=np.logical_and.reduce((D.U<S.limV[1], D.U>S.limV[0] , D.V< S.limV[3] , D.V>S.limV[2]))
   D.info=indGood   # IndGood is false for outliers
   if not np.all(indGood):
    substituteOutliers(indGood,D.X,D.Y,D.U) # Replacing the outliers
    substituteOutliers(indGood,D.X,D.Y,D.V)


   ker=1         # half dimension of the kernel for now equal to 1
   D.U,D.V,D.info=medianWS (D.U,D.V,D.info,ker,S.univEps,S.univTh)
   return D
def writeToVideo(S,D,it):
  ''' Write partial output to video '''
  if S.flagOutPut%2==1:
    it=it-S.Ov.size+1
    numVect=D.X.size
    numWrong=100*(numVect-D.info.ravel().sum())/numVect
    fcMed=D.Fc.ravel().mean()
    print(f'It={it:-3d}  Wa={D.Wa:3d} N={D.X.shape[0]:>4d}X{D.X.shape[1]:<4d} Outliers={numWrong:4.1f}%  Fc={fcMed:8.4f}')
  if S.flagOutPut//2==1:#Plot to be seen
    #figure(1),    clf,    imagesc(D.V),    colorbar
    #figure(2),clf, plot(mean(D.V))
    #figure(3),clf,imagesc(D.U),colorbar
    #figure(4),clf ,plot(mean(D.U))
    pass
def densePredictorandDeform(D,S):
  ''' Build the dense predictor and deform the images '''
  #def Ave(x):     return x.flatten().sum()/x.size
  #MeshGridMat=@(A)  meshgrid(1:size(A,2),1:size(A,1))
  #[Xd,Yd]=np.mgrid[0:S.ImgA.shape[0],0:S.ImgA.shape[1]]

  off=0.5
  xd=np.arange(off,S.imgA.shape[0]+off)
  yd=np.arange(off,S.imgA.shape[1]+off)
  F= RectBivariateSpline( D.Y[:,1], D.X[0],D.U,kx=S.velInterp,ky=S.velInterp)

  D.Ud=F(xd,yd,grid=True)
  F= RectBivariateSpline( D.Y[:,1], D.X[0],D.V,kx=S.velInterp,ky=S.velInterp)
  D.Vd=F(xd,yd,grid=True)

  Ym,Xm=np.mgrid[off:S.imgA.shape[0]+off,off:S.imgA.shape[1]+off]

  F= RectBivariateSpline(xd, yd, S.imgA,kx=S.imgInterp,ky=S.imgInterp)
  D.imgA=F((Ym-D.Vd/2).ravel(),(Xm-D.Ud/2).ravel(),grid=False)
  D.imgA=D.imgA.reshape(S.imgA.shape)
  F= RectBivariateSpline(xd, yd, S.imgB,kx=S.imgInterp,ky=S.imgInterp)
  D.imgB=F((Ym+D.Vd/2).ravel(),(Xm+D.Ud/2).ravel(),grid=False)
  D.imgB=D.imgB.reshape(S.imgB.shape)

  return D

def  weightedAverage(D,flagWinB):
  ''' Perform a weighted average of the predictor and sum it to the corrector
  Only top hat and blackman WW are considered
  '''
  if flagWinB==2: #
    We=blackman(D.Wb)
    We=np.outer(We,We)
    We=We/We.ravel().sum()
  else:
    We=np.ones((D.Wb,D.Wb),dtype=FormatDouble)/(D.Wb**2)


  fv = convolve2d(D.Vd,We,mode='same')
  fu = convolve2d(D.Ud,We,mode='same') #shifted with respect to matlab
  for i in range(0,D.yStart.size):
    for j in range(0,D.xStart.size):
      D.V[i,j]=D.V[i,j]+fv[ceil(D.Y[i,j]),ceil(D.X[i,j])]
      D.U[i,j]=D.U[i,j]+fu[ceil(D.Y[i,j]),ceil(D.X[i,j])]
  return D

def piv(S):
  ''' A simple implementaion of IDM for PIV
   Astarita 2021, in vki Lecture series
   Fundamentals and recent advances in Particle Image Velocimetry and Lagrangian Particle Tracking
  '''
  starttime = tt.default_timer()
  it=0
  D=PIVData()
  D=createGridandAlloc(S,D,it)
  D=dispFromCorr(S,D,it)
  D=validation(S,D)
  writeToVideo(S,D,it)
  for it in range (1,S.nIt+S.Ov.size):
    D=densePredictorandDeform(D,S)
    D=createGridandAlloc(S,D,it)
    D=dispFromCorr(S,D,it)
    D=weightedAverage(D,S.flagWinB)
    D=validation(S,D)
    writeToVideo(S,D,it)
  print("PIV total time is:", tt.default_timer() - starttime)
  return D



def readImagePIV(S,i):
  ''' read the images '''
  nomeImg=os.path.join(  f"{S.imgName}{i:04d}a{S.ext}")
  Ia=np.asarray(Image.open (nomeImg),dtype=FormatDouble)[S.ROI[0]:S.ROI[1],S.ROI[2]:S.ROI[3]]

  nomeImg=os.path.join(  f"{S.imgName}{i:04d}b{S.ext}")
  Ib=np.asarray(Image.open (nomeImg),dtype=FormatDouble)[S.ROI[0]:S.ROI[1],S.ROI[2]:S.ROI[3]]
  return [Ia,Ib]



def writePlt(NomeFileOut,Mat,Titolo,NomeVar,TitoloZona):
  ''' write tecplot binary files'''
  nVar=len(Mat)
  h,w=Mat[0].shape
  tecIntest=b'#!TDV71 '

  def writeLong(l):
    f.write(np.int32(l))
  def writeFloat32(fl):
    f.write(np.float32(fl))
  def writeString(s):
    for c in s:
      f.write(np.int32(c))
    return
  #apertura file binario di output .plt
  f=open(NomeFileOut,"wb")

  #scrittura nel file di output del TEST
  f.write(tecIntest)
    #scrittura nel file di output di 1 (ordine dei bytes BO???)
  #Lungo=np.int32(1)
  #f.write(Lungo)
  writeLong(1)
  #scrittura del titolo
  writeString(Titolo)

  #scrittura numero e nome delle variabili
  writeLong(nVar)
  writeString(NomeVar)
  writeFloat32(299.0)
  #ZONE NAME
  writeString(TitoloZona)
  #scrittura del BLOCK POINT
  writeLong(1)
  #scrittura del COLORE
  writeLong(-1)
  #scrittura nel file di output della larghezza e altezza img W e H
  writeLong(w)
  writeLong(h)
  #scrittura nel file di output della dimensione Z
  writeLong(1)
  #scrittura nel file di output di 357.0 (bo????)
  writeFloat32(357.0)
  #scrittura nel file di output di 299.0 (bo????)
  writeFloat32(299.0)
  #scrittura nel file di output di 0 (bo????)
  writeLong(0)
  #sizeof variabili
  for _ in range (0,nVar):
    writeLong(1)

  #scrittura nel file di output delle matrici x,y,u,v,up,vp (variabili)
  writeFloat32(np.ascontiguousarray(np.transpose(Mat,(1,2,0))))

  f.flush()
  f.close()

  return
def output(S,D,i):
  ''' Interface for writePlt '''
  nomeFileOut=os.path.join(  f"{S.outName}{i:0{S.nDig:d}d}.plt")
  titoloZona=bytes(nomeFileOut+'\0','utf-8')
  titolo=b'b16\0'
  nomeVar=b'X\0Y\0U\0V\0Fc\0info\0'
  varia=[ D.X,    D.Y,    D.U  ,  D.V   , D.Fc, D.info, ]
  writePlt(nomeFileOut,varia,titolo,nomeVar,titoloZona)


if __name__ == "__main__":
  Inp=PIVInput()
  for nImg in Inp.nImg:
    Inp.setImages(readImagePIV(Inp,nImg)  )
    #if flagProf:      cProfile.run('D=PIV(S)')    else:
    Di=piv(Inp)
    output(Inp,Di,nImg)
