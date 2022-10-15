"""#pylint: disable=W0311"""
import os
import timeit as tt
#tt.timeit(lambda:'a=a.reshape(IWa.shape)',number=100000 )
# in caso di codici lunghi e indentati creare una stringa ed eseguirla
#starttime = tt.default_timer()
# a=a.reshape(IWa.shape)
#print("The time difference is :", tt.default_timer() - starttime)
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.signal import convolve2d



FormatDouble=np.float64
"""  input data """
class PIVInput:

  def setImages(self, img):
    self.ImgA= img[0]
    self.ImgB= img[1]

  ImgA= np.zeros((1,1))                           # Input images
  ImgB= np.zeros((1,1))
  Wa=np.array([32, 16],dtype=np.intc)             # Interrogation window linear dimension
  Wb=np.array([16,  8],dtype=np.intc)             # Weighted Average window linear dimension
  Ov=np.array([16,  8 ],dtype=np.intc)             # Overlap
  Nit=2                                           # Number of final iterations
  UnivTh=2                                        # Universal median threshold
  UnivEps=0.1                                     # Universal median acceptable fluctuation level
  lV=10
  LimV=np.array([-lV, lV,-lV, lV],dtype=np.intc)  # Limits for searching the maximum [Min  Max (u) Min Max (v)]
  FlagWinA=0                                      # 2 Blackmann else TH
  FlagWinB=2                                      # 2 Blackmann else TH
  FlagOutPut=1                                    # 0 no output 1 text 2 plot 3 plot and text
  ImgInterp=1                                     # Spline order for image interpolation 
  VelInterp=1                                     # Spline order for dense predictor interpolation 

  # file names and option for input and output 
  ImgName='C:\\desk\\Attuali\\ricerca\\SynImgGen_Matlab\\PIV_vki\\img\\MultiTest_'
  Ext='.png'                                      # Extension
  Ndig=4                                          # Number of digits
  NImg=range(0,1)               # image numbers (ex. 1:101)
  ROI=np.array([0, 64, 0, 96],dtype=np.intc)  # Region of Interest [Min  Max (row) Min Max (col)]
  #ROI=np.array([0, 1184, 0, 2080],dtype=np.intc)  # Region of Interest [Min  Max (row) Min Max (col)]
  
  OutName='..\\Out\\MultiTest_vki_BL32BL3_10IT_'

class PIVData:
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
  Info=0*X                   # info on vector
   
def CreateGridandAlloc(S,D,it):
   # Build the grid and allocate memory
   
   D.Wa=S.Wa[min(it,S.Wa.size-1)]                      # set window size
   D.Wb=S.Wb[min(it,S.Wb.size-1)]                      # set filtering window size 
   D.Ov=S.Ov[min(it,S.Ov.size-1)]                      # set overlap
   Himg,Wimg=S.ImgA.shape           # Image size
   D.xStart=np.arange(0,Wimg-D.Wa+1,D.Ov,dtype=np.intc)          # Starting coordinate of the IW
   D.yStart=np.arange(0,Himg-D.Wa+1,D.Ov,dtype=np.intc)          # Starting coordinate of the IW
   D.X,D.Y=np.meshgrid(D.xStart+D.Wa/2,D.yStart+D.Wa/2)  # 2d grid
   D.H,D.W=D.X.shape                # grid size
   D.U=0*D.X    # allocating memory
   D.V=0*D.X
   D.Fc=0*D.X
   D.Info=0*D.X
   return D
def blackman(Wa):
  wh = np.arange(1,Wa+1,dtype=FormatDouble)/(Wa+1)
  return 0.42 - 0.5*np.cos(2*np.pi*wh) + 0.08*np.cos(4*np.pi*wh)
def InitNormCrossCorrWin(Wa):
  # Evaluate the normalized Cross Correlation
  Win=blackman(Wa)[np.newaxis]
  Win=Win.T@Win
  FW=np.fft.rfftn(Win)
  return (Win , FW)

def NormCrossCorrWin(IWa,IWb,Win,FW):
  # Evaluate the normalized Cross Correlation
  dum=1/IWa.size
  #ma=IWa.flatten().mean()
  ma=IWa.ravel().sum()*dum#faster than mean
  IWa=IWa-ma
  mb=IWb.ravel().sum()*dum#faster than meanb
  IWb=IWb-mb
  IWa=IWa/np.sqrt( (Win*(IWa**2)).ravel().sum())
  Num=np.fft.irfftn(np.fft.rfftn(IWb)*np.conj(np.fft.rfftn(IWa*Win)))
  den=np.sqrt(np.fft.irfftn(np.fft.rfftn(IWb*IWb)*np.conj(FW)))
  return Num/den

def NormCrossCorr(IWa,IWb):
  # Evaluate the normalized Cross Correlation
  dum=1/IWa.size
  ma=IWa.ravel().sum()*dum#faster than mean
  IWa=IWa-ma
  mb=IWb.ravel().sum()*dum#faster than meanb
  IWb=IWb-mb
  den=np.sqrt(np.square(IWa).ravel().sum()*np.square(IWb).ravel().sum())
  # cross-correlation theorem
  return  np.fft.irfftn(np.fft.rfftn(IWb)*np.conj(np.fft.rfftn(IWa)))/den


def GaussInt(a):
  # Three point Gaussian interpolation 
  # If at least one of the values is negative perform a standard parabolic interpolation 
  Flag=False # simple parabolic interpolation
  if np.all(a>0):  #perform Gaussian interpolation
    a=np.log(a)
    Flag=True
  

  d2=2*a[1] -a[0]-a[2]
  if d2 <=0: # this is not a maximum
    return (0, np.exp(a[1])) if Flag else  (0, a[1])
  d=a[2]-a[0]
  dx=(a[2]-a[0])/(2*d2)
  if np.abs(dx)>0.5: # this is not a maximum
    return (0, np.exp(a[1])) if Flag else  (0, a[1])

  Val=a[1]+dx*d*0.25
  if Flag: #Gaussian interpolation
    Val=np.exp(Val)
  return dx, Val

def  DispFromCC(R):
  # Finds the CC map maximum and then the sub pixel displacement
  Wa=R.shape[0]
  Limit=np.floor(Wa/4).astype(int)           # One quarter rule plus one 
  #  Limiting to a smaller (known) value can increase the robustness
  #RS=R[Limit:-Limit#Wa+1,Limit:-Limit%Wa+1]                 # Smaller CC Map
  ii=np.arange(-Limit,Limit+1)   # Index of the points to be extracted
  #RS=R[-Limit+1:+Limit+1,-Limit:+Limit+1]                 # Smaller CC Map
  RS=R[np.ix_(ii,ii)]
  #I=RS.argmax()           # Search for maximum
  ijm = np.unravel_index(RS.argmax(), RS.shape)# Search for maximum position
  Fc=RS[ijm]
  im=ii[ijm[0]]                  # Position of the maximum in the full CC map
  jm=ii[ijm[1]]

  dy, Valy=GaussInt(    R[np.arange(im-1,im+2)%Wa,jm])
  v=im+dy
  dx, Valx=GaussInt(    R[im,np.arange(jm-1,jm+2)%Wa])
  u=jm+dx
  #if(Valx*Valy>100):    print(Valx*Valy)
  Fc=np.sqrt(Valx*Valy)
  return (u, v,Fc)


def DispFromCorr(S,D,it):
  # DispFromCorr Evaluate the displacement field by using a Cross Correlation approach
  # Used in the predictor and corrector step
  if S.FlagWinA==2: # weighting windows call NormCrossCorrWin
    (Win , FW)=InitNormCrossCorrWin(D.Wa)
    def FunCC(IWa,IWb): 
      return NormCrossCorrWin(IWa,IWb,Win,FW)
  else: # standard normalized cross correlation
    FunCC=NormCrossCorr
    
  for i in range(0,D.yStart.size):
    ie=D.yStart[i]+D.Wa                         # final index for Window extraction
    for j in range(0,D.xStart.size):
      je=D.xStart[j]+D.Wa                       # final index for Window extraction
      if it==0:  # First iteration -> extract the IWs from raw images
        R=FunCC(S.ImgA[D.yStart[i]:ie,D.xStart[j]:je],S.ImgB[D.yStart[i]:ie,D.xStart[j]:je]) # Normalized CC
      else: # following iterations -> extract the IWs from deformed images
        R=FunCC(D.ImgA[D.yStart[i]:ie,D.xStart[j]:je],D.ImgB[D.yStart[i]:ie,D.xStart[j]:je]) # Normalized CC
      (D.U[i,j],D.V[i,j],D.Fc[i,j])=DispFromCC(R) # Displacement vector
  return D
# Replace the outliers
def substituteOutliers(IndGood,X,Y,U):
  F=LinearNDInterpolator(list(zip(X[IndGood], Y[IndGood])),U[IndGood])# Interpolating the "good" vectors
  U[np.logical_not(IndGood)] = F(X[np.logical_not(IndGood)],Y[np.logical_not(IndGood)])
  IndFinite=np.isfinite(U) # in some case return nan switch to nearest neighborhood 
  if not np.all(IndFinite):
    F=NearestNDInterpolator(list(zip(X[IndFinite], Y[IndFinite])),U[IndFinite])
    U[np.logical_not(IndFinite)] = F(X[np.logical_not(IndFinite)],Y[np.logical_not(IndFinite)])

def  medianWS(U,V,Info,ker,UnivEps,UnivTh):
  # Universal outlier detector Westerweel and Scarano (2005)
  # find outliers and substitute them with the median value
  H,W=U.shape
  UnivTh2=UnivTh*UnivTh
  for i in range(0,H): 
    iMin,iMax=max(0,i-ker),min(H-1,i+ker) #Index for extraction of neighboring points
    #ii=max([1,i-ker]):min([H,i+ker]) #Index for extraction of neighboring points
    for j in  range(0,W): 
      jMin,jMax=max(0,j-ker),min(W-1,j+ker) #Index for extraction of neighboring points
      #jj=max([1,j-ker]):min([W,j+ker])# Index for extraction of neighboring points
      nEl=(jMax-jMin+1)*(iMax-iMin+1)-1

      Unb=np.empty(nEl)
      Vnb=np.empty(nEl)
      # Extraction 
      c=0
      for i1 in range(iMin,iMax+1):
        for j1 in range(jMin,jMax+1):
          if(i1==i and j1==j):
            continue # jump the central point
          Unb[c]=U[i1,j1]
          Vnb[c]=V[i1,j1]
          c=c+1
      mu=np.median(Unb)
      rm=np.median(np.fabs(Unb-mu))
      erru=abs(U[i,j]-mu)/(rm+UnivEps)   

      mv=np.median(Vnb)
      rm=np.median(np.fabs(Vnb-mv))
      errv=abs(V[i,j]-mv)/(rm+UnivEps)
      if (erru**2+errv**2)>UnivTh2:                    # WS Check                      
        Info[i,j]=0                  # wrong vector     
        U[i,j]=mu                     # Substitution
        V[i,j]=mv                     # Substitution  
  return U,V,Info
def Validation(S,D):
   # Perform the validation step
   # First limit the displacement accordingly with S.LimV
   # The substitution of the outliers is made by using the 
   #     matlab function scatteredInterpolant 
   # medianSW perform the outlier detection and substitution
   #D.U[-1,-1]=15
   #D.U[2,1]=-15
   IndGood=np.logical_and.reduce((D.U<S.LimV[1], D.U>S.LimV[0] , D.V< S.LimV[3] , D.V>S.LimV[2]))
   D.Info=IndGood   # IndGood is false for outliers
   if not np.all(IndGood):
    substituteOutliers(IndGood,D.X,D.Y,D.U) # Replacing the outliers
    substituteOutliers(IndGood,D.X,D.Y,D.V) 
   

   ker=1         # half dimension of the kernel for now equal to 1
   D.U,D.V,D.Info=medianWS (D.U,D.V,D.Info,ker,S.UnivEps,S.UnivTh)
   return D
def WriteToVideo(S,D,it):
  if S.FlagOutPut%2==1:
    it=it-S.Ov.size+1
    NumVect=D.X.size
    NWrong=100*(NumVect-D.Info.ravel().sum())/NumVect
    FcMed=D.Fc.ravel().mean()
    print(f'It={it:-3d}  Wa={D.Wa:3d} N={D.X.shape[0]:>4d}X{D.X.shape[1]:<4d} Outliers={NWrong:4.1f}%  Fc={FcMed:8.4f}')
  if S.FlagOutPut//2==1:#Plot to be seen
    #figure(1),    clf,    imagesc(D.V),    colorbar
    #figure(2),clf, plot(mean(D.V))
    #figure(3),clf,imagesc(D.U),colorbar
    #figure(4),clf ,plot(mean(D.U))
    pass
def DensePredictorandDeform(D,S):
  # Build the dense predictor and deform the images
  #def Ave(x):     return x.flatten().sum()/x.size
  #MeshGridMat=@(A)  meshgrid(1:size(A,2),1:size(A,1))
  #[Xd,Yd]=np.mgrid[0:S.ImgA.shape[0],0:S.ImgA.shape[1]]

  Off=0.5
  Xd=np.arange(Off,S.ImgA.shape[0]+Off)
  Yd=np.arange(Off,S.ImgA.shape[1]+Off)
  F= RectBivariateSpline( D.Y[:,1], D.X[0],D.U,kx=S.VelInterp,ky=S.VelInterp)
  
  D.Ud=F(Xd,Yd,grid=True)
  F= RectBivariateSpline( D.Y[:,1], D.X[0],D.V,kx=S.VelInterp,ky=S.VelInterp)
  D.Vd=F(Xd,Yd,grid=True)
  
  Ym,Xm=np.mgrid[Off:S.ImgA.shape[0]+Off,Off:S.ImgA.shape[1]+Off]
  
  F= RectBivariateSpline(Xd, Yd, S.ImgA,kx=S.ImgInterp,ky=S.ImgInterp)
  D.ImgA=F((Ym-D.Vd/2).ravel(),(Xm-D.Ud/2).ravel(),grid=False)
  D.ImgA=D.ImgA.reshape(S.ImgA.shape)
  F= RectBivariateSpline(Xd, Yd, S.ImgB,kx=S.ImgInterp,ky=S.ImgInterp)
  D.ImgB=F((Ym+D.Vd/2).ravel(),(Xm+D.Ud/2).ravel(),grid=False)
  D.ImgB=D.ImgB.reshape(S.ImgB.shape)
  
  return D

def  WeightedAverage(D,FlagWinB):
  # Perform a weighted average of the predictor and sum it to the corrector
  # Only top hat and blackman WW are considered
  if FlagWinB==2: # 
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

def PIV(S):
  # A simple implementaion of IDM for PIV
  # Astarita 2021, in vki Lecture series
  # Fundamentals and recent advances in Particle Image Velocimetry and Lagrangian Particle Tracking
  starttime = tt.default_timer()
  it=0
  D=PIVData()
  D=CreateGridandAlloc(S,D,it)
  D=DispFromCorr(S,D,it)
  D=Validation(S,D)
  WriteToVideo(S,D,it)
  for it in range (1,S.Nit+S.Ov.size):
    D=DensePredictorandDeform(D,S)
    D=CreateGridandAlloc(S,D,it)
    D=DispFromCorr(S,D,it)
    D=WeightedAverage(D,S.FlagWinB)
    D=Validation(S,D)
    WriteToVideo(S,D,it)  
  print("PIV total time is:", tt.default_timer() - starttime)
  return D
    

    
def ReadImage_PIV(S,i):
  NomeImg=os.path.join(  f"{S.ImgName}{i:04d}a{S.Ext}")
  Ia=np.asarray(Image.open (NomeImg),dtype=FormatDouble)[S.ROI[0]:S.ROI[1],S.ROI[2]:S.ROI[3]]
  
  NomeImg=os.path.join(  f"{S.ImgName}{i:04d}b{S.Ext}")
  Ib=np.asarray(Image.open (NomeImg),dtype=FormatDouble)[S.ROI[0]:S.ROI[1],S.ROI[2]:S.ROI[3]]
  return [Ia,Ib]



def WritePlt(NomeFileOut,Mat,Titolo,NomeVar,TitoloZona):
  '''#write tecplot binary files'''
  NVar=len(Mat)
  H,W=Mat[0].shape
  TecIntest=b'#!TDV71 '
  
  writeLong = lambda l : f.write(np.int32(l))
  writeFloat32 = lambda fl : f.write(np.float32(fl))
  def writeString(s): 
    for c in s:f.write(np.int32(c))
    return
  #apertura file binario di output .plt
  f=open(NomeFileOut,"wb")
  
  #scrittura nel file di output del TEST 
  f.write(TecIntest)
    #scrittura nel file di output di 1 (ordine dei bytes BO???)
  #Lungo=np.int32(1)
  #f.write(Lungo)
  writeLong(1)
  #scrittura del titolo
  writeString(Titolo)
  
  #scrittura numero e nome delle variabili
  writeLong(NVar)
  writeString(NomeVar)
  writeFloat32(299.0)
  #ZONE NAME
  writeString(TitoloZona)
  #scrittura del BLOCK POINT
  writeLong(1)
  #scrittura del COLORE
  writeLong(-1)
  #scrittura nel file di output della larghezza e altezza img W e H
  writeLong(W)
  writeLong(H)
  #scrittura nel file di output della dimensione Z
  writeLong(1)
  #scrittura nel file di output di 357.0 (bo????)
  writeFloat32(357.0)
  #scrittura nel file di output di 299.0 (bo????)
  writeFloat32(299.0)
  #scrittura nel file di output di 0 (bo????)
  writeLong(0)
  #sizeof variabili
  for i in range (0,NVar):     writeLong(1)

  #scrittura nel file di output delle matrici x,y,u,v,up,vp (variabili)
  writeFloat32(np.ascontiguousarray(np.transpose(Mat,(1,2,0))))
      
  f.flush()
  f.close()
   
  return   
def Output(S,D,i):
  NomeFileOut=os.path.join(  f"{S.OutName}{i:0{S.Ndig:d}d}.plt")
  TitoloZona=bytes(NomeFileOut+'\0','utf-8')
  Titolo=b'b16\0'
  NomeVar=b'X\0Y\0U\0V\0Fc\0Info\0'
  Varia=[ D.X,    D.Y,    D.U  ,  D.V   , D.Fc, D.Info, ]
  WritePlt(NomeFileOut,Varia,Titolo,NomeVar,TitoloZona)


if __name__ == "__main__":
  S=PIVInput()
  for i in S.NImg:
    S.setImages(ReadImage_PIV(S,i)  )
    #if flagProf:      cProfile.run('D=PIV(S)')    else:
    D=PIV(S)
    Output(S,D,i)