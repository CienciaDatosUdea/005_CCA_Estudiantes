
#Version 9   07-09-2024
#Añadido Videoshow e Imshow para mostrar imagenes y videos en pantalla completa.
#Version 8   28-08-2024
#Añadido ASM Lente para hacer TF optica. Eliminado la funcion perceptual loss incompleta.
#Version 7   27-08-2008
#Modificado progmargcalc y asm para eliminar la necesidad de margx y margy y calcularlo a partir de la diferencia entre
#el tamaño de la funcion de transferencia y la entrada.
#Version 6  27-08-2024
#-Añadido ASMnp y ASMpropnp para probar ASM sin padding.
#Version 5  22-08-2024
#-Implementado progmargcalc y ASM para calcular propagacion con tamaños fisicos reales
#Version 4   28-05-2024
#-Añadido tfouriern y itfouriern para eliminar un shift.
#Version 3   19-05-2024
#-Eliminado print innecesario de fusion
#Version 2   21-04-2024
#-Añadido fusionm y tajadam basados en coordenadas matriciales.
#Version 1  11-12-2024
#-Cambiado propft a usar solo torch.
import torch
import cv2 as cv2
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models

def Videoshow(file,wx,wy,mode=0):
    """
    Esta funcion permite mostrar un video en pantalla completa
    file: string con direccion del video a mostrar
    wx:numero entero con posicion x de la ventana que va a mostrar el video. 
    wy:numero entero con posicion yde la ventana que va a mostrar el video. 
    mode: argumento de seleccion. Si es 0 mostrar el video en pantalla completa, ajustando a sus dimensiones con negro, de lo contrario hara una ventana.
    Si se usa mode=0 para pantalla completa, se seleccionara la pantalla correspondiente a la posicion dada por wx y wy.
    """
    cap = cv2.VideoCapture('recording_2024-09-02-multi.avi')
    if (cap.isOpened()== False):
        print("Error opening video file")
    cv2.namedWindow("WindowName",cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("WindowName", wx, wy) 
    cv2.waitKey(1)
    if mode==0:
        cv2.setWindowProperty("WindowName", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while(cap.isOpened()):   
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow("WindowName", frame)    
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            else:
                break
    cap.release()
    cv2.destroyAllWindows()

def Imshow(img,wx,wy,mode=0):
    """
    Esta funcion permite mostrar una imagen en pantalla completa
    img: imagen a mostrar. Debe ser un tensor de pytorch, en la cpu, sin seguimiento de gradiente (detach), con dimensiones ordenadas en forma de imagen [H,W,C].
    wx:numero entero con posicion x de la ventana que va a mostrar la imagen. 
    wy:numero entero con posicion yde la ventana que va a mostrar la imagen. 
    mode: argumento de seleccion. Si es 0 mostrar la imagen en pantalla completa, ajustando a sus dimensiones con negro, de lo contrario hara una ventana.
    Si se usa mode=0 para pantalla completa, se seleccionara la pantalla correspondiente a la posicion dada por wx y wy.
    """
    o=img
    o=o.numpy()
    cv2.namedWindow("WindowName",cv2.WINDOW_AUTOSIZE)
    cv2.imshow("WindowName",o)
    cv2.moveWindow("WindowName", wx, wy) 
    cv2.waitKey(1)
    if mode==0:
        cv2.setWindowProperty("WindowName", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(0)
 

def propft(args):
    z=args.z
    wl=args.wl
    dx=args.dx
    tx=args.sx
    ty=args.sy
    Lx=tx*dx
    Ly=ty*dx
    fx=torch.arange(-1/(2*dx),1/(2*dx),1/Lx)
    fy=torch.arange(-1/(2*dx),1/(2*dx),1/Ly)
    FX,FY=torch.meshgrid(fx,fy,indexing='xy')
    H=torch.exp(-1j*wl*z*np.pi*(FX**2+FY**2))
    H=torch.fft.fftshift(H)
    return H

def cil(args,dir):
    z=args.z
    wl=args.wl
    dx=args.dx
    tx=args.sx
    ty=args.sy
    Lx=tx*dx
    Ly=ty*dx
    fx=torch.arange(-1/(2*dx),1/(2*dx),1/Lx)
    fy=torch.arange(-1/(2*dx),1/(2*dx),1/Ly)
    FX,FY=torch.meshgrid(fx,fy,indexing='xy')
    if dir=='x':
        H=torch.exp(-1j*wl*z*np.pi*(FX**2))
    if dir=='y':
        H=torch.exp(-1j*wl*z*np.pi*(FY**2))
    H=torch.fft.fftshift(H)
    return H

def convprop(u1,H):
    U1=torch.fft.fft2(torch.fft.fftshift(u1))
    U2=H*U1
    u2=torch.fft.ifftshift(torch.fft.ifft2(U2))
    return u2


def tajadan(M1,M2,x,y):
    """
    Esta funcion permite recortar una seccion centrada en x, y del array M1 de
    tamaño igual al array M2. Las coordenadas x,y son dadas en un sistema matricial.

    Parameters
    ----------
    M1 : Array Numpy
        Arreglo del que se recorta una sección.
    M2 : Array Numpy
        Arreglo que determina el tamaño del recorte.
    x : Integer
        Coordenada en x del centro del recorte en un sistema matricial para M1.
    y : Integer
        Coordenada en y del centro del recorte en un sistema matricial para M1..

    Raises
    ------
    ValueError
        Si el tamaño del area a recortar o las coordenadas exceden el tamaño de la matriz original se disparara este error.

    Returns
    -------
    MC
        Arreglo de numpy con forma igual a M2.

    """
    m=M1.shape
    mm=M2.shape
    if mm[-1]>m[-1] or mm[-2]>m[-2]:
        raise ValueError('La matriz recortada debe ser mas pequeña que la original')
    MC=M1[...,y:y+mm[-2],x:x+mm[-1]]
    return MC

def fusionn(M1,M2,x,y):
    """
    Esta funcion permite insertar M2 a una seccion centrada en (x, y) del array M1.
    Las coordenadas x,y son dadas en un sistema matricial.

    Parameters
    ----------
    M1 : Array Numpy
        Arreglo al que se inserta una sección.
    M2 : Array Numpy
        Arreglo a insertar.
    x : Integer
        Coordenada en x del centro del recorte en un sistema matricial para M1.
    y : Integer
        Coordenada en y del centro del recorte en un sistema matricial para M1.

    Raises
    ------
    ValueError
        Si el tamaño del area a insertar o las coordenadas exceden el tamaño de la matriz original se disparara este error.

    Returns
    -------
    MC
        Arreglo de numpy con forma igual a M1.

    """
    m=M1.shape
    mm=M2.shape
    if mm[-2]>m[-2] or mm[-1]>m[-1]:
        raise ValueError('La matriz a instertar debe ser mas pequeña que la original')
    M1[...,y:y+mm[-2],x:x+mm[-1]]=M2

    MC=M1
    return MC


def tajada(M1,M2,x,y):
    """
    Esta funcion permite recortar una seccion centrada en x, y del array M1 de
    tamaño igual al array M2. Las coordenadas x,y son dadas en un sistema cartesiano centrado.

    Parameters
    ----------
    M1 : Array Numpy
        Arreglo del que se recorta una sección.
    M2 : Array Numpy
        Arreglo que determina el tamaño del recorte.
    x : Integer
        Coordenada en x del centro del recorte en un sistema cartesiano centrado para M1.
    y : Integer
        Coordenada en y del centro del recorte en un sistema cartesiano centrado para M1..

    Raises
    ------
    ValueError
        Si el tamaño del area a recortar o las coordenadas exceden el tamaño de la matriz original se disparara este error.

    Returns
    -------
    MC
        Arreglo de numpy con forma igual a M2.

    """
    m=M1.shape
    mm=M2.shape
    if mm[-1]>m[-1] or mm[-2]>m[-2]:
        raise ValueError('La matriz recortada debe ser mas pequeña que la original')
    gym=((m[-2]-mm[-2])//2)

    gyp=((m[-2]+mm[-2])//2)

    gxm=((m[-1]-mm[-1])//2)

    gxp=((m[-1]+mm[-1])//2)

    x=np.int32(x)

    y=np.int32(y)

    MC=M1[...,gym+y:gyp+y,gxm+x:gxp+x]
    return MC

def fusion(M1,M2,x,y):
    """
    Esta funcion permite insertar M2 a una seccion centrada en (x, y) del array M1.
    Las coordenadas x,y son dadas en un sistema cartesiano centrado.

    Parameters
    ----------
    M1 : Array Numpy
        Arreglo al que se inserta una sección.
    M2 : Array Numpy
        Arreglo a insertar.
    x : Integer
        Coordenada en x del centro del recorte en un sistema cartesiano centrado para M1.
    y : Integer
        Coordenada en y del centro del recorte en un sistema cartesiano centrado para M1..

    Raises
    ------
    ValueError
        Si el tamaño del area a insertar o las coordenadas exceden el tamaño de la matriz original se disparara este error.

    Returns
    -------
    MC
        Arreglo de numpy con forma igual a M1.

    """
    m=M1.shape
    mm=M2.shape
    if mm[-2]>m[-2] or mm[-1]>m[-1]:
        raise ValueError('La matriz a instertar debe ser mas pequeña que la original')
    gym=((m[-2]-mm[-2])//2)

    gyp=((m[-2]+mm[-2])//2)

    gxm=((m[-1]-mm[-1])//2)

    gxp=((m[-1]+mm[-1])//2)

    x=np.int32(x)
    y=np.int32(y)

    M1[...,gym+y:gyp+y,gxm+x:gxp+x]=M2

    MC=M1
    return MC

def itfourier(x):
    x=torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))
    return x
def tfourier(x):
    x=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))
    return x

def itfouriern(x):
    x=torch.fft.fftshift(torch.fft.ifft2(x))
    return x
def tfouriern(x):
    x=torch.fft.fftshift(torch.fft.fft2(x))
    return x

def ASMprop(args):
    wl=args.wl
    dx=args.dx
    dy=args.dy
    sx=args.sx
    sy=args.sy
    z=args.z
    angx=torch.asin(torch.tensor(wl/(2*dx)))
    angy=torch.asin(torch.tensor(wl/(2*dy)))
    margx=torch.abs(torch.tensor(z))*torch.tan(angx)
    margx=torch.ceil(margx/dx)
    margy=torch.abs(torch.tensor(z))*torch.tan(angy)
    margy=torch.ceil(margy/dy)
    margx=margx.int().item()
    margy=margy.int().item()
    if margx> 1000 or margy> 1000:
        print('Precaucion: Los parametros de propagacíon implican un padding de mas de ',margx, ' píxeles')
    fieldpad=torch.zeros((1,1,sy,sx))
    pad=nn.ConstantPad2d((margy,margy,margx,margx),0)
    fieldpad=pad(fieldpad)
    spy=fieldpad.shape[2]
    spx=fieldpad.shape[3]
    Tfx=1/dx
    dfx=Tfx/spx
    Tfy=1/dy
    dfy=Tfy/spy
    if np.remainder(spx,2)==0:
        fx=torch.arange(-Tfx/2,Tfx/2,dfx)
    else:
        fx=torch.arange(-(Tfx/2-dfx)/2,(Tfx-dfx)/2,dfx)
    if np.remainder(spy,2)==0:
        fy=torch.arange(-Tfy/2,Tfy/2,dfy)
    else:
        fy=torch.arange(-(Tfy/2-dfy)/2,(Tfy-dfy)/2,dfy)
    
    Fx,Fy=torch.meshgrid(fx,fy,indexing='xy')
    Gamma=(1/(wl**2))-Fx**2-Fy**2
    Prop=torch.exp(2*1j*torch.pi*torch.sqrt(Gamma)*z)
    return Prop



def ASMpropnp(args):
    wl=args.wl
    dx=args.dx
    dy=args.dy
    sx=args.sx
    sy=args.sy
    z=args.z
    Tfx=1/dx
    dfx=Tfx/sx
    Tfy=1/dy
    dfy=Tfy/sy
    if np.remainder(sx,2)==0:
        fx=torch.arange(-Tfx/2,Tfx/2,dfx)
    else:
        fx=torch.arange(-(Tfx/2-dfx)/2,(Tfx-dfx)/2,dfx)
    if np.remainder(sy,2)==0:
        fy=torch.arange(-Tfy/2,Tfy/2,dfy)
    else:
        fy=torch.arange(-(Tfy/2-dfy)/2,(Tfy-dfy)/2,dfy)
    
    Fx,Fy=torch.meshgrid(fx,fy,indexing='xy')
    Gamma=(1/(wl**2))-Fx**2-Fy**2
    Prop=torch.exp(2*1j*torch.pi*torch.sqrt(Gamma)*z)
    return Prop

def ASMnp(A,Prop):
    A=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(A)))
    A=torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(A*Prop)))
    R=A
    return R

def ASMtf(A,args):
    f=args.f
    wl=args.wl
    dx=args.dx
    dy=args.dy
    sx=A.shape[-1]
    sy=A.shape[-2]
    Lx=dx*sx
    Ly=dx*sy
    args.z=f
    X=torch.arange(-Lx/2,Lx/2,dx)
    Y=torch.arange(-Ly/2,Ly/2,dx)
    X,Y=torch.meshgrid(X,Y,indexing='xy')
    F=torch.exp(-1j*(torch.pi/(wl*f))*(X**2+Y**2)).to(args.device).unsqueeze(0).unsqueeze(0)
    Pr=ASMprop(args).to(args.device)
    A=ASM(A,Pr)
    A=A*F
    A=ASM(A,Pr)
    return A

def ASMitf(A,args):
    f=args.f
    wl=args.wl
    dx=args.dx
    dy=args.dy
    sx=A.shape[-1]
    sy=A.shape[-2]
    Lx=dx*sx
    Ly=dx*sy
    args.z=f
    X=torch.arange(-Lx/2,Lx/2,dx)
    Y=torch.arange(-Ly/2,Ly/2,dx)
    X,Y=torch.meshgrid(X,Y,indexing='xy')
    F=torch.exp(-1j*(torch.pi/(wl*f))*(X**2+Y**2)).to(args.device).unsqueeze(0).unsqueeze(0)
    Pr=ASMprop(args).to(args.device)
    A=ASM(A,torch.conj(Pr))
    A=A*torch.conj(F)
    A=ASM(A,torch.conj(Pr))
    return A
    
    
def ASM(A,Prop):
    n = A.shape[-2]

    m = A.shape[-1]
    ns= Prop.shape[-2]
    mm= Prop.shape[-1]

    if (mm-m)%2 ==0:
        margx1=(mm-m)/2
        margx1=np.floor(margx1)
        margx2=(mm-m)/2
        margx2=np.floor(margx2)
    else:
        margx1=(mm-m)/2
        margx1=np.floor(margx1)
        margx2=(mm-m)/2+1
        margx2=np.floor(margx2)
    if (ns-n)%2 ==0:
        margy1=(ns-n)/2
        margy1=np.floor(margy1)
        margy2=(ns-n)/2
        margy2=np.floor(margy2)
    else:
        margy1=(ns-n)/2
        margy1=np.floor(margy1)
        margy2=(ns-n)/2+1
        margy2=np.floor(margy2)
    margx1=int(margx1)
    margx2=int(margx2)
    margy1=int(margy1)
    margy2=int(margy2)

        
    pad=nn.ConstantPad2d((margx1,margx2,margy1,margy2),0)
    A=pad(A)

    A=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(A)))
    A=torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(A*Prop)))
    R=A[...,int(margy1):A.shape[-2]-int(margy2),int(margx1):A.shape[-1]-int(margx2)]
    return R

def Ofrapgen(args,M):
    g=torch.zeros((args.sy,args.sx),device=args.device)
    S=g
    S=fusion(S,torch.ones((args.sy,args.sx),device=args.device),0,0)
    g1=fusion(g,torch.ones((args.tx,args.tx),device=args.device),0,0)
    g=g1*M
    prop=propft(args)
    prop=prop.to(args.device)
    for i in range(0,args.iters):
        G=convprop(g,torch.conj(prop))
        G=S*torch.exp(1j*torch.angle(G))
        g=convprop(G,prop)
        g=g1*torch.exp(1j*torch.angle(g))
    return g

def Ifrta(H,args):
    g=torch.zeros((args.sx,args.sx),device=args.device)
    S=g
    S=fusion(S,torch.ones((args.sy,args.sx),device=args.device),0,0)
    g1=fusion(g,H,0,0)
    g=g1*torch.exp(1j*2*torch.pi*torch.rand((args.sx,args.sx),device=args.device))
    #g=g1
    prop=propft(args)
    prop=prop.to(args.device)
    for i in range(0,args.iters):
        G=convprop(g,torch.conj(prop))
        G=S*torch.exp(1j*torch.angle(G))
        g=convprop(G,prop)
        g=g1*torch.exp(1j*torch.angle(g))
    return G

def corr2(A,B):
    cc=(((A-torch.mean(A))*((B-torch.mean(B)))).sum())/(torch.sqrt((A-torch.mean(A)).pow(2).sum()*(B-torch.mean(B)).pow(2).sum()))
    return cc

def abwavephase(zcoef,ix,iy,pup):
    
    numz=len(zcoef)
    nextz=1
    n_table=[0]
    m_table=[0]
    while nextz<=numz-1:
        if m_table[nextz-1]==n_table[nextz-1]:
            n_table.append(n_table[nextz-1]+1)
            m_table.append(-n_table[nextz])
        else:
            n_table.append(n_table[nextz-1])
            m_table.append(m_table[nextz-1]+2)
        nextz=nextz+1
    nmax=max(n_table)

    fac= np.zeros((nmax+1))
    fac[0]=1
    for ni in range(1,nmax+1):
        print(ni)
        fac[ni]=fac[ni-1]*(ni)

    ycenter=(iy+1)/2
    xcenter=(ix+1)/2
    x=x-xcenter
    y=ycenter-y
    rho=np.sqrt(x**2+y**2)/np.abs(pup)
    theta=np.arctan2(y,x)
    map=np.zeros(iy,ix)

    #for z in range(1,numz):
     #   zernike=np.zeros((ix))
      #  n=n_table[z]
       # m=m_table[z]
        #m_sign=np.sign(m_table(z))
        #for s in range(0,(n-m)/2):
    return map


def comb2d(x,sx,sy):
    [qy,qx]=x.shape
    R=torch.zeros(x.shape)
    for i in range(qy):
        for j in range(qx):
            if np.mod(i,sy+1)==0 and np.mod(j,sx+1)==0:
                R[i,j]=1
    return R

