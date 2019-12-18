##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to analyse stratigraphic sequences from Badlands outputs.
"""

import os
import math
import h5py
import errno
import numpy as np
import pandas as pd
from cmocean import cm
import matplotlib as mpl
from matplotlib import mlab, cm
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
import colorlover as cl
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ETO
import scipy.ndimage.filters as filters
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
import matplotlib.colors
from pylab import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.spatial import cKDTree

import plotly
from plotly import tools
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

def readSea(seafile):
    """
    Plot sea level curve.
    Parameters
    ----------
    variable: seafile
        Absolute path of the sea-lelve data.
    """

    df=pd.read_csv(seafile, sep=r'\s+',header=None)
    SLtime,sealevel = df[0],df[1]

    return SLtime,sealevel

def viewData(x0 = None, y0 = None, width = 800, height = 400, linesize = 3, color = '#6666FF',
             xlegend = 'xaxis', ylegend = 'yaxis', title = 'view data'):
    """
    Plot multiple data on a graph.
    Parameters
    ----------
    variable: x0, y0
        Data for plot
    variable: width, height
        Figure width and height.
    variable: linesize
        Requested size for the line.
    variable: color
        
    variable: xlegend
        Legend of the x axis.
    variable: ylegend
        Legend of the y axis.
    variable: title
        Title of the graph.
    """
    trace = Scatter(
        x=x0,
        y=y0,
        mode='lines',
        line=dict(
            shape='line',
            color = color,
            width = linesize
        ),
        fill=None
    )

    layout = dict(
            title=title,
            font=dict(size=10),
            width=width,
            height=height,
            showlegend = False,
            xaxis=dict(title=xlegend,
                       ticks='outside',
                       zeroline=False,
                       showline=True,
                       mirror='ticks'),
            yaxis=dict(title=ylegend,
                       ticks='outside',
                       zeroline=False,
                       showline=True,
                       mirror='ticks')
            )

    fig = Figure(data=[trace], layout=layout)
    plotly.offline.iplot(fig)

    return

def viewSection(width = 800, height = 400, cs = None, dnlay = None,
                rangeX = None, rangeY = None, linesize = 3, title = 'Cross section'):
    """
    Plot multiple cross-sections data on a graph.
    Parameters
    ----------
    variable: width
        Figure width.
    variable: height
        Figure height.
    variable: cs
        Cross-sections dataset.
    variable: dnlay
        Layer step to plot the cross-section.
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: linesize
        Requested size for the line.
    variable: title
        Title of the graph.
    """
    nlay = len(cs.secDep)
    colors = cl.scales['9']['div']['BrBG']
    hist = cl.interp( colors, nlay )
    colorrgb = cl.to_rgb( hist )

    trace = {}
    data = []

    trace[0] = Scatter(
        x=cs.dist,
        y=cs.secDep[0],
        mode='lines',
        line=dict(
            shape='line',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[0])

    for i in range(1,nlay-1,dnlay):
        trace[i] = Scatter(
            x=cs.dist,
            y=cs.secDep[i],
            mode='lines',
            line=dict(
                shape='line',
                width = linesize,
                color = 'rgb(0,0,0)'
            ),
            opacity=0.5,
            fill='tonexty',
            fillcolor=colorrgb[i]
        )
        data.append(trace[i])

    trace[nlay-1] = Scatter(
        x=cs.dist,
        y=cs.secDep[nlay-1],
        mode='lines',
        line=dict(
            shape='line',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        ),
        fill='tonexty',
        fillcolor=colorrgb[nlay-1]
    )
    data.append(trace[nlay-1])

    trace[nlay] = Scatter(
        x=cs.dist,
        y=cs.secDep[0],
        mode='lines',
        line=dict(
            shape='line',
            width = linesize+2,
            color = 'rgb(0, 0, 0)'
        )
    )
    data.append(trace[nlay])

    if rangeX is not None and rangeY is not None:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height,
                showlegend = False,
                xaxis=dict(title='distance [m]',
                            range=rangeX,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks'),
                yaxis=dict(title='elevation [m]',
                            range=rangeY,
                            ticks='outside',
                            zeroline=False,
                            showline=True,
                            mirror='ticks')
        )
    else:
        layout = dict(
                title=title,
                font=dict(size=10),
                width=width,
                height=height
        )
    fig = Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

    return

def buildEnviID(cs = None, depthID = None):
    """
    Plot stratal stacking pattern colored by paleo-depositional environments.
    Parameters
    ----------
    variable: cs
        Cross-sections dataset.
    variable: depthID
        Ranges of water depth for depositional environments.
    """
    
    enviID = np.zeros((cs.nz, len(cs.dist)))
    
    for j in range(cs.nz):
        for i in range(len(cs.dist)):
            if (cs.secElev[j][i]) > (depthID[0]):
                enviID[j][i] = 0
            elif (cs.secElev[j][i]) > (depthID[1]):
                enviID[j][i] = 1
            elif (cs.secElev[j][i]) > (depthID[2]):
                enviID[j][i] = 2
            elif (cs.secElev[j][i]) > (depthID[3]):
                enviID[j][i] = 3
            elif (cs.secElev[j][i]) > (depthID[4]):
                enviID[j][i] = 4
            else:
                enviID[j][i] = 5
    
    for j in range(cs.nz):
        for i in range(len(cs.dist)):
            if (cs.secTh[j][i]) <= 0.01:
                enviID[j][i] = -1
    
    return enviID

def viewDepoenvi(width = 8, height = 5, cs = None, enviID = None, dnlay = None, color = None,
                      rangeX = None, rangeY = None, savefig = 'Yes', figname = 'strata_depoenv'):
    """
    Plot stratal stacking pattern colored by paleo-depositional environments.
    Parameters
    ----------
    variable: cs
        Cross-sections dataset.
    variable: enviID
        An array of depositional environment ID.
    variable: dnlay
        Layer step to plot the cross-section.
    variable: colors
        Colors for different ranges of water depth (i.e. depositional environments).
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: savefig
        'Yes' means to save the figure. 'No' means not to save the figure.
        If 'Yes', this figure will be saved to the current directory.
    variable: figname
        Figure name of the graph.
    """
    
    fig = plt.figure(figsize = (width,height))
    plt.rc("font", size=11)

    ax = fig.add_axes([0.14, 0.18, 0.82, 0.76])
    layID = []
    p = 0
    xi00 = cs.dist

    for i in range(0,cs.nz+1,dnlay):
        if i == cs.nz:
            i = cs.nz-1
        layID.append(i)
        if len(layID) > 1:
            for j in range(0,len(xi00)-1):
                index = int(enviID[layID[p-1]][j])
                plt.fill_between([xi00[j],xi00[j+1]], [cs.secDep[layID[p-1]][j],
                                  cs.secDep[layID[p-1]][j+1]], max(cs.secDep[layID[p-1]]), color=color[index+1])
        if (max(cs.secDep[layID[p]]) <= max(cs.secDep[layID[p-1]])):
            plt.fill_between(xi00, cs.secDep[layID[p]], max(cs.secDep[layID[p-1]]), color='white')
        else:
            plt.fill_between(xi00, cs.secDep[layID[p]], max(cs.secDep[layID[p]]), color='white')
        p=p+1
    for i in range(0,cs.nz,dnlay):
        if i>0:
            plt.plot(xi00,cs.secDep[i],'-',color='k',linewidth=0.2)
    plt.plot(xi00,cs.secDep[cs.nz-1],'-',color='k',linewidth=0.7)
    plt.plot(xi00,cs.secDep[0],'-',color='k',linewidth=0.7)
    plt.xlim( rangeX )
    plt.ylim( rangeY )
    plt.xlabel('Distance (m)')
    plt.ylabel('Depth (m)')
    
    if savefig == 'Yes':
        fig.savefig("%s.pdf"%(figname), dpi=300)  # save this figure in the current folder

    return

def viewWheeler(width = 8, height = 5, cs = None, enviID = None, time = None, dnlay = 5, color = None, 
                rangeX = None, rangeY = None, savefig = 'Yes', figname = 'Wheeler_diagram'):
    """
    Plot Wheeler diagram colored by paleo-depositional environments.
    Parameters
    ----------
    variable: cs
        Cross-sections dataset.
    variable: enviID
        An array of depositional environment ID.
    variable: time
        Time series of the strata.
    variable: dnlay
        Layer step to plot the cross-section.
    variable: colors
        Colors for different ranges of water depth (i.e. depositional environments).
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: savefig
        'Yes' means to save the figure. 'No' means not to save the figure.
        If 'Yes', this figure will be saved to the current directory.
    variable: figname
        Figure name of the graph.
    """
    
    fig = plt.figure(figsize = (width,height))
    plt.rc("font", size=11)
    
    ax = fig.add_axes([0.15, 0.18, 0.74, 0.76])
    # Define the meshgrid of the Wheeler diagram, in which X axis is distance along the cross-section and Y axis is time
    dist = cs.dist  
    Dist, Time = np.meshgrid(dist, time)
    
    cmap = mpl.colors.ListedColormap(color)
    #boundaries = [-1, 0, 1, 2, 3, 4, 5, 6]
    #norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    plt.contourf(Dist, Time, enviID, cmap=cmap, norm=None)
#     plt.colorbar()

    # Plot the horizontal time lines with the same interval with the stratal stacking pattern
    for i in range(0,cs.nz,dnlay): 
        plt.axhline(time[i], color='k', linewidth=0.2)

    plt.xlim( rangeX )
    plt.ylim( rangeY )
    plt.xlabel('Distance (m)')
    plt.ylabel('Time (yr)')
    
    if savefig == 'Yes':
        fig.savefig("%s.pdf"%(figname), dpi=300)  # save this figure in the current folder
    
    return

def viewCore(width = 2, height = 5, cs = None, enviID = None, posit = None, time = None, color = None, 
             rangeX = None, rangeY = None, savefig = 'Yes', figname = 'Core'):
    """
    Plot synthetic cores.
    Parameters
    ----------
    variable: cs
        Cross-sections dataset.
    variable: enviID
        An array of depositional environment ID.
    variable: posit
        Location of the core on the extracted cross-section.
    variable: time
        Time series of the strata.
    variable: colors
        Colors for different ranges of water depth (i.e. depositional environments).
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: savefig
        'Yes' means to save the figure. 'No' means not to save the figure.
        If 'Yes', this figure will be saved to the current directory.
    variable: figname
        Figure name of the graph.
    """
    
    # Index of the core on the cross-section
    positID = np.amax(np.where(cs.dist<posit)).astype(int)
    # EnviID of this core
    enviID_posit = enviID[:, positID][::-1]  # top to bottom
    # Thickness of this core
    thick_posit = np.array(cs.secTh)[:,positID][::-1]
    # Depth of the core
    depth_posit = np.array(cs.secDep)[:,positID][::-1]
    
    # time interval between stratigraphic layers
    dtime = time[1] - time[0]
    
    # Build core structure with three columns: enviID (color), thickness, time
    core_envi = [enviID_posit[0]]
    core_thickness = [thick_posit[0]]
    core_depth = [depth_posit[0]]
    core_time = [0]
    p = 0
    for i in range(1,cs.nz):
        if (enviID_posit[i] != enviID_posit[i-1]):
            core_envi.append(enviID_posit[i])
            core_thickness.append(thick_posit[i])
            core_depth.append(depth_posit[i-1])
            core_time.append(dtime*i)
            p = p+1
        else:
            core_thickness[p] = core_thickness[p] + thick_posit[i]
            
    # Plot the core
    fig = plt.figure(figsize = (width,height))
    plt.rc("font", size=11)
    # 
    ax = fig.add_axes([0.45,0.1,0.25,0.85])
    for i in range(0,len(core_envi)):
        plt.bar(posit, -core_thickness[i], color=color[int(core_envi[i])+1], edgecolor='black', bottom=core_depth[i])
    
    plt.ylim( rangeY )  # If you want to fix the depth range. Would be useful when plotting multiple cores.
    plt.xticks([posit],fontsize=11)
    plt.xlabel('Location (m)')
    plt.ylabel('Depth (m)')

    if savefig == 'Yes':
        fig.savefig("%s.pdf"%(figname), dpi=300)  # save this figure in the current folder
        
    return

def strataAnimate(width = 7, height = 3, cs = None, dnlay = 5, 
               rangeX = None, rangeY = None, folder = None, videoname = None):
    """
    Plot temporal stratal layers and create a video.
    Parameters
    ----------
    variable: cs
        Cross-sections dataset.
    variable: dnlay
        Layer step to plot the cross-section.
    variable: rangeX, rangeY
        Extent of the cross section plot.
    variable: folder
        Folder path to the figure and movie outputs..
    variable: figname
        Figure name of the graph.
    variable: videoname
        Name of the video.
    """
    
    for k in range(len(cs)):
        fig = plt.figure(figsize = (width,height))
        plt.rc("font", size=11)
        #
        for j in range(0,cs[k].nz,dnlay):
            plt.plot(cs[0].dist,cs[k].secDep[j],color='dimgrey',linewidth=0.2)
        
        plt.title('%3.1f Myr'%(k*nstep*(time[1]-time[0])))
        plt.xlim( rangeX )
        plt.ylim( rangeY )
        plt.xlabel('Distance (m)')
        plt.ylabel('Depth (m)')
        fig.savefig("%s/strata%d.png"%(folder,k), dpi=400)
        plt.close(fig)

class stratalSection:
    """
    Class for creating stratigraphic cross-sections from Badlands outputs.
    """

    def __init__(self, folder=None, ncpus=1):
        """
        Initialization function which takes the folder path to Badlands outputs
        and the number of CPUs used to run the simulation.

        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.
        variable: ncpus
            Number of CPUs used to run the simulation.
        """

        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        self.ncpus = ncpus
        if ncpus > 1:
            raise RuntimeError('Multi-processors function not implemented yet!')
        
        self.ncpus = ncpus
        self.bbox = None
        self.x = None
        self.y = None
        self.z = None
        self.xi = None
        self.yi = None
        self.zi = None
        self.dx = None
        self.dy = None
        self.nx = None
        self.ny = None
        self.nz = None
        self.dist = None
        self.dep = None
        self.th = None
        self.elev = None
        self.xsec = None
        self.ysec = None
        self.secTh = []
        self.secDep = []
        self.secElev = []
        self.cumchange = None

        return
    
    def loadStratigraphy(self, timestep=0):
        """
        Read the strata HDF5 file for a given time step.
        Parameters
        ----------
        variable : timestep
            Specific step at which the strata variables will be read.
        """

        for i in range(0, self.ncpus):
            df = h5py.File('%s/sed.time%s.p%s.hdf5'%(self.folder, timestep, i), 'r')
            #print(list(df.keys()))
            coords = np.array((df['/coords']))
            layDepth = np.array((df['/layDepth']))
            layElev = np.array((df['/layElev']))
            layThick = np.array((df['/layThick']))
            if i == 0:
                x, y = np.hsplit(coords, 2)
                dep = layDepth
                elev = layElev
                th = layThick

        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.x = x
        self.y = y
        self.nx = int((x.max() - x.min())/self.dx+1)
        self.ny = int((y.max() - y.min())/self.dx+1)
        self.nz = dep.shape[1]
        self.xi = np.linspace(x.min(), x.max(), self.nx)
        self.yi = np.linspace(y.min(), y.max(), self.ny)
        self.dep = dep.reshape((self.ny,self.nx,self.nz))
        self.elev = elev.reshape((self.ny,self.nx,self.nz))
        self.th = th.reshape((self.ny,self.nx,self.nz))


        return
    
    def loadTIN(self, timestep=0):
        """
        Read the TIN HDF5 file for a given time step.
        Parameters
        ----------
        variable : timestep
            Specific step at which the TIN variables will be read.
        """

        for i in range(0, self.ncpus):
            df = h5py.File('%s/tin.time%s.p%s.hdf5'%(self.folder, timestep, i), 'r')
            coords = np.array((df['/coords']))
            cumdiff = np.array((df['/cumdiff']))
            if i == 0:
                x, y, z = np.hsplit(coords, 3)
                c = cumdiff
            else:
                c = np.append(c, cumdiff)
                x = np.append(x, coords[:,0])
                y = np.append(y, coords[:,1])
                z = np.append(z, coords[:,2])
                
            self.bbox = np.zeros(4,dtype=float)
            self.bbox[0] = x.min()
            self.bbox[1] = y.min()
            self.bbox[2] = x.max()
            self.bbox[3] = y.max()

        self.xii, self.yii = np.meshgrid(self.xi, self.yi)
        xyi = np.dstack([self.xii.flatten(), self.yii.flatten()])[0]
        XY = np.column_stack((x,y))
        tree = cKDTree(XY)
        distances, indices = tree.query(xyi, k=3)
        z_vals = z[indices][:,:,0]
        c_vals = c[indices][:,:,0]
        
        zi = np.zeros(len(xyi))
        ci = np.zeros(len(xyi))
        onIDs = np.where(distances[:,0] > 0)[0]
        zi[onIDs] = np.average(z_vals[onIDs,:],weights=(1./distances[onIDs,:]), axis=1)
        ci[onIDs] = np.average(c_vals[onIDs,:],weights=(1./distances[onIDs,:]), axis=1)

        onIDs = np.where(distances[:,0] == 0)[0]
        
        if len(onIDs) > 0:
            zi[onIDs] = z[indices[onIDs,0],0]
            ci[onIDs] = c[indices[onIDs,0],0]

        self.zi = np.reshape(zi,(self.ny,self.nx))
        self.cumchange = np.reshape(ci,(self.ny,self.nx))

        
        return
    
    def plotSectionMap(self, title='Location of the cross-section', xlegend=None, ylegend=None, color=None,
                       colorcs=None, crange=None, cs=None, ctr='k',size=(8,8)):
        """
        Plot a given set of sections on the map

        Parameters
        ----------
        variable: title
            Title of the plot

        variable: color
            Colormap of the topography
            
        variable: colorcs
            Color of the cross-section

        variable: crange
            Range of values for the topography

        variable: cs
            Defined cross-section

        variable: size
            Figure size

        """

        rcParams['figure.figsize'] = size
        ax=plt.gca()
        
        im = ax.imshow(np.flipud(self.zi),interpolation='nearest',cmap=color,
                           vmin=crange[0], vmax=crange[1], extent=[self.bbox[0], self.bbox[2], self.bbox[1], self.bbox[3]])

        plt.contour(self.xi, self.yi, self.zi, (0,), colors=ctr, linewidths=2)

        if cs is not None:
            plt.plot(cs[:,0],cs[:,1], '-x', color=colorcs, markersize=4)
        plt.title(title)
        plt.xlabel(xlegend)
        plt.ylabel(ylegend)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        plt.colorbar(im,cax=cax)
        plt.show()
        plt.close()

        return

    def _cross_section(self, xo, yo, xm, ym, pts):
        """
        Compute cross section coordinates.
        """

        if xm == xo:
            ysec = np.linspace(yo, ym, pts)
            xsec = np.zeros(pts)
            xsec.fill(xo)
        elif ym == yo:
            xsec = np.linspace(xo, xm, pts)
            ysec = np.zeros(pts)
            ysec.fill(yo)
        else:
            a = (ym-yo)/(xm-xo)
            b = yo - a * xo
            xsec = np.linspace(xo, xm, pts)
            ysec = a * xsec + b

        return xsec, ysec

    def buildSection(self, xo = None, yo = None, xm = None, ym = None,
                    pts = 100, gfilter = 5):
        """
        Extract a slice from the 3D data set and compute the stratigraphic layers.
        Parameters
        ----------
        variable: xo, yo
            Lower X,Y coordinates of the cross-section.
        variable: xm, ym
            Upper X,Y coordinates of the cross-section.
        variable: pts
            Number of points to discretise the cross-section.
        variable: gfilter
            Gaussian smoothing filter.
        """
        
        if pts is None:
            pts = self.nx * 10

        if xm > self.x.max():
            xm = self.x.max()

        if ym > self.y.max():
            ym = self.y.max()

        if xo < self.x.min():
            xo = self.x.min()

        if yo < self.y.min():
            yo = self.y.min()

        xsec, ysec = self._cross_section(xo, yo, xm, ym, pts)
        self.dist = np.sqrt(( xsec - xo )**2 + ( ysec - yo )**2)
        self.xsec = xsec
        self.ysec = ysec
        for k in range(self.nz):
            # Thick
            rect_B_spline = RectBivariateSpline(self.yi, self.xi, self.th[:,:,k])
            data = rect_B_spline.ev(ysec, xsec)
            secTh = filters.gaussian_filter1d(data,sigma=gfilter)
            secTh[secTh < 0] = 0
            self.secTh.append(secTh)

            # Elev
            rect_B_spline1 = RectBivariateSpline(self.yi, self.xi, self.elev[:,:,k])
            data1 = rect_B_spline1.ev(ysec, xsec)
            secElev = filters.gaussian_filter1d(data1,sigma=gfilter)
            self.secElev.append(secElev)

            # Depth
            rect_B_spline2 = RectBivariateSpline(self.yi, self.xi, self.dep[:,:,k])
            data2 = rect_B_spline2.ev(ysec, xsec)
            secDep = filters.gaussian_filter1d(data2,sigma=gfilter)
            self.secDep.append(secDep)

        # Ensure the spline interpolation does not create underlying layers above upper ones
        topsec = self.secDep[self.nz-1]
        for k in range(self.nz-2,-1,-1):
            secDep = self.secDep[k]
            self.secDep[k] = np.minimum(secDep, topsec)
            topsec = self.secDep[k]

        return