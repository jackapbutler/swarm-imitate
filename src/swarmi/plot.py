
from plot4d import plotter
import matplotlib.pyplot as plt
import os
import imageio
import numpy as np

def plot4d(func, z_values, path="", wbounds=None, frame=plotter.Frame2D(), save_images=False, fps=1, func_name=None, z_label='z', png_path=None):
    if png_path==None:
        png_path = os.getcwd() + "/plot4d_temp"
    filenames = []
    
    values = []
    
    if not wbounds:
        # find the wbounds first then plot
        wmin = +float('inf')
        wmax = -float('inf')
        for z in z_values:
            x, y, w = plotter._evaluate(func, frame, z)
            wmin = min(wmin, w.min())
            wmax = max(wmax, w.max())
            values.append((x,y,w,z))
        
        for x,y,w,z in values:
            fn = _plot(x, y, w, frame, z, z_label, (wmin, wmax), png_path, func_name, show=False)
            filenames.append(fn)
    else:
        for z in z_values:
            x, y, w = plotter._evaluate(func, frame, z)
            fn = _plot(x, y, w, frame, z, z_label, wbounds, png_path, func_name, show=False)
            filenames.append(fn)
    
    gif_name = "Cross Sections" if func_name==None else func_name
    gif_name += ".gif"
    gif_name = path+gif_name
    with imageio.get_writer(gif_name, mode='I', fps=fps) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            if not save_images:
                os.remove(filename)

    # remove path if folder is empty
    if not os.listdir(png_path):
        os.rmdir(png_path)
        
    print("Animation saved as \"%s\""%gif_name)
    
    return gif_name

def _plot(x, y, w, frame, z_plot, z_label, wbounds=None, path=None, func_name=None, show=True):
    # Save plot if path is set, show plot if show==True. Otherwise do nothing and return nothing. 
    W = w.reshape(frame.xnum, frame.ynum)
    if not wbounds:
        wbounds = (W.min(), W.max())

    plt.imshow(W, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
    plt.colorbar()
    plt.clim(wbounds)    
    plt.xlabel(frame.xlabel)
    plt.ylabel(frame.ylabel)
    if func_name is None:
        func_name = "Crosssection"
    title_str = "%s at %s=%.2f"%(func_name, z_label, z_plot)
    plt.title(title_str)
    
    filename = None
    if path:
        save_path = path
        save_path += '/' if path[-1]!= '/' else ''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        filename = save_path + title_str + ".png"
        plt.savefig(filename)
        
    plt.show() if show else plt.close()
    
    return filename