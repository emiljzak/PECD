import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.colors



def gparams_rho1D_ini_rad():

    cont1D_params = { }


def gparams_rho1D_wf_rad():

    cont1D_params = { }


def gparams_rho2D_polar():

    """" ===== Plot parameters ====="""
    #vmin,vmax: float : value range for the plot
    #ncont:      int: number of contours

    vmin    = 0.0
    vmax    = 1.0
    ncont   = 300
    resolution = 200
    fig_size_x = 3.5
    fig_size_y = 3.5


    cont2D_params = {      
                            "vmin":     vmin,
                            "vmax":     vmax,
                            "ncont":    ncont,

                            ### TITLE ###
                            "title_text":       "electron density",
                            "title_color":      "black",
                            "title_size":       8,
                            "title_vertical":   "baseline", #vertical alignment of the title: {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
                            "title_horizontal": "center", #{'center', 'left', 'right'},
                            #"title_position":   (pos_x,pos_y), #manual setting of title position
                            "title_pad":        None, #offset from the top of the axes given in points 
                            "title_fontstyle":  'normal', #{'normal', 'italic', 'oblique'}
                            "title_fontname":   'Sans', #'Sans' | 'Courier' | '
                            "title_background": "None",   

                            ### LABELS ###
                            "xlabel_format":    '%.0f',
                            "ylabel_format":    '%.0f',
                            "label_color":      'yellow',
                            "xlabel_size":       10,
                            "ylabel_size":       10,   
                            "xlabel_pad":       -10.0,     
                            "ylabel_pad":       -5.0,
                            "xlabel_loc":       "center",  #left, right      
                            "ylabel_loc":       "center",  #left, right        
                            "nticks_rad":        12,
                            "nticks_th":         12,

                            ### TIME TEXT LABEL ###

                            "time_size":        8,
                            "time_colour":      'red',

                            ### COLORBAR ###: see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
                            #"cbar_mappable":       matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                            "cbar_orientation":   'horizontal', #vertical #note that orientation overrides panchor
                            "cbar_fraction":      0.7, #fraction of the original axes to be displayed in the colorbar
                            "cbar_aspect":        40, #ratio of long to short dimensions
                            "cbar_shrink":        0.70, #shrink the colorbar
                            "cbar_pad":           0.05, #distance of colorbar from the adjacent plot axis
                            "cbar_panchor":       (0.3,0.2), #TThe anchor point of the colorbar parent axes. If False, the parent axes' anchor will be unchanged. Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.
                            
                            "cbar_ticks":         None, #or list of custom ticks: list(np.linspace(vmin,vmax,10))
                            "cbar_drawedges":     False, #draw edges of colorbar
                            "cbar_label":         None,#"some title", #title on long axis of colorbar
                            "cbar_format":        '%.2f', #format of tick labels
                            "cbar_extend":        "neither", #{'neither', 'both', 'min', 'max'} If not 'neither', make pointed end(s) for out-of- range values. These are set for a given colormap using the colormap set_under and set_over methods.

                            ### SAVE PROPERTIES ###       
                            "save":             True,
                            "save_name":        "rho2D_",
                            "save_dpi":         'figure', #float or 'figure' for same resolution as figure
                            "save_orientation": 'landscape', #portrait
                            "save_bbox_inches": 'tight', #or float in inches - which portion of the figure to save?
                            "save_pad_inches":   0.1,


                            ### FIGURE PROPERTIES ###    
                            "figsize_x":        fig_size_x,
                            "figsize_y":        fig_size_y,
                            "resolution":       resolution
                            }
    return cont2D_params


def gparams_W2Dav_polar():

    """" ===== Plot parameters ====="""
    #vmin,vmax: float : value range for the plot
    #ncont:      int: number of contours

    vmin    = 0.0
    vmax    = 1.0
    ncont   = 100
    resolution = 200
    fig_size_x = 3.5
    fig_size_y = 3.5


    cont2D_params = {      
                            "vmin":     vmin,
                            "vmax":     vmax,
                            "ncont":    ncont,

                            ### TITLE ###
                            "title_text":       "electron momentum probability distribution",
                            "title_color":      "black",
                            "title_size":       10,
                            "title_vertical":   "baseline", #vertical alignment of the title: {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
                            "title_horizontal": "center", #{'center', 'left', 'right'},
                            #"title_position":   (pos_x,pos_y), #manual setting of title position
                            "title_pad":        None, #offset from the top of the axes given in points 
                            "title_fontstyle":  'normal', #{'normal', 'italic', 'oblique'}
                            "title_fontname":   'Sans', #'Sans' | 'Courier' | '
                            "title_background": "None",   

                            ### LABELS ###
                            "xlabel_format":    '%.0f',
                            "ylabel_format":    '%.0f',
                            "label_color":      'yellow',
                            "xlabel_size":       10,
                            "ylabel_size":       10,   
                            "xlabel_pad":       -10.0,     
                            "ylabel_pad":       -5.0,
                            "xlabel_loc":       "center",  #left, right      
                            "ylabel_loc":       "center",  #left, right        
                            "nticks_rad":        12,
                            "nticks_th":         12,

                            ### TIME TEXT LABEL ###

                            "time_size":        8,
                            "time_colour":      'red',
                            ### COLORBAR ###: see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
                            #"cbar_mappable":       matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                            "cbar_orientation":   'horizontal', #vertical #note that orientation overrides panchor
                            "cbar_label":         "Some units",   
                            "cbar_fraction":      0.7, #fraction of the original axes to be displayed in the colorbar
                            "cbar_aspect":        40, #ratio of long to short dimensions
                            "cbar_shrink":        0.70, #shrink the colorbar
                            "cbar_pad":           0.05, #distance of colorbar from the adjacent plot axis
                            "cbar_panchor":       (0.3,0.2), #TThe anchor point of the colorbar parent axes. If False, the parent axes' anchor will be unchanged. Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.
                            
                            "cbar_ticks":         None, #or list of custom ticks: list(np.linspace(vmin,vmax,10))
                            "cbar_drawedges":     False, #draw edges of colorbar
                            "cbar_label":         None,#"some title", #title on long axis of colorbar
                            "cbar_format":        '%.2f', #format of tick labels
                            "cbar_extend":        "neither", #{'neither', 'both', 'min', 'max'} If not 'neither', make pointed end(s) for out-of- range values. These are set for a given colormap using the colormap set_under and set_over methods.

                            ### SAVE PROPERTIES ###       
                            "save":             True,
                            "save_name":        "W2Dav_",
                            "save_dpi":         'figure', #float or 'figure' for same resolution as figure
                            "save_orientation": 'landscape', #portrait
                            "save_bbox_inches": 'tight', #or float in inches - which portion of the figure to save?
                            "save_pad_inches":   0.1,


                            ### FIGURE PROPERTIES ###    
                            "figsize_x":        fig_size_x,
                            "figsize_y":        fig_size_y,
                            "resolution":       resolution
                            }
    return cont2D_params



def gparams_W2D_polar():

    """" ===== Plot parameters ====="""
    #vmin,vmax: float : value range for the plot
    #ncont:      int: number of contours

    vmin    = 0.0
    vmax    = 1.0
    ncont   = 200
    resolution = 200
    fig_size_x = 3.5
    fig_size_y = 3.5


    cont2D_params = {      
                            "vmin":     vmin,
                            "vmax":     vmax,
                            "ncont":    ncont,

                            ### TITLE ###
                            "title_text":       "electron momentum probability distribution",
                            "title_color":      "black",
                            "title_size":       10,
                            "title_vertical":   "baseline", #vertical alignment of the title: {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
                            "title_horizontal": "center", #{'center', 'left', 'right'},
                            #"title_position":   (pos_x,pos_y), #manual setting of title position
                            "title_pad":        None, #offset from the top of the axes given in points 
                            "title_fontstyle":  'normal', #{'normal', 'italic', 'oblique'}
                            "title_fontname":   'Sans', #'Sans' | 'Courier' | '
                            "title_background": "None",   

                            ### LABELS ###
                            "xlabel_format":    '%.0f',
                            "ylabel_format":    '%.0f',
                            "label_color":      'yellow',
                            "xlabel_size":       10,
                            "ylabel_size":       10,   
                            "xlabel_pad":       -10.0,     
                            "ylabel_pad":       -5.0,
                            "xlabel_loc":       "center",  #left, right      
                            "ylabel_loc":       "center",  #left, right        
                            "nticks_rad":        12,
                            "nticks_th":         12,

                            ### TIME TEXT LABEL ###

                            "time_size":        8,
                            "time_colour":      'red',
                            ### COLORBAR ###: see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
                            #"cbar_mappable":       matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                            "cbar_orientation":   'horizontal', #vertical #note that orientation overrides panchor
                            "cbar_label":         "Some units",   
                            "cbar_fraction":      0.7, #fraction of the original axes to be displayed in the colorbar
                            "cbar_aspect":        40, #ratio of long to short dimensions
                            "cbar_shrink":        0.70, #shrink the colorbar
                            "cbar_pad":           0.05, #distance of colorbar from the adjacent plot axis
                            "cbar_panchor":       (0.3,0.2), #TThe anchor point of the colorbar parent axes. If False, the parent axes' anchor will be unchanged. Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.
                            
                            "cbar_ticks":         None, #or list of custom ticks: list(np.linspace(vmin,vmax,10))
                            "cbar_drawedges":     False, #draw edges of colorbar
                            "cbar_label":         None,#"some title", #title on long axis of colorbar
                            "cbar_format":        '%.2f', #format of tick labels
                            "cbar_extend":        "neither", #{'neither', 'both', 'min', 'max'} If not 'neither', make pointed end(s) for out-of- range values. These are set for a given colormap using the colormap set_under and set_over methods.

                            ### SAVE PROPERTIES ###       
                            "save":             True,
                            "save_name":        "W2D_",
                            "save_dpi":         'figure', #float or 'figure' for same resolution as figure
                            "save_orientation": 'landscape', #portrait
                            "save_bbox_inches": 'tight', #or float in inches - which portion of the figure to save?
                            "save_pad_inches":   0.1,


                            ### FIGURE PROPERTIES ###    
                            "figsize_x":        fig_size_x,
                            "figsize_y":        fig_size_y,
                            "resolution":       resolution
                            }
    return cont2D_params




def gparams_PES():

    """" ===== Plot parameters ====="""
    resolution = 200
    fig_size_x = 3.5
    fig_size_y = 3.5


    cont1D_params = {      

                            ### TITLE ###
                            "title_text":       "Photoelectron energy cross-section",
                            "title_color":      "black",
                            "title_size":       10,
                            "title_vertical":   "baseline", #vertical alignment of the title: {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
                            "title_horizontal": "center", #{'center', 'left', 'right'},
                            "title_pad":        None, #offset from the top of the axes given in points 
                            "title_fontstyle":  'normal', #{'normal', 'italic', 'oblique'}
                            "title_fontname":   'Sans', #'Sans' | 'Courier' | '
                            "title_background": "None",   

                            ### GRAPH ###
                            "plot_colour":      'red',
                            "plot_marker":      '.',
                            "plot_label":       r"$\sigma(k)$",
                            "markersize":       4,
                            ### LABELS ###
                            "xlabel_format":    '%.0f',
                            "ylabel_format":    '%.0f',
                            "label_color":      'black',
                            "xlabel_size":       10,
                            "ylabel_size":       10,   
                            "xlabel_pad":       -10.0,     
                            "ylabel_pad":       -5.0,
                            "xlabel_loc":       "center",  #left, right      
                            "ylabel_loc":       "center",  #left, right        
                            "nticks_rad":        12,
                            "nticks_th":         12,


                            ### TIME TEXT LABEL ###

                            "time_size":        8,
                            "time_colour":      'black',

                            ### SAVE PROPERTIES ###       
                            "save":             True,
                            "save_name":        "PES_",
                            "save_dpi":         'figure', #float or 'figure' for same resolution as figure
                            "save_orientation": 'landscape', #portrait
                            "save_bbox_inches": 'tight', #or float in inches - which portion of the figure to save?
                            "save_pad_inches":   0.1,


                            ### FIGURE PROPERTIES ###    
                            "figsize_x":        fig_size_x,
                            "figsize_y":        fig_size_y,
                            "resolution":       resolution
                            }
    return cont1D_params



def gparams_PECD1D():

    """" ===== Plot parameters ====="""

    vmin    = 0.0
    vmax    = 1.0
    ncont   = 200
    resolution = 200
    fig_size_x = 3.5
    fig_size_y = 3.5


    cont1D_params = {      

                            ### TITLE ###
                            "title_text":       "Photoelectron energy cross-section",
                            "title_color":      "black",
                            "title_size":       10,
                            "title_vertical":   "baseline", #vertical alignment of the title: {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
                            "title_horizontal": "center", #{'center', 'left', 'right'},
                            "title_pad":        None, #offset from the top of the axes given in points 
                            "title_fontstyle":  'normal', #{'normal', 'italic', 'oblique'}
                            "title_fontname":   'Sans', #'Sans' | 'Courier' | '
                            "title_background": "None",   

                            ### GRAPH ###
                            "plot_colour":      'red',
                            "plot_marker":      '.',
                            "plot_label":       r"$\sigma(k)$",

                            ### LABELS ###
                            "xlabel_format":    '%.0f',
                            "ylabel_format":    '%.0f',
                            "label_color":      'black',
                            "xlabel_size":       10,
                            "ylabel_size":       10,   
                            "xlabel_pad":       -10.0,     
                            "ylabel_pad":       -5.0,
                            "xlabel_loc":       "center",  #left, right      
                            "ylabel_loc":       "center",  #left, right        
                            "nticks_rad":        12,
                            "nticks_th":         12,


                            ### TIME TEXT LABEL ###

                            "time_size":        8,
                            "time_colour":      'black',

                            ### SAVE PROPERTIES ###       
                            "save":             True,
                            "save_name":        "PECD1D_",
                            "save_dpi":         'figure', #float or 'figure' for same resolution as figure
                            "save_orientation": 'landscape', #portrait
                            "save_bbox_inches": 'tight', #or float in inches - which portion of the figure to save?
                            "save_pad_inches":   0.1,


                            ### FIGURE PROPERTIES ###    
                            "figsize_x":        fig_size_x,
                            "figsize_y":        fig_size_y,
                            "resolution":       resolution
                            }
    return cont1D_params



def gparams_PECD2D():

    """" ===== Plot parameters ====="""

    vmin    = 0.0
    vmax    = 1.0
    ncont   = 200
    resolution = 200
    fig_size_x = 3.5
    fig_size_y = 3.5


    cont2D_params = {      
                            "vmin":     vmin,
                            "vmax":     vmax,
                            "ncont":    ncont,

                            ### TITLE ###
                            "title_text":       "electron momentum probability distribution",
                            "title_color":      "black",
                            "title_size":       10,
                            "title_vertical":   "baseline", #vertical alignment of the title: {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
                            "title_horizontal": "center", #{'center', 'left', 'right'},
                            #"title_position":   (pos_x,pos_y), #manual setting of title position
                            "title_pad":        None, #offset from the top of the axes given in points 
                            "title_fontstyle":  'normal', #{'normal', 'italic', 'oblique'}
                            "title_fontname":   'Sans', #'Sans' | 'Courier' | '
                            "title_background": "None",   

                            ### LABELS ###
                            "xlabel_format":    '%.0f',
                            "ylabel_format":    '%.0f',
                            "label_color":      'yellow',
                            "xlabel_size":       10,
                            "ylabel_size":       10,   
                            "xlabel_pad":       -10.0,     
                            "ylabel_pad":       -5.0,
                            "xlabel_loc":       "center",  #left, right      
                            "ylabel_loc":       "center",  #left, right        
                            "nticks_rad":        12,
                            "nticks_th":         12,

                            ### TIME TEXT LABEL ###

                            "time_size":        8,
                            "time_colour":      'red',
                            ### COLORBAR ###: see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
                            #"cbar_mappable":       matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                            "cbar_orientation":   'horizontal', #vertical #note that orientation overrides panchor
                            "cbar_label":         "Some units",   
                            "cbar_fraction":      0.7, #fraction of the original axes to be displayed in the colorbar
                            "cbar_aspect":        40, #ratio of long to short dimensions
                            "cbar_shrink":        0.70, #shrink the colorbar
                            "cbar_pad":           0.05, #distance of colorbar from the adjacent plot axis
                            "cbar_panchor":       (0.3,0.2), #TThe anchor point of the colorbar parent axes. If False, the parent axes' anchor will be unchanged. Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.
                            
                            "cbar_ticks":         None, #or list of custom ticks: list(np.linspace(vmin,vmax,10))
                            "cbar_drawedges":     False, #draw edges of colorbar
                            "cbar_label":         None,#"some title", #title on long axis of colorbar
                            "cbar_format":        '%.2f', #format of tick labels
                            "cbar_extend":        "neither", #{'neither', 'both', 'min', 'max'} If not 'neither', make pointed end(s) for out-of- range values. These are set for a given colormap using the colormap set_under and set_over methods.

                            ### SAVE PROPERTIES ###       
                            "save":             True,
                            "save_name":        "PECD2D_",
                            "save_dpi":         'figure', #float or 'figure' for same resolution as figure
                            "save_orientation": 'landscape', #portrait
                            "save_bbox_inches": 'tight', #or float in inches - which portion of the figure to save?
                            "save_pad_inches":   0.1,


                            ### FIGURE PROPERTIES ###    
                            "figsize_x":        fig_size_x,
                            "figsize_y":        fig_size_y,
                            "resolution":       resolution,
                            }
    return cont2D_params

def gparams_barray2D():
    """" ===== Plot parameters ====="""
    #xrange,yrange: tuple (xmin,xmax): x,y-range for the plot
    #vmin,vmax: float : value range for the plot
    #ncont:      int: number of contours


    xmax    = 2.0 * np.pi
    ymax    = 1.0 * np.pi
    xrange  = (0.0, xmax)
    yrange  = (0.0, ymax)

    vmin    = -1.0
    vmax    = 1.0
    ncont   = 10

    params = {   "xrange":   xrange,
                        "yrange":   yrange,
                        "vmin":     vmin,
                        "vmax":     vmax,
                        "ncont":    ncont,

                        ### TITLE ###
                        "title_text":       "Title",
                        "title_color":      "black",
                        "title_size":       10,
                        "title_vertical":   "baseline", #vertical alignment of the title: {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
                        "title_horizontal": "center", #{'center', 'left', 'right'},
                        #"title_position":   (pos_x,pos_y), #manual setting of title position
                        "title_pad":        None, #offset from the top of the axes given in points 
                        "title_fontstyle":  'normal', #{'normal', 'italic', 'oblique'}
                        "title_fontname":   'Helvetica', #'Sans' | 'Courier' | '
                        "title_background": "None",   

                        ### LABELS ###
                        "xlabel":           "Euler angle "+r'$\gamma$'+"(in "+r'$\pi$'+" units)",
                        "ylabel":           "Euler angle "+r'$\beta$'+"(in "+r'$\pi$'+" units)",
                        "xlabel_format":    '%.1f',
                        "ylabel_format":    '%.1f',
                        "label_color":      'black',
                        "xlabel_size":       10,
                        "ylabel_size":       10,   
                        "xlabel_pad":       None,     
                        "ylabel_pad":       None,
                        "xlabel_loc":       "center",  #left, right         
                        "xticks":           list(np.linspace(xrange[0]/np.pi,xrange[1]/np.pi,5)),
                        "yticks":           list(np.linspace(yrange[0]/np.pi,yrange[1]/np.pi,3)), 
                                            
                        ### COLORBAR ###: see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
                       
                        "cbar_orientation":   'vertical', #vertical #note that orientation overrides panchor
                        "cbar_fraction":      1.0, #fraction of the original axes to be displayed in the colorbar
                        "cbar_aspect":        40, #ratio of long to short dimensions
                        "cbar_shrink":        1.0, #shrink the colorbar
                        "cbar_pad":           0.0, #distance of colorbar from the adjacent plot axis
                        "cbar_panchor":       (0.3,0.2), #TThe anchor point of the colorbar parent axes. If False, the parent axes' anchor will be unchanged. Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.
                        
                        "cbar_ticks":         None, #or list of custom ticks: list(np.linspace(vmin,vmax,10))
                        "cbar_drawedges":     False, #draw edges of colorbar
                        "cbar_label":         "some title", #title on long axis of colorbar
                        "cbar_format":        '%.1f', #format of tick labels
                        "cbar_extend":        "neither", #{'neither', 'both', 'min', 'max'} If not 'neither', make pointed end(s) for out-of- range values. These are set for a given colormap using the colormap set_under and set_over methods.

                        ### SAVE PROPERTIES ###       
                        "save":             True,
                        "save_name":        "barray2D.pdf",
                        "save_dpi":         'figure', #float or 'figure' for same resolution as figure
                        "save_orientation": 'landscape', #portrait
                        "save_bbox_inches": 'tight', #or float in inches - which portion of the figure to save?
                        "save_pad_inches":   0.1,


                        ### FIGURE PROPERTIES ###    
                        "figsize_x":        4.0,
                        "figsize_y":        2.0,
                        "resolution":       300}                       
    return params