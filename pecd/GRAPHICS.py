import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.colors

def gparams_rho2D_polar():

    """" ===== Plot parameters ====="""
    #xrange,yrange: tuple (xmin,xmax): x,y-range for the plot
    #vmin,vmax: float : value range for the plot
    #ncont:      int: number of contours
    nptsx = 200
    nptsy = 200

    xmax    = 5.0
    ymax    = 5.0
    xrange  = (-xmax, xmax)
    yrange  = (-ymax, ymax)

    vmin    = -1.0
    vmax    = 1.0
    ncont   = 100


    cont2D_params = {       "xrange":   xrange,
                            "yrange":   yrange,
                            "vmin":     vmin,
                            "vmax":     vmax,
                            "ncont":    ncont,

                            ### TITLE ###
                            "title_text":       "Title",
                            "title_color":      "blue",
                            "title_size":       15,
                            "title_vertical":   "baseline", #vertical alignment of the title: {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
                            "title_horizontal": "center", #{'center', 'left', 'right'},
                            #"title_position":   (pos_x,pos_y), #manual setting of title position
                            "title_pad":        None, #offset from the top of the axes given in points 
                            "title_fontstyle":  'normal', #{'normal', 'italic', 'oblique'}
                            "title_fontname":   'Helvetica', #'Sans' | 'Courier' | '
                            "title_background": "None",   

                            ### LABELS ###
                            "xlabel":           "x",
                            "ylabel":           "y",
                            "xlabel_format":    '%.2f',
                            "ylabel_format":    '%.1f',
                            "label_color":      'red',
                            "xlabel_size":       12,
                            "ylabel_size":       12,   
                            "xlabel_pad":       None,     
                            "ylabel_pad":       None,
                            "xlabel_loc":       "center",  #left, right         
                            "xticks":           list(np.linspace(xrange[0],xrange[1],4)),
                            "yticks":           list(np.linspace(yrange[0],yrange[1],8)), 
                                                
                            ### COLORBAR ###: see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
                            #"cbar_mappable":       matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                            "cbar_orientation":   'horizontal', #vertical #note that orientation overrides panchor
                            "cbar_label":         "Some units",   
                            "cbar_fraction":      1.0, #fraction of the original axes to be displayed in the colorbar
                            "cbar_aspect":        20, #ratio of long to short dimensions
                            "cbar_shrink":        1.00, #shrink the colorbar
                            "cbar_pad":           0.05, #distance of colorbar from the adjacent plot axis
                            "cbar_panchor":       (0.3,0.2), #TThe anchor point of the colorbar parent axes. If False, the parent axes' anchor will be unchanged. Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.
                            
                            "cbar_ticks":         None, #or list of custom ticks: list(np.linspace(vmin,vmax,10))
                            "cbar_drawedges":     False, #draw edges of colorbar
                            "cbar_label":         "some title", #title on long axis of colorbar
                            "cbar_format":        '%.1f', #format of tick labels
                            "cbar_extend":        "neither", #{'neither', 'both', 'min', 'max'} If not 'neither', make pointed end(s) for out-of- range values. These are set for a given colormap using the colormap set_under and set_over methods.

                            ### SAVE PROPERTIES ###       
                            "save":             True,
                            "save_name":        "cont2D.pdf",
                            "save_dpi":         'figure', #float or 'figure' for same resolution as figure
                            "save_orientation": 'landscape', #portrait
                            "save_bbox_inches": 'tight', #or float in inches - which portion of the figure to save?
                            "save_pad_inches":   0.1,


                            ### FIGURE PROPERTIES ###    
                            "figsize_x":        3.5,
                            "figsize_y":        3.5,
                            "resolution":       200
                            }
    return cont2D_params