import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.colors

def gparams_rho2D_polar():

    """" ===== Plot parameters ====="""
    #vmin,vmax: float : value range for the plot
    #ncont:      int: number of contours

    vmin    = 0.0
    vmax    = 0.1
    ncont   = 200
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
                            "title_size":       12,
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
                            "xlabel_size":       12,
                            "ylabel_size":       12,   
                            "xlabel_pad":       -5.0,     
                            "ylabel_pad":       -5.0,
                            "xlabel_loc":       "center",  #left, right      
                            "ylabel_loc":       "center",  #left, right        
                            "nticks_rad":        12,
                            "nticks_th":         12,

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