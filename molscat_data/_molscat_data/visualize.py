from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import lines
from matplotlib import ticker
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .effective_probability import effective_probability, p0
from .analytical import probabilities_from_matrix_elements_cold, probabilities_from_matrix_elements_hot
from .chi_squared import chi_squared

class BicolorHandler:
    def __init__(self, color1, color2, hatch):
        self.color1=color1
        self.color2=color2
        self.hatch=hatch
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = plt.Rectangle([x0, y0], width, height, facecolor='none',
                                   edgecolor='k', transform=handlebox.get_transform(), hatch=self.hatch)
        patch1 = plt.Rectangle([x0, y0], width*2/3, height, facecolor=self.color1,
                                   edgecolor='none', transform=handlebox.get_transform())
        patch2 = plt.Rectangle([x0+width*2/3., y0], width*1/3., height, facecolor=self.color2,
                                   edgecolor='none', transform=handlebox.get_transform())
        handlebox.add_artist(patch2)
        handlebox.add_artist(patch1)
        handlebox.add_artist(patch)
        
        return patch


class BarplotWide:
    """Wide bar plot for the single-ion RbSr+ experiment."""

    @staticmethod
    def _initiate_plot(figsize = (16,5), dpi = 100):
        fig= plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(1, 16, (1,5))
        ax2 = fig.add_subplot(1, 16, (6,10))
        ax3 = fig.add_subplot(1, 16, (11,13))
        legend_ax = fig.add_subplot(1, 16, (14,16))
        legend_ax.axis('off')
        
        return fig, ax1, ax2, ax3, legend_ax
    
    @staticmethod
    def _initiate_plot_ten_bars(figsize = (10,5), dpi = 100):
        fig= plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(1, 10, (1,5))
        ax2 = fig.add_subplot(1, 10, (6,10))
        
        return fig, ax1, ax2
    
    @classmethod
    def barplot(cls, theory_data, exp_data, std_data, SE_theory_data = None, figsize = (16,5), dpi = 100):
        """barplot
        
        :param theory_data: dictionary of the form {'hpf': (p(-2), p(-1), ..., p(2)),
        'cold_higher': (p(-2), p(-1), ..., p(2)), 'cold_lower': (p(-1), p(0), p(1)) }
        :param exp_data: dictionary of the form {'hpf': (p(-2), p(-1), ..., p(2)),
        'cold_higher': (p(-2), p(-1), ..., p(2)), 'cold_lower': (p(-1), p(0), p(1)) }
        """

        fig, ax1, ax2, ax3, legend_ax = cls._initiate_plot(figsize, dpi)

        probabilities_theory_f_max_hot = theory_data['hpf']
        probabilities_theory_f_max_cold = theory_data['cold_higher']
        probabilities_theory_f_min_cold = theory_data['cold_lower']
        
        probabilities_exp_f_max_hot = exp_data['hpf']
        probabilities_exp_f_max_cold = exp_data['cold_higher']
        probabilities_exp_f_min_cold = exp_data['cold_lower']

        std_f_max_hot = std_data.get('hpf', np.full_like(probabilities_exp_f_max_hot, 0))
        std_f_max_cold = std_data.get('cold_higher', np.full_like(probabilities_exp_f_max_cold, 0))
        std_f_min_cold = std_data.get('cold_lower', np.full_like(probabilities_exp_f_min_cold, 0))

        cold_color, hot_color = 'midnightblue', 'firebrick'
        exp_cold_color, exp_hot_color = 'midnightblue', 'firebrick'
        exp_hatch, theory_hatch = '////', ''

        if SE_theory_data is not None:
            SE_cold_color, SE_hot_color = 'midnightblue', 'firebrick'
            cold_color, hot_color = 'royalblue', 'indianred'

        y_max = 1
        
        f_max = 2.
        f_min = f_max-1
        f_ion = 0.5

        positions_f_max = np.array([ [3*k+1, 3*k+2] for k in np.arange(2*f_max+1) ]).flatten()
        positions_theory_f_max = np.array([3*k+1 for k in np.arange(2*f_max+1)])
        positions_exp_f_max = np.array([3*k+2 for k in np.arange(2*f_max+1)])
        positions_f_min = np.array([ [3*k+1, 3*k+2] for k in np.arange(2*f_min+1) ]).flatten()
        positions_theory_f_min = np.array([3*k+1 for k in np.arange(2*f_min+1)])
        positions_exp_f_min = np.array([3*k+2 for k in np.arange(2*f_min+1)])

        ticks_f_max = [ 3*k + 3/2 for k in np.arange(2*f_max+1)]
        ticks_f_min = [ 3*k + 3/2 for k in np.arange(2*f_min+1)]

        labels_f_max = [ '$\\left|\\right.$'+str(int(f_max))+', '+str(int(mf))+'$\\left.\\right>$' for mf in np.arange (-f_max, f_max+1)]
        labels_f_min = [ '$\\left|\\right.$'+str(int(f_min))+', '+str(int(mf))+'$\\left.\\right>$' for mf in np.arange (-f_min, f_min+1)]

        ax1.bar(positions_theory_f_max, probabilities_theory_f_max_hot, width = 1, color = hot_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
        ax1.bar(positions_exp_f_max, probabilities_exp_f_max_hot, yerr = std_f_max_hot, width = 1, color = exp_hot_color, hatch = exp_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)

        ax1.set_xticks(ticks_f_max)
        ax1.set_xticklabels(labels_f_max, fontsize = 'x-large')
        ax1.grid(color = 'gray', axis='y')
        ax1.set_axisbelow(True)
        ax1.tick_params(axis = 'y', labelsize = 'large')
        ax1.set_ylim(0,y_max)

        ax2.bar(positions_theory_f_max, probabilities_theory_f_max_cold, width = 1, color = cold_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
        ax2.bar(positions_exp_f_max, probabilities_exp_f_max_cold, yerr = std_f_max_cold, width = 1, color = exp_cold_color, hatch = exp_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)

        ax2.set_xticks(ticks_f_max)
        ax2.set_xticklabels(labels_f_max, fontsize = 'x-large')
        ax2.grid(color = 'gray', axis='y')
        ax2.set_axisbelow(True)
        ax2.tick_params(axis = 'y', labelsize = 'large')
        ax2.set_ylim(0,y_max)

        ax3.bar(positions_theory_f_min, probabilities_theory_f_min_cold, width = 1, color = cold_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)     
        ax3.bar(positions_exp_f_min, probabilities_exp_f_min_cold, yerr = std_f_min_cold, width = 1, color = exp_cold_color, hatch = exp_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
        
        ax3.set_xticks(ticks_f_min)
        ax3.set_xticklabels(labels_f_min, fontsize = 'x-large')
        ax3.grid(color = 'gray', axis='y')
        ax3.set_axisbelow(True)
        ax3.tick_params(axis = 'y', labelsize = 'large')
        ax3.set_ylim(0,y_max)

        if SE_theory_data is not None:
            ax1.bar(positions_theory_f_max, SE_theory_data['hpf'], width = 1, facecolor = SE_hot_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
            ax2.bar(positions_theory_f_max, SE_theory_data['cold_higher'], width = 1, color = SE_cold_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
            ax3.bar(positions_theory_f_min, SE_theory_data['cold_lower'], width = 1, color = SE_cold_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)     

        ### Create legend
        # mpl.rcParams['hatch.linewidth'] = 2.0
        labels_and_colors = { 'hyperfine relaxation': exp_hot_color, 'cold spin change': exp_cold_color } 
        labels_and_hatch = { 'coupled-channel\ncalculations': '', 'experiment': '////' }
        interlude = 'estimation from the\nmatrix elements:'
        labels_and_lines = { 'normalized\nto $p_\\mathrm{hot}(\\left|\\right.$'+str(int(2))+', '+str(int(-2))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('k', 'D', 8), 'normalized\nto $p_\\mathrm{cold}(\\left|\\right.$'+str(int(2))+', '+str(int(-2))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('magenta', 'x', 10), 'normalized\nto $p_\\mathrm{cold}(\\left|\\right.$'+str(int(1))+', '+str(int(-1))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('orange', '2', 16) }
        handles_colors = [ plt.Rectangle((0,0), 1, 1, facecolor = labels_and_colors[color_label], edgecolor = 'k', hatch = '' ) for color_label in labels_and_colors.keys() ]
        handles_hatch = [ plt.Rectangle((0,0), 1, 1, facecolor = 'white', edgecolor = 'k', hatch = nhatch ) for nhatch in labels_and_hatch.values() ]
        handles_interlude = [ plt.Rectangle((0,0), 1, 1, facecolor = 'white', edgecolor = 'white', hatch = '' ) ]
        handles_lines = [ lines.Line2D([0], [0], color = labels_and_lines[line_label][0], linewidth = 3, marker = labels_and_lines[line_label][1], markersize = labels_and_lines[line_label][2]) for line_label in labels_and_lines.keys() ]


        colors_and_hatches = [ *[ (exp_hot_color, exp_hot_color, ''), (exp_cold_color, exp_cold_color, '') ],
                    *[('white', 'white', hatch) for hatch in labels_and_hatch.values()],
                     *[('white', 'white', '') for hh in handles_interlude]
                    ]
        if SE_theory_data is not None:
            labels_and_colors = { 'hyperfine relaxation\n(w/o & with $\\lambda_\\mathrm{so}$)': exp_hot_color,
                                  'cold spin change\n(w/o & with $\\lambda_\\mathrm{so}$)': exp_cold_color}
            colors_and_hatches = [ *[ (SE_hot_color, hot_color, ''), (SE_cold_color, cold_color, '') ],
                    *[('white', 'white', hatch) for hatch in labels_and_hatch.values()],
                     *[('white', 'white', '') for hh in handles_interlude]
                    ]
            


        labels = [ *list(labels_and_colors.keys()), *list(labels_and_hatch.keys()),]# interlude, *list(labels_and_lines.keys()) ]
        handles = [ *handles_colors, *handles_hatch, *handles_interlude,]# *handles_lines ]

        
        hmap = dict(zip(handles, [BicolorHandler(*color) for color in colors_and_hatches] ))
        legend_ax.legend(handles, labels, handler_map = hmap, loc = 'center', bbox_to_anchor = (8/16, 0.50), fontsize = 'large', labelspacing = 1)

        plt.tight_layout()
        
        return fig, ax1, ax2, ax3, legend_ax
    
    @classmethod
    def barplot_ten_bars(cls, theory_data, exp_data, std_data, SE_theory_data = None, figsize = (10,5), dpi = 300):
        """barplot
        
        :param theory_data: dictionary of the form {'hpf': (p(-2), p(-1), ..., p(2)),
        'cold_higher': (p(-2), p(-1), ..., p(2)), 'cold_lower': (p(-1), p(0), p(1)) }
        :param exp_data: dictionary of the form {'hpf': (p(-2), p(-1), ..., p(2)),
        'cold_higher': (p(-2), p(-1), ..., p(2)), 'cold_lower': (p(-1), p(0), p(1)) }
        """

        fig, ax1, ax2 = cls._initiate_plot_ten_bars(figsize, dpi)

        probabilities_theory_f_max_hot = theory_data['hpf']
        probabilities_theory_f_max_cold = theory_data['cold_higher']
        probabilities_theory_f_min_cold = theory_data['cold_lower']
        
        probabilities_exp_f_max_hot = exp_data['hpf']
        probabilities_exp_f_max_cold = exp_data['cold_higher']
        probabilities_exp_f_min_cold = exp_data['cold_lower']

        std_f_max_hot = std_data.get('hpf', np.full_like(probabilities_exp_f_max_hot, 0))
        std_f_max_cold = std_data.get('cold_higher', np.full_like(probabilities_exp_f_max_cold, 0))
        std_f_min_cold = std_data.get('cold_lower', np.full_like(probabilities_exp_f_min_cold, 0))

        cold_color, hot_color = 'midnightblue', 'firebrick'
        exp_cold_color, exp_hot_color = 'midnightblue', 'firebrick'
        exp_hatch, theory_hatch = '////', ''

        if SE_theory_data is not None:
            SE_cold_color, SE_hot_color = 'midnightblue', 'firebrick'
            cold_color, hot_color = 'royalblue', 'indianred'

        y_max = 1
        
        f_max = 2.
        f_min = f_max-1
        f_ion = 0.5

        positions_f_max = np.array([ [3*k+1, 3*k+2] for k in np.arange(2*f_max+1) ]).flatten()
        positions_theory_f_max = np.array([3*k+1 for k in np.arange(2*f_max+1)])
        positions_exp_f_max = np.array([3*k+2 for k in np.arange(2*f_max+1)])
        positions_f_min = np.array([ [3*k+1, 3*k+2] for k in np.arange(2*f_min+1) ]).flatten()
        positions_theory_f_min = np.array([3*k+1 for k in np.arange(2*f_min+1)])
        positions_exp_f_min = np.array([3*k+2 for k in np.arange(2*f_min+1)])

        ticks_f_max = [ 3*k + 3/2 for k in np.arange(2*f_max+1)]
        ticks_f_min = [ 3*k + 3/2 for k in np.arange(2*f_min+1)]

        labels_f_max = [ '$\\left|\\right.$'+str(int(f_max))+', '+str(int(mf))+'$\\left.\\right>$' for mf in np.arange (-f_max, f_max+1)]
        labels_f_min = [ '$\\left|\\right.$'+str(int(f_min))+', '+str(int(mf))+'$\\left.\\right>$' for mf in np.arange (-f_min, f_min+1)]

        ax1.bar(positions_theory_f_max, probabilities_theory_f_max_hot, width = 1, color = hot_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
        ax1.bar(positions_exp_f_max, probabilities_exp_f_max_hot, yerr = std_f_max_hot, width = 1, color = exp_hot_color, hatch = exp_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)

        ax1.set_xticks(ticks_f_max)
        ax1.set_xticklabels(labels_f_max, fontsize = 'x-large')
        ax1.grid(color = 'gray', axis='y')
        ax1.set_axisbelow(True)
        ax1.tick_params(axis = 'y', labelsize = 'large')
        ax1.set_ylim(0,y_max)

        ax2.bar(positions_theory_f_max, probabilities_theory_f_max_cold, width = 1, color = cold_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
        ax2.bar(positions_exp_f_max, probabilities_exp_f_max_cold, yerr = std_f_max_cold, width = 1, color = exp_cold_color, hatch = exp_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)

        ax2.set_xticks(ticks_f_max)
        ax2.set_xticklabels(labels_f_max, fontsize = 'x-large')
        ax2.grid(color = 'gray', axis='y')
        ax2.set_axisbelow(True)
        ax2.tick_params(axis = 'y', labelsize = 'large')
        ax2.set_ylim(0,y_max)

        if SE_theory_data is not None:
            ax1.bar(positions_theory_f_max, SE_theory_data['hpf'], width = 1, facecolor = SE_hot_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
            ax2.bar(positions_theory_f_max, SE_theory_data['cold_higher'], width = 1, color = SE_cold_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)

        ax1.set_ylabel('trap-enhanced probability per collision', fontsize = 'xx-large')

        ### Create legend
        # mpl.rcParams['hatch.linewidth'] = 2.0
        labels_and_colors = { 'hyperfine relaxation': exp_hot_color, 'cold spin change': exp_cold_color } 
        labels_and_hatch = { 'coupled-channel\nscattering calculations': '', 'experiment': '////' }
        interlude = 'estimation from the\nmatrix elements:'
        labels_and_lines = { 'normalized\nto $p_\\mathrm{hot}(\\left|\\right.$'+str(int(2))+', '+str(int(-2))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('k', 'D', 8), 'normalized\nto $p_\\mathrm{cold}(\\left|\\right.$'+str(int(2))+', '+str(int(-2))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('magenta', 'x', 10), 'normalized\nto $p_\\mathrm{cold}(\\left|\\right.$'+str(int(1))+', '+str(int(-1))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('orange', '2', 16) }
        handles_colors = [ plt.Rectangle((0,0), 1, 1, facecolor = labels_and_colors[color_label], edgecolor = 'k', hatch = '' ) for color_label in labels_and_colors.keys() ]
        handles_hatch = [ plt.Rectangle((0,0), 1, 1, facecolor = 'white', edgecolor = 'k', hatch = nhatch ) for nhatch in labels_and_hatch.values() ]
        handles_interlude = [ plt.Rectangle((0,0), 1, 1, facecolor = 'white', edgecolor = 'white', hatch = '' ) ]
        handles_lines = [ lines.Line2D([0], [0], color = labels_and_lines[line_label][0], linewidth = 3, marker = labels_and_lines[line_label][1], markersize = labels_and_lines[line_label][2]) for line_label in labels_and_lines.keys() ]


        colors_and_hatches = [ *[ (exp_hot_color, exp_hot_color, ''), (exp_cold_color, exp_cold_color, '') ],
                    *[('white', 'white', hatch) for hatch in labels_and_hatch.values()],
                     *[('white', 'white', '') for hh in handles_interlude]
                    ]
        if SE_theory_data is not None:
            labels_and_colors = { 'hyperfine relaxation\n(w/o & with SO coupling)': exp_hot_color,
                                  'cold spin change\n(w/o & with SO coupling)': exp_cold_color}
            colors_and_hatches = [ *[ (SE_hot_color, hot_color, ''), (SE_cold_color, cold_color, '') ],
                    *[('white', 'white', hatch) for hatch in labels_and_hatch.values()],
                     *[('white', 'white', '') for hh in handles_interlude]
                    ]
            


        labels = [ *list(labels_and_colors.keys()), *list(labels_and_hatch.keys()),]# interlude, *list(labels_and_lines.keys()) ]
        handles = [ *handles_colors, *handles_hatch, *handles_interlude,]# *handles_lines ]

        
        hmap = dict(zip(handles, [BicolorHandler(*color) for color in colors_and_hatches] ))
        ax2.legend(handles, labels, handler_map = hmap, loc = 'upper right', bbox_to_anchor = (1, 1), fontsize = 'x-large', labelspacing = 1)


        plt.tight_layout()
        
        return fig, ax1, ax2
    
    @staticmethod
    def prepareDataFromFiles(theory_hpf, theory_cold_higher, theory_cold_lower, exp_hpf, exp_cold_higher, exp_cold_lower):

        paths = locals().copy()

        theory_data = {
            'hpf': np.loadtxt(theory_hpf),
            'cold_higher': np.loadtxt(theory_cold_higher),
            'cold_lower': np.loadtxt(theory_cold_lower),
        }

        exp_data = {
            'hpf': np.loadtxt(exp_hpf)[0,:],
            'cold_higher': np.loadtxt(exp_cold_higher)[0,:],
            'cold_lower': np.loadtxt(exp_cold_lower)[0,:],
        }

        std_data = {
            'hpf': np.loadtxt(exp_hpf)[1,:],
            'cold_higher': np.loadtxt(exp_cold_higher)[1,:],
            'cold_lower': np.loadtxt(exp_cold_lower)[1,:],
        }


        return theory_data, exp_data, std_data

    @staticmethod
    def compareWithMatrixElements(fig, ax1, ax2, ax3, legend_ax, theory_data, SE_theory_data = None, pmf_array = None):

        if pmf_array is None:
            pmf_array = np.array([[1.0, 1.0]])
        
        f_max = 2.
        f_min = f_max-1
        f_ion = 0.5

        positions_theory_f_max = np.array([3*k+1 for k in np.arange(2*f_max+1)])
        positions_theory_f_min = np.array([3*k+1 for k in np.arange(2*f_min+1)])

        ### Prepare the data
        from_matrix_elements_f_max_hot = effective_probability(p0(np.array([theory_data['hpf'][0]*probabilities_from_matrix_elements_hot(f = f_max, mf = k, ms = f_ion)/probabilities_from_matrix_elements_hot(f = f_max, mf = -f_max, ms = f_ion) for k in np.arange(-f_max, f_max + 1)]), pmf_array = pmf_array), pmf_array = pmf_array)
        from_matrix_elements_f_max_cold = effective_probability(p0(np.array([theory_data['hpf'][0]*probabilities_from_matrix_elements_cold(f = f_max, mf = k, ms = f_ion)/probabilities_from_matrix_elements_hot(f = f_max, mf = -f_max, ms = f_ion) for k in np.arange(-f_max, f_max + 1)]), pmf_array = pmf_array), pmf_array = pmf_array)
        from_matrix_elements_f_max_cold_grouped = effective_probability(p0(np.array([theory_data['cold_higher'][0]*probabilities_from_matrix_elements_cold(f = f_max, mf = k, ms = f_ion)/probabilities_from_matrix_elements_cold(f = f_max, mf = -f_max, ms = f_ion) for k in np.arange(-f_max, f_max + 1)]), pmf_array = pmf_array), pmf_array = pmf_array)
        from_matrix_elements_f_min_cold = effective_probability(p0(np.array([theory_data['hpf'][0]*probabilities_from_matrix_elements_cold(f = f_min, mf = k, ms = f_ion)/probabilities_from_matrix_elements_hot(f = f_max, mf = -f_max, ms = f_ion) for k in np.arange(-f_min, f_min + 1)]), pmf_array = pmf_array), pmf_array = pmf_array)
        from_matrix_elements_f_min_cold_grouped = effective_probability(p0(np.array([theory_data['cold_lower'][0]*probabilities_from_matrix_elements_cold(f = f_min, mf = k, ms = f_ion)/probabilities_from_matrix_elements_cold(f = f_min, mf = -f_min, ms = f_ion) for k in np.arange(-f_min, f_min + 1)]), pmf_array = pmf_array), pmf_array = pmf_array)

        ax1.bar(positions_theory_f_max, np.zeros(np.shape(positions_theory_f_max)), bottom = from_matrix_elements_f_max_hot, width = 1, edgecolor = 'k', linewidth = 3)
        ax1.scatter(positions_theory_f_max, from_matrix_elements_f_max_hot, marker = 'D', s = 60, c = 'k')

        ax2.bar(positions_theory_f_max, np.zeros(np.shape(positions_theory_f_max)), bottom = from_matrix_elements_f_max_cold, width = 1, edgecolor = 'k', linewidth = 3)
        ax2.bar(positions_theory_f_max, np.zeros(np.shape(positions_theory_f_max)), bottom = from_matrix_elements_f_max_cold_grouped, width = 1, edgecolor = 'magenta', linewidth = 3 )
        ax2.scatter(positions_theory_f_max, from_matrix_elements_f_max_cold, marker = 'D', s = 50, c = 'k')
        ax2.scatter(positions_theory_f_max, from_matrix_elements_f_max_cold_grouped, marker = 'x', s = 80, c = 'magenta')

        ax3.bar(positions_theory_f_min, np.zeros(np.shape(positions_theory_f_min)), bottom = from_matrix_elements_f_min_cold, width = 1, edgecolor = 'k', linewidth = 3)
        ax3.bar(positions_theory_f_min, np.zeros(np.shape(positions_theory_f_min)), bottom = from_matrix_elements_f_min_cold_grouped, width = 1, edgecolor = 'orange', linewidth = 3)
        ax3.scatter(positions_theory_f_min, from_matrix_elements_f_min_cold, marker = 'D', s = 50, c = 'k')
        ax3.scatter(positions_theory_f_min, from_matrix_elements_f_min_cold_grouped, marker = '2', s = 200, c = 'orange')

        ### Recreate legend
        cold_color, hot_color = 'midnightblue', 'firebrick'
        exp_cold_color, exp_hot_color = 'midnightblue', 'firebrick'
        exp_hatch, theory_hatch = '////', ''

        if SE_theory_data is not None:
            SE_cold_color, SE_hot_color = 'midnightblue', 'firebrick'
            cold_color, hot_color = 'royalblue', 'indianred'

        labels_and_colors = { 'hyperfine relaxation': hot_color, 'cold spin change': cold_color } 
        labels_and_hatch = { 'coupled-channel\nscattering calculations': '', 'experiment': '////' }
        interlude = 'estimation from the\nmatrix elements:'
        labels_and_lines = { 'normalized\nto $p_\\mathrm{hot}(\\left|\\right.$'+str(int(2))+', '+str(int(-2))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('k', 'D', 8), 'normalized\nto $p_\\mathrm{cold}(\\left|\\right.$'+str(int(2))+', '+str(int(-2))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('magenta', 'x', 10), 'normalized\nto $p_\\mathrm{cold}(\\left|\\right.$'+str(int(1))+', '+str(int(-1))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('orange', '2', 16) }
        handles_colors = [ plt.Rectangle((0,0), 1, 1, facecolor = labels_and_colors[color_label], edgecolor = 'k', hatch = '' ) for color_label in labels_and_colors.keys() ]
        handles_hatch = [ plt.Rectangle((0,0), 1, 1, facecolor = 'white', edgecolor = 'k', hatch = labels_and_hatch[hatch_label] ) for hatch_label in labels_and_hatch.keys() ]
        handles_interlude = [ plt.Rectangle((0,0), 1, 1, facecolor = 'white', edgecolor = 'white', hatch = '' ) ]
        handles_lines = [ lines.Line2D([0], [0], color = labels_and_lines[line_label][0], linewidth = 3, marker = labels_and_lines[line_label][1], markersize = labels_and_lines[line_label][2]) for line_label in labels_and_lines.keys() ]

        colors = [ *[ (exp_hot_color, exp_hot_color), (exp_cold_color, exp_cold_color) ],
                   *[('white', 'white') for hh in [*handles_hatch, handles_interlude] ]
                   ]
        if SE_theory_data is not None:
            labels_and_colors = { 'hyperfine relaxation\n(w/o & with $\\lambda_\\mathrm{so}$)': exp_hot_color,
                                  'cold spin change\n(w/o & with $\\lambda_\\mathrm{so}$)': exp_cold_color}
            colors = [ *[ (SE_hot_color, hot_color), (SE_cold_color, cold_color) ],
                    *[('white', 'white') for hh in [*handles_hatch, handles_interlude] ]
                    ]

        labels = [ *list(labels_and_colors.keys()), *list(labels_and_hatch.keys()), interlude, *list(labels_and_lines.keys()) ]
        handles = [ *handles_colors, *handles_hatch, *handles_interlude, *handles_lines ]

        hmap = dict(zip(handles, [BicolorHandler(*color) for color in colors] ))
        legend_ax.legend(handles, labels, handler_map = hmap, loc = 'upper center', bbox_to_anchor = (8/16, 1.00), fontsize = 'large', labelspacing = 0.7)
        
        return fig, ax1, ax2, ax3

    @staticmethod
    def addParams(fig, legend_ax, singlet_phase, triplet_phase, so_scaling):
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstring = (f'($\\tilde{{\\Phi}}_\mathrm{{s}}, \\tilde{{\\Phi}}_\mathrm{{t}}$) = ({singlet_phase}, {triplet_phase})\n'
                        f'$c_\\mathrm{{so}} = {so_scaling}$')


        legend = legend_ax.get_legend()
        legend_extent = legend.get_window_extent()
        legend_ax_extent = legend_ax.get_window_extent()
        fig_extent = fig.get_window_extent()

        param_frame_ax = fig.add_axes([legend_extent.x0/fig_extent.width,
                    legend_ax_extent.y0/fig_extent.height,
                    legend_extent.width/fig_extent.width,
                    (legend_extent.y0-legend_ax_extent.y0)/fig_extent.height if (legend_extent.y0-legend_ax_extent.y0)/fig_extent.height > 0 else 0])

        param_frame_ax.text(0.5, 0.5, textstring, fontsize = 'x-large', va = 'center', ha = 'center', bbox = props)
        param_frame_ax.axis('off')

        return param_frame_ax


class ProbabilityVersusSpinOrbit:
    """Plot of the calculated probability as a function of the spin-orbit coupling parameter."""

    p_eff_exp = 0.0895
    p_eff_exp_std = 0.0242

    @staticmethod
    def _initiate_plot(figsize = (6.4,4.8), dpi = 300):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)        
        return fig, ax
    
    @classmethod
    def plotBareProbability(cls, so_parameter, probability, p0_exp = None, p0_exp_std = None, relative = False, figsize = (6.4,4.8), dpi = 300):
        
        pmf_path = Path(__file__).parents[2] / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
        pmf_array = np.loadtxt(pmf_path)

        if p0_exp is None:
            p_eff_exp = cls.p_eff_exp
            p0_exp = p0(cls.p_eff_exp, pmf_array=pmf_array)
        if p0_exp_std is None:
            dpeff = 1e-3
            p0_exp_std = (p0(p_eff_exp+dpeff/2, pmf_array=pmf_array)-p0(p_eff_exp-dpeff/2, pmf_array=pmf_array))/dpeff * cls.p_eff_exp_std

        so_parameter = np.asarray(so_parameter)
        probability = np.asarray(probability)
        nearest_idx = np.abs(probability - p0_exp).argmin()

        data_label = '$p_0(c_\mathrm{so})$'
        fit_label = '$p_0 \sim c_\mathrm{so}^2$'
        plot_title = 'Probability of the hyperfine energy release for $\left|2,2\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state'
        
        if relative:
            data_label = '$p_0(c_\mathrm{so}) / p_0(c_\mathrm{so} = 1)$'
            plot_title = 'Probability of the hyperfine energy release for $\left|2,2\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state\n(relative to the result for the original spin-orbit coupling from M.T.)'
            probability = probability / max(probability)
        
        xx = np.logspace(-1, 1, 100)

        fig, ax = cls._initiate_plot(figsize, dpi)

        ax.scatter(so_parameter, probability, s = 6**2, color = 'k', marker = 'o', label = data_label)
        ax.plot(so_parameter[nearest_idx]*xx, probability[nearest_idx] * xx**2, color = 'red', linewidth = 1.5, linestyle = '--', label = fit_label)

        ax.axhline(p0_exp, color = 'k', linewidth = 1.5, linestyle = '--', label = 'experimental value')
        ax.axhspan(p0_exp-p0_exp_std, p0_exp+p0_exp_std, color='0.5', alpha=0.5)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='both', direction='in', labelsize = 12)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.xaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.2f}'))

        ax.set_ylim(0.25*p0_exp, 2*p0_exp)
        ax.set_xlim(0.5*so_parameter[nearest_idx], 1.5*so_parameter[nearest_idx])
        ax.set_xlabel('$c_\mathrm{so}$', fontsize = 'xx-large')
        ax.set_ylabel('$p_0$', fontsize = 'xx-large')
        ax.set_title(plot_title, fontsize = 10)
        ax.legend()
        
        plt.tight_layout()

        return fig, ax

    @classmethod
    def plotEffectiveProbability(cls, so_parameter, probability, p_eff_exp = None, p_eff_exp_std = None, pmf_array = None, figsize = (6.4,4.8), dpi = 300):
        
        if pmf_array is None:
            pmf_path = Path(__file__).parents[1] / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
            pmf_array = np.loadtxt(pmf_path)
        
        if p_eff_exp is None:
            p_eff_exp = cls.p_eff_exp
        if p_eff_exp_std is None:
            p_eff_exp_std = cls.p_eff_exp_std

        so_parameter = np.asarray(so_parameter)
        probability = np.asarray(probability)
        nearest_idx = np.abs(probability - p_eff_exp).argmin()

        data_label = '$p_\mathrm{eff}(c_\mathrm{so})$'
        fit_label = '$p_\mathrm{eff}$ for $p_0 \sim c_\mathrm{so}^2$'
        plot_title = 'Effective probability of the hyperfine energy release for $\left|2,2\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state'
        
        xx = np.logspace(-2, 0.5*np.log10(0.99/probability[nearest_idx]), 100)
        # print( so_parameter[nearest_idx], p0(probability[nearest_idx], pmf_array = pmf_array) )
        fig, ax = cls._initiate_plot(figsize, dpi)

        ax.scatter(so_parameter, probability, s = 6**2, color = 'k', marker = 'o', label = data_label)
        # print(xx)
        ax.plot(so_parameter[nearest_idx]*xx, effective_probability(p0(probability[nearest_idx], pmf_array = pmf_array) * xx**2, pmf_array = pmf_array), color = 'red', linewidth = 1.5, linestyle = '--', label = fit_label)
        # print(effective_probability(p0(probability[nearest_idx], pmf_array = pmf_array) * xx**2, pmf_array = pmf_array) )

        ax.axhline(p_eff_exp, color = 'k', linewidth = 1.5, linestyle = '--', label = 'experimental value')
        ax.axhspan(p_eff_exp-p_eff_exp_std, p_eff_exp+p_eff_exp_std, color='0.5', alpha=0.5)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='both', direction='in', labelsize = 12)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.xaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.2f}'))

        ax.set_ylim(0.25*p_eff_exp, 2*p_eff_exp)
        ax.set_xlim(0.5*so_parameter[nearest_idx], 1.5*so_parameter[nearest_idx])
        ax.set_xlabel('$c_\mathrm{so}$', fontsize = 'xx-large')
        ax.set_ylabel('$p_\mathrm{eff}$', fontsize = 'xx-large')
        ax.set_title(plot_title, fontsize = 10)
        ax.legend()
        
        plt.tight_layout()

        return fig, ax


class PartialRateVsEnergy:
    """Plot of the partial and total collision rates as a function of the collision energy."""

    def _initiate_plot(figsize = (8, 6), dpi = 100):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)        
        return fig, ax
    
    @classmethod
    def plotRate(cls, energy, rate, figsize = (9.5, 7.2), dpi = 100):
        
        rate = np.array(rate)
        l_max = rate.shape[0] - 1


        fig, ax = cls._initiate_plot(figsize, dpi)

        for l in range(l_max+1):
            ax.plot(energy, rate[l], linewidth = 1.5, linestyle = 'solid', marker = '.', markersize = 1, color = mpl.colormaps['cividis'](l/30) )
        
        ax.set_xlim(np.min(energy), np.max(energy))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis = 'both', labelsize= 'x-large')
        ax.grid(color = 'gray')

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.1 inch.
        divider = make_axes_locatable(ax)
        ax_bar = divider.append_axes("right", size="3%", pad=0.1)
        cb = mpl.colorbar.ColorbarBase(ax_bar, cmap='cividis', norm = mpl.colors.Normalize(0, l_max+1), ticks = [0, 10, 20, 30])
        ax_bar.tick_params(axis = 'both', labelsize = 'x-large')
        ax_bar.get_yaxis().labelpad = 4
        ax_bar.set_ylabel('L', rotation = 0, fontsize = 'xx-large')

        return fig, ax
    
    
class RateVsMagneticField:
    """Plot of the thermally averaged collision rate as a function of the magnetic field."""

    def _initiate_plot(figsize = (9.5, 7.2), dpi = 300):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)        
        return fig, ax
    
    @classmethod
    def plotRate(cls, magnetic_field, rate, figsize = (9.5, 7.2), dpi = 300):
        rate = np.array(rate)

        fig, ax = cls._initiate_plot(figsize, dpi)

        ax.plot(magnetic_field, rate, linewidth = 2, linestyle = 'solid', marker = '.', markersize = 2, color = 'midnightblue')
        ax.set_xlim(np.min(magnetic_field), np.max(magnetic_field))
        ax.tick_params(axis = 'both', labelsize= 'x-large')
        ax.grid(color = 'gray')

        return fig, ax
    

class ValuesVsModelParameters:
    """Plot of the theoretical results and chi-squared as a function of a given parameter together with the experimental ones."""

    def _initiate_plot(figsize = (9.5, 7.2), dpi = 300):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax_chisq = ax.twinx()
        return fig, ax, ax_chisq
    
    @classmethod
    def plotValuesAndChiSquared(cls, xx, theory, experiment, std, theory_distinguished = None, figsize = (9.5, 7.2), dpi = 300):
        chi_sq = chi_squared(theory, experiment, std)

        theory_colors = ['darksalmon', 'lightsteelblue', 'moccasin']
        theory_distinguished_colors = ['firebrick', 'midnightblue', 'darkorange']

        # theory_distinguished_mask = np.isfinite(theory_distinguished)
        # theory_mask = np.isfinite(theory)

        fig, ax, ax_chisq = cls._initiate_plot(figsize, dpi)

        for i, yy in enumerate(np.moveaxis(theory, -1, 0)):
            yy = yy.transpose()
            yy_mask = np.isfinite(yy)
            print(f'{xx.shape=}, {xx[yy_mask].shape=}, {yy.shape=}, {yy[yy_mask].shape=}')
            ax.plot(xx[yy_mask].reshape(-1, xx.shape[-1]), yy[yy_mask].reshape(-1, yy.shape[-1]), color = theory_colors[i], linewidth = .1)
            ax.axhspan(experiment[i]-std[i], experiment[i]+std[i], color = theory_colors[i], alpha=0.5)
            ax.axhline(experiment[i], color = theory_distinguished_colors[i], linestyle = '--', linewidth = 4)

        chi_sq = chi_sq.transpose()
        chi_sq_mask = np.isfinite(chi_sq)
        ax_chisq.plot(xx[chi_sq_mask].reshape(-1, xx.shape[-1]), chi_sq[chi_sq_mask].reshape(-1, chi_sq.shape[-1]), color = '0.7', linewidth = 0.1)

        if theory_distinguished is not None:
            
            chi_sq_distinguished = chi_squared(theory_distinguished, experiment, std)
            
            for i, yy in enumerate(np.moveaxis(theory_distinguished, -1, 0)):
                yy = yy.transpose()
                yy_mask = np.isfinite(yy)
                ax.plot(xx[tuple(map(slice, yy.shape))][yy_mask], yy[yy_mask], color = theory_distinguished_colors[i], linewidth = 4)

            chi_sq_distinguished = chi_sq_distinguished.transpose()
            chi_sq_distinguished_mask = np.isfinite(chi_sq_distinguished)
            ax_chisq.plot(xx[tuple(map(slice, chi_sq_distinguished.shape))][chi_sq_distinguished_mask], chi_sq_distinguished[chi_sq_distinguished_mask], 'k--', linewidth = 4, label = '$\chi^2$')
        
        ax.set_xlim(np.min(xx), np.max(xx))

        ax.tick_params(which='both', direction='in', top = True, labelsize = 30, length = 10)
        ax.tick_params(which='minor', length = 5)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.xaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.2f}'))

        ax_chisq.set_yscale('log')
        ax_chisq.tick_params(which='both', direction='in', labelsize = 30, length = 10)
        ax_chisq.tick_params(which='minor', length = 5)
        
        return fig, ax, ax_chisq
    
    @classmethod
    def plotPeffAndChiSquaredVsDPhi(cls, xx, theory, experiment, std, theory_distinguished = None, figsize = (9.5, 7.2), dpi = 300):
        fig, ax, ax_chisq = cls.plotValuesAndChiSquared(xx, theory, experiment, std, theory_distinguished, figsize, dpi)
        
        ax.set_xlim(0,1)
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val,pos: '0' if val == 0 else f'$\\pi$' if val == 1. else f'$-\\pi$' if val == -1. else f'${val}\\pi$' if val % 1 == 0 else f'$\\frac{{{val*2:.0g}}}{{2}}\\pi$' if (val *2)  % 1 == 0 else f'$\\frac{{{val*4:.0g}}}{{4}}\\pi$' if (val*4) % 1 == 0 else f'${val:.2g}\\pi$'
        ))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1/4))
        ax.xaxis.set_minor_formatter('')
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=0.05))

        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
        ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter(''))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.1))

        ax.set_xlabel(f'Singlet-triplet phase difference', fontsize = 36)
        ax.set_ylabel(f'Effective probability', fontsize = 36)
        ax_chisq.set_ylabel(f'$\\chi^2$', fontsize = 36, rotation = 0, labelpad = 20)
        
        
        fig.tight_layout()

        return fig, ax, ax_chisq