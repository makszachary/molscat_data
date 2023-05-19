from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import lines
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .effective_probability import effective_probability, p0
from .analytical import probabilities_from_matrix_elements_cold, probabilities_from_matrix_elements_hot

class BicolorHandler:
    def __init__(self, color1, color2):
        self.color1=color1
        self.color2=color2
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = plt.Rectangle([x0, y0], width, height, facecolor='none',
                                   edgecolor='k', transform=handlebox.get_transform())
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
            


        labels = [ *list(labels_and_colors.keys()), *list(labels_and_hatch.keys()),]# interlude, *list(labels_and_lines.keys()) ]
        handles = [ *handles_colors, *handles_hatch, *handles_interlude,]# *handles_lines ]

        
        hmap = dict(zip(handles, [BicolorHandler(*color) for color in colors] ))
        legend_ax.legend(handles, labels, handler_map = hmap, loc = 'center', bbox_to_anchor = (8/16, 0.50), fontsize = 'large', labelspacing = 1)

        plt.tight_layout()
        
        return fig, ax1, ax2, ax3, legend_ax
    
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
        labels_and_hatch = { 'coupled-channel\ncalculations': '', 'experiment': '////' }
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

    @staticmethod
    def _initiate_plot(figsize = (6.4,4.8), dpi = 100):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)        
        return fig, ax
    
    @classmethod
    def plotBareProbability(cls, so_parameter, probability, relative = False, figsize = (6.4,4.8), dpi = 100):
        
        data_label = '$p_0(c_\mathrm{so})$'
        plot_title = 'Probability of the hyperfine energy release for $\left|2,2\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state'
        
        if relative:
            data_label = '$p_0(c_\mathrm{so}) / p_0(c_\mathrm{so} = 1)$'
            plot_title = 'Probability of the hyperfine energy release for $\left|2,2\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state\n(relative to the result for the original spin-orbit coupling from M.T.)'
            probability = probability / max(probability)
        
        xx = np.logspace(-4, 1, 100)

        fig, ax = cls._initiate_plot(figsize, dpi)

        ax.scatter(so_parameter, probability, s = 4**2, color = 'k', marker = 'o', label = data_label)
        ax.plot(xx, max(probability) * xx**2, color = 'red', linewidth = 1, linestyle = '--', label = '$p_0 \sim c_\mathrm{so}^2$')
        ax.plot(xx, max(probability) * xx**1.8, color = 'orange', linewidth = 1, linestyle = '--', label = '$p_0 \sim c_\mathrm{so}^{1.8}$')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(0.1*min(probability), 10*max(probability))
        ax.set_xlim(0.1*min(so_parameter), 10*max(so_parameter))
        ax.set_xlabel('spin-orbit coupling scaling factor $c_\mathrm{so}$', fontsize = 'large')
        ax.set_ylabel('$p_0$', fontsize = 'large')
        ax.set_title(plot_title, fontsize = 10)
        ax.legend()
        
        plt.tight_layout()

        return fig, ax

    @classmethod
    def plotEffectiveProbability(cls, so_parameter, probability, pmf_array = None, figsize = (6.4,4.8), dpi = 100):
        
        data_label = '$p_\mathrm{eff}(c_\mathrm{so})$'
        plot_title = 'Effective probability of the hyperfine energy release for $\left|2,2\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state'
        
        if pmf_array is None:
            pmf_path = Path(__file__).parents[2] / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
            pmf_array = np.loadtxt(pmf_path)
        
        xx = np.logspace(-4, 0, 100)

        fig, ax = cls._initiate_plot(figsize, dpi)

        ax.scatter(so_parameter, probability, s = 4**2, color = 'k', marker = 'o', label = data_label)
        ax.plot(xx, effective_probability(p0(max(probability), pmf_array = pmf_array) * xx**2, pmf_array = pmf_array), color = 'red', linewidth = 1, linestyle = '--', label = '$p_\mathrm{eff}$ for $p_0 \sim c_\mathrm{so}^2$')
        ax.plot(xx, effective_probability(p0(max(probability), pmf_array = pmf_array) * xx**1.8, pmf_array = pmf_array), color = 'orange', linewidth = 1, linestyle = '--', label = '$p_\mathrm{eff}$ for $p_0 \sim c_\mathrm{so}^{1.8}$')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(0.1*min(probability), 10*max(probability))
        ax.set_xlim(0.1*min(so_parameter), 10*max(so_parameter))
        ax.set_xlabel('spin-orbit coupling scaling factor $c_\mathrm{so}$', fontsize = 'large')
        ax.set_ylabel('$p_\mathrm{eff}$', fontsize = 'large')
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
    
    