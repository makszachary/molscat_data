import numpy as np

from matplotlib import pyplot as plt
from matplotlib import lines

class Barplot_Wide:
    """Wide bar plot for the single-ion RbSr+ experiment."""

    @staticmethod
    def _initiate_plot(figsize = (16,5), dpi = 100):
        fig= plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(1, 16, (1,5))
        ax2 = fig.add_subplot(1, 16, (6,10))
        ax3 = fig.add_subplot(1, 16, (11,13))
        
        return fig, ax1, ax2, ax3
    
    @staticmethod
    def barplot(theory_data, exp_data, std_data, figsize = (16,5), dpi = 100):
        """barplot
        
        :param theory_data: dictionary of the form {'hpf': (p(-2), p(-1), ..., p(2)),
        'cold_higher': (p(-2), p(-1), ..., p(2)), 'cold_lower': (p(-1), p(0), p(1)) }
        :param exp_data: dictionary of the form {'hpf': (p(-2), p(-1), ..., p(2)),
        'cold_higher': (p(-2), p(-1), ..., p(2)), 'cold_lower': (p(-1), p(0), p(1)) }
        """
        # print(theory_data)
        # print(exp_data)
        # print(std_data)
        fig, ax1, ax2, ax3 = Barplot_Wide._initiate_plot(figsize, dpi)

        probabilities_theory_f_max_hot = theory_data['hpf']
        probabilities_theory_f_max_cold = theory_data['cold_higher']
        probabilities_theory_f_min_cold = theory_data['cold_lower']
        
        probabilities_exp_f_max_hot = exp_data['hpf']
        probabilities_exp_f_max_cold = exp_data['cold_higher']
        probabilities_exp_f_min_cold = exp_data['cold_lower']

        std_f_max_hot = std_data.get('hpf', np.full_like(probabilities_exp_f_max_hot, 0))
        # print(std_f_max_hot)
        std_f_max_cold = std_data.get('cold_higher', np.full_like(probabilities_exp_f_max_cold, 0))
        std_f_min_cold = std_data.get('cold_lower', np.full_like(probabilities_exp_f_min_cold, 0))

        cold_color, hot_color = 'midnightblue', 'firebrick'
        exp_hatch, theory_hatch = '////', ''
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
        ax1.bar(positions_exp_f_max, probabilities_exp_f_max_hot, yerr = std_f_max_hot, width = 1, color = hot_color, hatch = exp_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
        # ax1.bar(positions_theory_f_max, np.zeros(np.shape(positions_theory_f_max)), bottom = classical_f_max_hot, width = 1, edgecolor = 'k', linewidth = 3)
        # ax1.scatter(positions_theory_f_max, classical_f_max_hot, marker = 'D', s = 60, c = 'k')
        ax1.set_xticks(ticks_f_max)
        ax1.set_xticklabels(labels_f_max, fontsize = 'x-large')
        ax1.grid(color = 'gray', axis='y')
        ax1.set_axisbelow(True)
        ax1.tick_params(axis = 'y', labelsize = 'large')
        ax1.set_ylim(0,y_max)

        ax2.bar(positions_theory_f_max, probabilities_theory_f_max_cold, width = 1, color = cold_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
        ax2.bar(positions_exp_f_max, probabilities_exp_f_max_cold, yerr = std_f_max_cold, width = 1, color = cold_color, hatch = exp_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
        
        # ax2.bar(positions_theory_f_max, np.zeros(np.shape(positions_theory_f_max)), bottom = classical_f_max_cold, width = 1, edgecolor = 'k', linewidth = 3)
        # ax2.bar(positions_theory_f_max, np.zeros(np.shape(positions_theory_f_max)), bottom = classical_f_max_cold_grouped, width = 1, edgecolor = 'magenta', linewidth = 3 )
        # ax2.scatter(positions_theory_f_max, classical_f_max_cold, marker = 'D', s = 50, c = 'k')
        # ax2.scatter(positions_theory_f_max, classical_f_max_cold_grouped, marker = 'x', s = 80, c = 'magenta')
        ax2.set_xticks(ticks_f_max)
        ax2.set_xticklabels(labels_f_max, fontsize = 'x-large')
        ax2.grid(color = 'gray', axis='y')
        ax2.set_axisbelow(True)
        ax2.tick_params(axis = 'y', labelsize = 'large')
        ax2.set_ylim(0,y_max)

        ax3.bar(positions_theory_f_min, probabilities_theory_f_min_cold, width = 1, color = cold_color, hatch = theory_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
        print(probabilities_exp_f_min_cold)
        
        ax3.bar(positions_exp_f_min, probabilities_exp_f_min_cold, yerr = std_f_min_cold, width = 1, color = cold_color, hatch = exp_hatch, edgecolor = 'black', alpha = 0.9, ecolor = 'black', capsize = 5)
        
        # ax3.bar(positions_theory_f_min, np.zeros(np.shape(positions_theory_f_min)), bottom = classical_f_min_cold, width = 1, edgecolor = 'k', linewidth = 3)
        # ax3.bar(positions_theory_f_min, np.zeros(np.shape(positions_theory_f_min)), bottom = classical_f_min_cold_grouped, width = 1, edgecolor = 'orange', linewidth = 3)
        # ax3.scatter(positions_theory_f_min, classical_f_min_cold, marker = 'D', s = 50, c = 'k')
        # ax3.scatter(positions_theory_f_min, classical_f_min_cold_grouped, marker = '2', s = 200, c = 'orange')
        # print(classical_f_min_cold_grouped)
        ax3.set_xticks(ticks_f_min)
        ax3.set_xticklabels(labels_f_min, fontsize = 'x-large')
        ax3.grid(color = 'gray', axis='y')
        ax3.set_axisbelow(True)
        ax3.tick_params(axis = 'y', labelsize = 'large')
        ax3.set_ylim(0,y_max)

        ### Create legend

        labels_and_colors = { 'hyperfine relaxation': hot_color, 'cold spin change': cold_color } 
        labels_and_hatch = { 'coupled-channel\ncalculations': '', 'experiment': '////' }
        interlude = 'estimation from the\nmatrix elements:'
        labels_and_lines = { 'normalized\nto $p_\\mathrm{hot}(\\left|\\right.$'+str(int(2))+', '+str(int(-2))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('k', 'D', 8), 'normalized\nto $p_\\mathrm{cold}(\\left|\\right.$'+str(int(2))+', '+str(int(-2))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('magenta', 'x', 10), 'normalized\nto $p_\\mathrm{cold}(\\left|\\right.$'+str(int(1))+', '+str(int(-1))+', '+'$\\left.\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>)$': ('orange', '2', 16) }
        handles_colors = [ plt.Rectangle((0,0), 1, 1, facecolor = labels_and_colors[color_label], edgecolor = 'k', hatch = '' ) for color_label in labels_and_colors.keys() ]
        handles_hatch = [ plt.Rectangle((0,0), 1, 1, facecolor = 'white', edgecolor = 'k', hatch = labels_and_hatch[hatch_label] ) for hatch_label in labels_and_hatch.keys() ]
        handles_interlude = [ plt.Rectangle((0,0), 1, 1, facecolor = 'white', edgecolor = 'white', hatch = '' ) ]
        handles_lines = [ lines.Line2D([0], [0], color = labels_and_lines[line_label][0], linewidth = 3, marker = labels_and_lines[line_label][1], markersize = labels_and_lines[line_label][2]) for line_label in labels_and_lines.keys() ]
        # patches.Polygon(np.asarray(((0,0),(1,0))))
        
        labels = [ *list(labels_and_colors.keys()), *list(labels_and_hatch.keys()),]# interlude, *list(labels_and_lines.keys()) ]
        handles = [ *handles_colors, *handles_hatch, *handles_interlude,]# *handles_lines ]

        fig.legend(handles, labels, loc = 'center', bbox_to_anchor = (14.5/16, 0.50), fontsize = 'large', labelspacing = 1)
        # fig.legend(handles, labels, loc = 'upper center', bbox_to_anchor = (14.5/16, 0.97), fontsize = 'large', labelspacing = 1)

        plt.tight_layout()

        return fig, ax1, ax2, ax3
    
    @classmethod
    def barplot_from_files(cls, theory_hpf, theory_cold_higher, theory_cold_lower, exp_hpf, exp_cold_higher, exp_cold_lower, figsize = (16,5), dpi = 100):

        paths = locals().copy()
        paths.pop('figsize')
        paths.pop('dpi')

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

        fig, ax1, ax2, ax3 = cls.barplot(theory_data, exp_data, std_data, figsize, dpi)

        return fig, ax1, ax2, ax3