
'''
Riemann Solver GUI, Based on Riemann Solvers and Numerical Methods for Fluid Dynamic by Eleuterio.F.Toro.
Created by Etay Dabach as final project for course 77819-Computational Hydrodynamics taught by Elad Steinberg, 2024.

Able to solve the Riemann problem for the Euler equation by the exact solution and HLLC with Godunov scheme first and second approximations.
The exact solution is interactive and both of the approximations require calculations which takes few seconds to be plotted. 
It might take a few second for the GUI to upload due to pre-calculations of the approximations for the Riemann problem tests. (Disabled for now, no pre-calculations.)

To calculate approximations, use the custom parameters and press the button for the desired order. The exact solution will always be calculated in both orders and is controlled by the sliders positions.
The 'Apply' button is only for the exact solution, and 'Clear' button return the custom parameters values to default without changing the sliders position,
and therefore does not affect the graphs for the exact solution.
Use the rescale button to scale the axis of the visible graphs.
The Tests buttons have pre-calculated approximations for both first and second order, and are used to test the accuracy of the calculations given different scenarios that might occur in the tube.
Checkboxes are used to revel or hide the different methods, and for some other fetures in the plots.
'Reset All' button return everything to default. Does not affect checkboxes.

'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons, TextBox
from functions_for_shocktube import *


#==================================================================================================================================================#
#==================================================================Creating The GUI================================================================#
#==================================================================================================================================================#

def main():

    # Setting parameters
    initial_left_parameters = (1, 0, 1) # (rho, u, P)
    initial_right_parameters = (0.125, 0, 0.1)
    gamma = 1.4
    N_cells = 300
    t_final = 0.2
    x_divider = 0.5

    # Exact solution
    x_vec_ex, rho_vec_ex, u_vec_ex, P_vec_ex = exact_solution(initial_left_parameters, initial_right_parameters, t_final, gamma, N_cells, TOL=1e-9, x_center=x_divider)
    # Godunov first order with HLLC
    U_final_hllc, x_vec_1 = tube_time_evolution(initial_left_parameters, initial_right_parameters, N_cells, t_final, gamma, HLLC_Riemann_Solver, x_0=x_divider)
    rho_vec_1st, u_vec_1st, P_vec_1st = primitive_parameters_vectors(U_final_hllc, gamma)
    # Godunov second order with HLLC
    U_final_2nd_order, x_vec_2 = tube_time_evolution_2nd_order(initial_left_parameters, initial_right_parameters, t_final, N_cells, gamma, x_0=x_divider)
    rho_vec_2nd, u_vec_2nd, P_vec_2nd = primitive_parameters_vectors(U_final_2nd_order, gamma)


    plt.close('all')
    fig, axs = plt.subplots(3,1 ,figsize = (14,10), dpi=90, num='Sod Shock Tube GUI')
    fig.subplots_adjust(left=0.1, bottom=0.40) #[left, bottom, width, height]
    # Grid settings
    def set_global_variables():
        global grid_visibility
        grid_visibility = True
    set_global_variables()
    # grid_visibility = True # grid_visibility
    density_grid = axs[0].grid(visible=grid_visibility)
    velocity_grid = axs[1].grid(visible=grid_visibility)
    pressure_grid = axs[2].grid(visible=grid_visibility)

    # Exact solution
    graph_visibility0 = True
    density_curve, = axs[0].plot(x_vec_ex, rho_vec_ex, linestyle='--', label='exact', visible=graph_visibility0)
    velocity_curve, = axs[1].plot(x_vec_ex, u_vec_ex, linestyle='--', label='exact', visible=graph_visibility0)
    pressure_curve, = axs[2].plot(x_vec_ex, P_vec_ex, linestyle='--', label='exact', visible=graph_visibility0)

    # First order
    graph_visibility1 = True
    density_curve_1, = axs[0].plot(x_vec_1, rho_vec_1st, label='first order', visible=graph_visibility1)
    velocity_curve_1, = axs[1].plot(x_vec_1, u_vec_1st, label='first order', visible=graph_visibility1)
    pressure_curve_1, = axs[2].plot(x_vec_1, P_vec_1st, label='first order', visible=graph_visibility1)

    # Second order
    graph_visibility2 = True
    density_curve_2, = axs[0].plot(x_vec_2, rho_vec_2nd, label='second order', visible=graph_visibility2)
    velocity_curve_2, = axs[1].plot(x_vec_2, u_vec_2nd, label='second order', visible=graph_visibility2)
    pressure_curve_2, = axs[2].plot(x_vec_2, P_vec_2nd, label='second order', visible=graph_visibility2)

    # Plotting x0 on the graphs
    x0_visibilty = True
    x0_in_density = axs[0].axvline(x=x_divider, color='black', linestyle='--', visible=x0_visibilty, label=r'x0') # , label=r'x_{0}'
    x0_in_velocity = axs[1].axvline(x=x_divider, color='black', linestyle='--', visible=x0_visibilty)
    x0_in_pressure = axs[2].axvline(x=x_divider, color='black', linestyle='--', visible=x0_visibilty)

    axs[0].set_ylabel('Density')
    axs[1].set_ylabel('Velocity')
    axs[2].set_ylabel('Pressure')
    axs[2].set_xlabel('tube lenght')
    legend =  axs[0].legend()

    # Creating axes for sliders
    ax_rhoL = fig.add_axes([0.1, 0.25, 0.2, 0.01]) #[left, bottom, width, height]
    ax_uL = fig.add_axes([0.1, 0.2, 0.2, 0.01])
    ax_PL = fig.add_axes([0.1, 0.15, 0.2, 0.01])

    ax_rhoR = fig.add_axes([0.7, 0.25, 0.2, 0.01]) #[left, bottom, width, height]
    ax_uR = fig.add_axes([0.7, 0.2, 0.2, 0.01])
    ax_PR = fig.add_axes([0.7, 0.15, 0.2, 0.01])


    ax_t = fig.add_axes([0.1, 0.3, 0.8, 0.01])
    ax_gamma = fig.add_axes([0.4, 0.25, 0.2, 0.01])
    ax_cell = fig.add_axes([0.4, 0.2, 0.2, 0.01])
    ax_div = fig.add_axes([0.4, 0.15, 0.2, 0.01])


    # Creating the sliders
    rhoL_slider = Slider(ax=ax_rhoL, label=r'$\rho_{L}$', valmin= 0.01, valmax=2, valinit=initial_left_parameters[0])
    uL_slider = Slider(ax=ax_uL, label=r'$u_{L}$', valmin= -2, valmax=2, valinit=initial_left_parameters[1])
    PL_slider = Slider(ax=ax_PL, label=r'$P_{L}$', valmin= 0.05, valmax=2, valinit=initial_left_parameters[2])

    rhoR_slider = Slider(ax=ax_rhoR, label=r'$\rho_{R}$', valmin= 0.01, valmax=2, valinit=initial_right_parameters[0])
    uR_slider = Slider(ax=ax_uR, label=r'$u_{R}$', valmin= -2, valmax=2, valinit=initial_right_parameters[1])
    PR_slider = Slider(ax=ax_PR, label=r'$P_{R}$', valmin= 0.05, valmax=2, valinit=initial_right_parameters[2])

    gamma_slider = Slider(ax=ax_gamma, label=r'$\gamma$', valmin=1.05, valmax=2, valinit=gamma)
    t_slider = Slider(ax=ax_t, label=r'$t$', valmin=0.01, valmax=2, valinit=t_final)
    cell_slider = Slider(ax=ax_cell, label=r'$N_{cells}$', valmin=10, valmax=1000, valinit=N_cells, valfmt='%0.0f', valstep=2)
    div_slider = Slider(ax=ax_div, label=r'$x_{0}$', valmin=0.1, valmax=0.9, valinit=x_divider, valstep=0.1)


    def set_sliders_values(left_params, right_params, t, gamma, x0=0.5, N=300):
        rhoL, uL, PL = left_params 
        rhoR, uR, PR = right_params

        rhoL_slider.set_val(rhoL)
        uL_slider.set_val(uL)
        PL_slider.set_val(PL)

        rhoR_slider.set_val(rhoR)
        uR_slider.set_val(uR)
        PR_slider.set_val(PR)

        gamma_slider.set_val(gamma)
        t_slider.set_val(t)
        div_slider.set_val(x0)
        cell_slider.set_val(N)


    # Precalculated approximations for tests (disabled for now)
    # Test 1 
    # # # godunov first order with HLLC
    # U_final_hllc_test1, x_vec_1_test1 = tube_time_evolution((1, 0.75, 1), (0.125, 0, 0.1), M_x=300, t_final=0.2, gamma=1.4, solver=HLLC_Riemann_Solver, x_0=0.3)
    # rho_vec_1st_test1, u_vec_1st_test1, P_vec_1st_test1 = primitive_parameters_vectors(U_final_hllc_test1, gamma=1.4)
    # # # Godunov second order with HLLC
    # U_final_2nd_order_test1, x_vec_2_test1 = tube_time_evolution_2nd_order((1, 0.75, 1), (0.125, 0, 0.1), t_final=0.2, M_x=300, gamma=1.4, x_0=0.3)
    # rho_vec_2nd_test1, u_vec_2nd_test1, P_vec_2nd_test1 = primitive_parameters_vectors(U_final_2nd_order_test1, gamma=1.4)

    # # Test 2
    # # # godunov first order with HLLC
    # U_final_hllc_test2, x_vec_1_test2 = tube_time_evolution((1, -2, 0.4), (1, 2, 0.4), M_x=300, t_final=0.15, gamma=1.4, solver=HLLC_Riemann_Solver, x_0=0.5)
    # rho_vec_1st_test2, u_vec_1st_test2, P_vec_1st_test2 = primitive_parameters_vectors(U_final_hllc_test2, gamma=1.4)
    # # # Godunov second order with HLLC
    # U_final_2nd_order_test2, x_vec_2_test2 = tube_time_evolution_2nd_order((1, -2, 0.4), (1, 2, 0.4), t_final=0.15, M_x=300, gamma=1.4, x_0=0.5)
    # rho_vec_2nd_test2, u_vec_2nd_test2, P_vec_2nd_test2 = primitive_parameters_vectors(U_final_2nd_order_test2, gamma=1.4)

    # # Test 3
    # # # godunov first order with HLLC
    # U_final_hllc_test3, x_vec_1_test3 = tube_time_evolution((1, 0, 1000), (1, 0, 0.01), M_x=300, t_final=0.012, gamma=1.4, solver=HLLC_Riemann_Solver, x_0=0.5)
    # rho_vec_1st_test3, u_vec_1st_test3, P_vec_1st_test3 = primitive_parameters_vectors(U_final_hllc_test3, gamma=1.4)
    # # # Godunov second order with HLLC
    # U_final_2nd_order_test3, x_vec_2_test3 = tube_time_evolution_2nd_order((1, 0, 1000), (1, 0, 0.01), t_final=0.012, M_x=300, gamma=1.4, x_0=0.5)
    # rho_vec_2nd_test3, u_vec_2nd_test3, P_vec_2nd_test3 = primitive_parameters_vectors(U_final_2nd_order_test3, gamma=1.4)

    # # Test 4
    # # # godunov first order with HLLC
    # U_final_hllc_test4, x_vec_1_test4 = tube_time_evolution((5.99924, 19.5975, 460.894), (5.99924, -6.19633, 46.0950), M_x=300, t_final=0.035, gamma=1.4, solver=HLLC_Riemann_Solver, x_0=0.4)
    # rho_vec_1st_test4, u_vec_1st_test4, P_vec_1st_test4 = primitive_parameters_vectors(U_final_hllc_test4, gamma=1.4)
    # # # Godunov second order with HLLC
    # U_final_2nd_order_test4, x_vec_2_test4 = tube_time_evolution_2nd_order((5.99924, 19.5975, 460.894), (5.99924, -6.19633, 46.0950), t_final=0.035, M_x=300, gamma=1.4, x_0=0.4)
    # rho_vec_2nd_test4, u_vec_2nd_test4, P_vec_2nd_test4 = primitive_parameters_vectors(U_final_2nd_order_test4, gamma=1.4)

    # # Test 5
    # # # godunov first order with HLLC
    # U_final_hllc_test5, x_vec_1_test5 = tube_time_evolution((1, -19.5975, 1000), (1, -19.5975, 0.01), M_x=300, t_final=0.012, gamma=1.4, solver=HLLC_Riemann_Solver, x_0=0.8)
    # rho_vec_1st_test5, u_vec_1st_test5, P_vec_1st_test5 = primitive_parameters_vectors(U_final_hllc_test5, gamma=1.4)
    # # # Godunov second order with HLLC
    # U_final_2nd_order_test5, x_vec_2_test5 = tube_time_evolution_2nd_order((1, -19.5975, 1000), (1, -19.5975, 0.01), t_final=0.012, M_x=300, gamma=1.4, x_0=0.8)
    # rho_vec_2nd_test5, u_vec_2nd_test5, P_vec_2nd_test5 = primitive_parameters_vectors(U_final_2nd_order_test5, gamma=1.4)


    # Function to rescale the axis
    def rescale_axis():
        axs[0].relim()
        axs[0].autoscale_view()
        axs[1].relim()
        axs[1].autoscale_view()
        axs[2].relim()
        axs[2].autoscale_view()


    def setting_xy_data_approximations(rho1, u1, P1, x1, rho2, u2, P2, x2):
        density_curve_1.set_ydata(rho1) ; density_curve_1.set_xdata(x1)
        velocity_curve_1.set_ydata(u1) ; velocity_curve_1.set_xdata(x1)
        pressure_curve_1.set_ydata(P1) ; pressure_curve_1.set_xdata(x1)

        density_curve_2.set_ydata(rho2) ; density_curve_2.set_xdata(x2)
        velocity_curve_2.set_ydata(u2) ; velocity_curve_2.set_xdata(x2)
        pressure_curve_2.set_ydata(P2) ; pressure_curve_2.set_xdata(x2)

        rescale_axis()


    def setting_textboxes(left_params, right_params, t, gamma, x0=0.5, N=300):
        rhoL, uL, PL = left_params 
        rhoR, uR, PR = right_params

        rhoL_text.set_val(rhoL)
        uL_text.set_val(uL)
        PL_text.set_val(PL)

        rhoR_text.set_val(rhoR)
        uR_text.set_val(uR)
        PR_text.set_val(PR)

        t_text.set_val(t)
        gamma_text.set_val(gamma)
        x0_text.set_val(x0)
        N_cells_text.set_val(N)


    # Function for each preset
    def preset1(event): # test 1
        set_sliders_values((1, 0.75, 1), (0.125, 0, 0.1), t=0.2, gamma=1.4, x0=0.3, N=300)
        # setting_xy_data_approximations(rho_vec_1st_test1, u_vec_1st_test1, P_vec_1st_test1, x_vec_1_test1, rho_vec_2nd_test1, u_vec_2nd_test1, P_vec_2nd_test1, x_vec_2_test1)
        setting_textboxes((1, 0.75, 1), (0.125, 0, 0.1), t=0.2, gamma=1.4, x0=0.3, N=300)

    def preset2(event): # test 2
        set_sliders_values((1, -2, 0.4), (1, 2, 0.4), t=0.15, gamma=1.4, x0=0.5, N=300)
        # setting_xy_data_approximations(rho_vec_1st_test2, u_vec_1st_test2, P_vec_1st_test2, x_vec_1_test2, rho_vec_2nd_test2, u_vec_2nd_test2, P_vec_2nd_test2, x_vec_2_test2)
        setting_textboxes((1, -2, 0.4), (1, 2, 0.4), t=0.15, gamma=1.4, x0=0.5, N=300)

    def preset3(event): # test 3
        set_sliders_values((1, 0, 1000), (1, 0, 0.01), t=0.012, gamma=1.4, x0=0.5, N=300)
        # setting_xy_data_approximations(rho_vec_1st_test3, u_vec_1st_test3, P_vec_1st_test3, x_vec_1_test3, rho_vec_2nd_test3, u_vec_2nd_test3, P_vec_2nd_test3, x_vec_2_test3)
        setting_textboxes((1, 0, 1000), (1, 0, 0.01), t=0.012, gamma=1.4, x0=0.5, N=300)

    def preset4(event): # test 4
        set_sliders_values((5.99924, 19.5975, 460.894), (5.99924, -6.19633, 46.0950), t=0.035, gamma=1.4, x0=0.4, N=300)
        # setting_xy_data_approximations(rho_vec_1st_test4, u_vec_1st_test4, P_vec_1st_test4, x_vec_1_test4, rho_vec_2nd_test4, u_vec_2nd_test4, P_vec_2nd_test4, x_vec_2_test4)
        setting_textboxes((5.99924, 19.5975, 460.894), (5.99924, -6.19633, 46.0950), t=0.035, gamma=1.4, x0=0.4, N=300)

    def preset5(event): # test 5
        set_sliders_values((1, -19.5975, 1000), (1, -19.5975, 0.01), t=0.012, gamma=1.4, x0=0.8, N=300)
        # setting_xy_data_approximations(rho_vec_1st_test5, u_vec_1st_test5, P_vec_1st_test5, x_vec_1_test5, rho_vec_2nd_test5, u_vec_2nd_test5, P_vec_2nd_test5, x_vec_2_test5)
        setting_textboxes((1, -19.5975, 1000), (1, -19.5975, 0.01), t=0.012, gamma=1.4, x0=0.8, N=300)
        

    # Updating the curve based on the slider value:
    def update_curve(val):
        rhoL = rhoL_slider.val # take the value of the slider
        uL = uL_slider.val
        PL = PL_slider.val

        rhoR = rhoR_slider.val
        uR = uR_slider.val
        PR = PR_slider.val

        gamma_val = gamma_slider.val
        t_val = t_slider.val
        N_cells_val = cell_slider.val
        x_divider_val = div_slider.val

        # Exact solution update
        new_x_vec_ex, new_rho_vec_ex, new_u_vec_ex, new_P_vec_ex = exact_solution((rhoL, uL, PL), (rhoR, uR, PR), t_val, gamma_val, N_cells_val, TOL=1e-9, x_center=x_divider_val)
        density_curve.set_ydata(new_rho_vec_ex) ; density_curve.set_xdata(new_x_vec_ex)
        velocity_curve.set_ydata(new_u_vec_ex) ; velocity_curve.set_xdata(new_x_vec_ex)
        pressure_curve.set_ydata(new_P_vec_ex) ; pressure_curve.set_xdata(new_x_vec_ex)

        # x0 vline update
        x0_in_density.set_xdata([x_divider_val])
        x0_in_velocity.set_xdata([x_divider_val])
        x0_in_pressure.set_xdata([x_divider_val])

        
        rescale_axis()
        fig.canvas.draw_idle()


    # Presets buttons
    ax_preset1 = fig.add_axes([0.75, 0.03, 0.04, 0.04]) #[left, bottom, width, height]
    preset1_button = Button(ax=ax_preset1, label='Test\n1')
    preset1_button.on_clicked(preset1)
    ax_preset2 = fig.add_axes([0.8, 0.03, 0.04, 0.04])
    preset2_button = Button(ax=ax_preset2, label='Test\n2')
    preset2_button.on_clicked(preset2)
    ax_preset3 = fig.add_axes([0.85, 0.03, 0.04, 0.04])
    preset3_button = Button(ax=ax_preset3, label='Test\n3')
    preset3_button.on_clicked(preset3)
    ax_preset4 = fig.add_axes([0.9, 0.03, 0.04, 0.04])
    preset4_button = Button(ax=ax_preset4, label='Test\n4')
    preset4_button.on_clicked(preset4)
    ax_preset5 = fig.add_axes([0.95, 0.03, 0.04, 0.04])
    preset5_button = Button(ax=ax_preset5, label='Test\n5')
    preset5_button.on_clicked(preset5)

    # Register the update function with each slider
    rhoL_slider.on_changed(update_curve)
    uL_slider.on_changed(update_curve)
    PL_slider.on_changed(update_curve)

    rhoR_slider.on_changed(update_curve)
    uR_slider.on_changed(update_curve)
    PR_slider.on_changed(update_curve)

    gamma_slider.on_changed(update_curve)
    t_slider.on_changed(update_curve)
    cell_slider.on_changed(update_curve)
    div_slider.on_changed(update_curve)


    # Register checkbuttons
    def reveal_x0(label):
        if label == ' x0':
            x0_in_density.set_visible(not x0_in_density.get_visible())
            x0_in_velocity.set_visible(not x0_in_velocity.get_visible())
            x0_in_pressure.set_visible(not x0_in_pressure.get_visible())
        elif label==' legend':
            legend.set_visible(not legend.get_visible())
        elif label==' grid':
            global grid_visibility
            if 'grid_visibility' in globals():
                grid_visibility = not grid_visibility
                axs[0].grid(visible=grid_visibility)
                axs[1].grid(visible=grid_visibility)
                axs[2].grid(visible=grid_visibility)
        fig.canvas.draw_idle()


    def reveal_graphs(label):
        if label=='exact':
            density_curve.set_visible(not density_curve.get_visible())
            pressure_curve.set_visible(not pressure_curve.get_visible())
            velocity_curve.set_visible(not velocity_curve.get_visible())
        elif label=='first':
            density_curve_1.set_visible(not density_curve_1.get_visible())
            pressure_curve_1.set_visible(not pressure_curve_1.get_visible())
            velocity_curve_1.set_visible(not velocity_curve_1.get_visible())
        elif label=='second':
            density_curve_2.set_visible(not density_curve_2.get_visible())
            pressure_curve_2.set_visible(not pressure_curve_2.get_visible())
            velocity_curve_2.set_visible(not velocity_curve_2.get_visible())
        fig.canvas.draw_idle()


    ax_x0_checkbox = fig.add_axes([0.15, 0.02, 0.08, 0.05]) #[left, bottom, width, height]
    x0_checkbox = CheckButtons(ax=ax_x0_checkbox, labels=[' x0', ' legend', ' grid'], actives=[True, True, True])
    x0_checkbox.on_clicked(reveal_x0)

    ax_graphs_checkbox = fig.add_axes([0.05, 0.02, 0.08, 0.05])
    graphs_checkbutton = CheckButtons(ax=ax_graphs_checkbox, labels=['exact', 'first', 'second'], actives=[True, True, True])
    graphs_checkbutton.on_clicked(reveal_graphs)


    # Creating the option to input values into the sliders
    ax_rhoL_text = fig.add_axes([0.39, 0.06, 0.035, 0.02])
    rhoL_text = TextBox(ax=ax_rhoL_text, label=r'$\rho_{L}$', initial='1.0')
    ax_rhoR_text = fig.add_axes([0.45, 0.06, 0.035, 0.02])
    rhoR_text = TextBox(ax=ax_rhoR_text, label=r'$\rho_{R}$', initial='0.125')

    ax_uL_text = fig.add_axes([0.39, 0.035, 0.035, 0.02])
    uL_text = TextBox(ax=ax_uL_text, label=r'$u_{L}$', initial='0.0')
    ax_uR_text = fig.add_axes([0.45, 0.035, 0.035, 0.02])
    uR_text = TextBox(ax=ax_uR_text, label=r'$u_{R}$', initial='0.0')

    ax_PL_text = fig.add_axes([0.39, 0.01, 0.035, 0.02])
    PL_text = TextBox(ax=ax_PL_text, label=r'$P_{L}$', initial='1.0')
    ax_PR_text = fig.add_axes([0.45, 0.01, 0.035, 0.02])
    PR_text = TextBox(ax=ax_PR_text, label=r'$P_{R}$', initial='0.1')

    ax_gamma_text = fig.add_axes([0.51, 0.06, 0.035, 0.02])
    gamma_text = TextBox(ax=ax_gamma_text, label=r'$\gamma$', initial='1.4')

    ax_N_cells_text = fig.add_axes([0.51, 0.035, 0.035, 0.02])
    N_cells_text = TextBox(ax=ax_N_cells_text, label=r'$N$', initial='300')

    ax_x0_text = fig.add_axes([0.51, 0.01, 0.035, 0.02])
    x0_text = TextBox(ax=ax_x0_text, label=r'$x_{0}$', initial='0.5')

    ax_t_text = fig.add_axes([0.57, 0.06, 0.035, 0.02])
    t_text = TextBox(ax=ax_t_text, label=r'$t$', initial='0.2')


    # Creating apply and clear buttons for text values:
    ax_apply = fig.add_axes([0.565, 0.035, 0.05, 0.02])
    apply_button = Button(ax=ax_apply, label='Apply')

    ax_clear = fig.add_axes([0.565, 0.01, 0.05, 0.02])
    clear_button = Button(ax=ax_clear, label='Clear')

    def text_input(input):
        # float(input)
        try:
            float(input)
            print(float(input))
        except ValueError:
            print('please enter a number')

        
    def apply_vals(event):
        rhoL = float(rhoL_text.text)
        uL = float(uL_text.text)
        PL = float(PL_text.text)

        rhoR = float(rhoR_text.text)
        uR = float(uR_text.text)
        PR = float(PR_text.text)

        t_val = float(t_text.text)
        gamma_val = float(gamma_text.text)
        N_cells_val = int(N_cells_text.text)
        x_divider_val = float(x0_text.text)

        set_sliders_values((rhoL, uL, PL), (rhoR, uR, PR), t=t_val, gamma=gamma_val, x0=x_divider_val, N=N_cells_val)


    def initial_text_vals(event):
        rhoL_text.set_val('1.0')
        uL_text.set_val('0.0')
        PL_text.set_val('1.0')

        rhoR_text.set_val('0.125')
        uR_text.set_val('0.0')
        PR_text.set_val('0.1')

        t_text.set_val('0.2')
        gamma_text.set_val('1.4')
        N_cells_text.set_val('300')
        x0_text.set_val('0.5')


    apply_button.on_clicked(apply_vals)
    clear_button.on_clicked(initial_text_vals)

    # Creating buttons to calculate first and second approximations using the data in the textbox
    ax_first = fig.add_axes([0.63, 0.05, 0.05, 0.03])
    first_button = Button(ax=ax_first, label='First')

    ax_second = fig.add_axes([0.63, 0.01, 0.05, 0.03])
    second_button = Button(ax=ax_second, label='Second')


    def first_approximation(event):
        rhoL = float(rhoL_text.text)
        uL = float(uL_text.text)
        PL = float(PL_text.text)

        rhoR = float(rhoR_text.text)
        uR = float(uR_text.text)
        PR = float(PR_text.text)

        t_val = float(t_text.text)
        gamma_val = float(gamma_text.text)
        N_cells_val = int(N_cells_text.text)
        x_divider_val = float(x0_text.text)

        set_sliders_values((rhoL, uL, PL), (rhoR, uR, PR), t=t_val, gamma=gamma_val, x0=x_divider_val, N=N_cells_val)

        # Godunov first order with HLLC
        U_final_hllc_btn, x_vec_1_btn = tube_time_evolution((rhoL, uL, PL), (rhoR, uR, PR), M_x=N_cells_val, t_final=t_val, gamma=gamma_val, solver=HLLC_Riemann_Solver, x_0=x_divider_val)
        rho_vec_1st_btn, u_vec_1st_btn, P_vec_1st_btn = primitive_parameters_vectors(U_final_hllc_btn, gamma=gamma_val)

        density_curve_1.set_ydata(rho_vec_1st_btn) ; density_curve_1.set_xdata(x_vec_1_btn)
        velocity_curve_1.set_ydata(u_vec_1st_btn) ; velocity_curve_1.set_xdata(x_vec_1_btn)
        pressure_curve_1.set_ydata(P_vec_1st_btn) ; pressure_curve_1.set_xdata(x_vec_1_btn)

        rescale_axis()


    def second_approximation(event):
        rhoL = float(rhoL_text.text)
        uL = float(uL_text.text)
        PL = float(PL_text.text)

        rhoR = float(rhoR_text.text)
        uR = float(uR_text.text)
        PR = float(PR_text.text)

        t_val = float(t_text.text)
        gamma_val = float(gamma_text.text)
        N_cells_val = int(N_cells_text.text)
        x_divider_val = float(x0_text.text)

        set_sliders_values((rhoL, uL, PL), (rhoR, uR, PR), t=t_val, gamma=gamma_val, x0=x_divider_val, N=N_cells_val)

        # Godunov second order with HLLC
        U_final_2nd_order_btn, x_vec_2_btn = tube_time_evolution_2nd_order((rhoL, uL, PL), (rhoR, uR, PR), t_final=t_val, M_x=N_cells_val, gamma=gamma_val, x_0=x_divider_val)
        rho_vec_2nd_btn, u_vec_2nd_btn, P_vec_2nd_btn = primitive_parameters_vectors(U_final_2nd_order_btn, gamma_val)

        density_curve_2.set_ydata(rho_vec_2nd_btn) ; density_curve_2.set_xdata(x_vec_2_btn)
        velocity_curve_2.set_ydata(u_vec_2nd_btn) ; velocity_curve_2.set_xdata(x_vec_2_btn)
        pressure_curve_2.set_ydata(P_vec_2nd_btn) ; pressure_curve_2.set_xdata(x_vec_2_btn)

        rescale_axis()


    first_button.on_clicked(first_approximation)
    second_button.on_clicked(second_approximation)


    # Textbox for information
    ax_tests_exp = fig.add_axes([0.8, 0.1, 0.00, 0.00])
    text_exp_title = TextBox(ax=ax_tests_exp, label='', initial='Riemann Problem Tests')

    ax_checkbox_exp = fig.add_axes([0.1, 0.1, 0.00, 0.00])
    checkbox_exp_title = TextBox(ax=ax_checkbox_exp, label='', initial='Hide/Reveal')

    ax_hllc_exp = fig.add_axes([0.64, 0.1, 0.00, 0.00])
    hllc_exp_title = TextBox(ax=ax_hllc_exp, label='', initial='HLLC')

    ax_textbox_parameters = fig.add_axes([0.40, 0.1, 0.00, 0.00])
    textbox_parameters_title = TextBox(ax=ax_textbox_parameters, label='', initial='Custom Parameters (Exact/HLLC)')

    # Rescale axis button
    def rescale_axis_func_for_btn(event):
        axs[0].relim(visible_only=True)
        axs[1].relim(visible_only=True)
        axs[2].relim(visible_only=True)
        axs[0].autoscale_view()
        axs[1].autoscale_view()
        axs[2].autoscale_view()

    ax_rescale_btn = fig.add_axes([0.69, 0.03, 0.05, 0.03])
    rescale_btn = Button(ax=ax_rescale_btn, label='Rescale')
    rescale_btn.on_clicked(rescale_axis_func_for_btn)

    # Creating a reset to initial values button
    ax_reset = fig.add_axes([0.25, 0.03, 0.1, 0.04]) #[left, bottom, width, height]
    reset_button = Button(ax=ax_reset, label='Reset All', hovercolor='0.975')

    def reset_vals(event):
        rhoL_slider.reset()
        uL_slider.reset()
        PL_slider.reset()

        rhoR_slider.reset()
        uR_slider.reset()
        PR_slider.reset()

        gamma_slider.reset()
        t_slider.reset()
        cell_slider.reset()
        div_slider.reset()

        initial_text_vals(event)
        setting_xy_data_approximations(rho_vec_1st, u_vec_1st, P_vec_1st, x_vec_1, rho_vec_2nd, u_vec_2nd, P_vec_2nd, x_vec_2)

        fig.canvas.draw_idle()

    reset_button.on_clicked(reset_vals)

    plt.show()


if __name__ == '__main__':
    main()
