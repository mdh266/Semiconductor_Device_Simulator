#ifndef _PARAMETERS_H__
#define _PARAMETERS_H__

namespace ParameterSpace
{
using namespace dealii;

/// Struct which olds the parameters used for simulations

/*! These are the parameters which will be used by
*		DriftDiffusionPoisson class to do simulations. They will be read
*		read in using the ParameterReader class to read them from
*		input_file.prm and then use the ParameterHandler from deal.ii to
* 	set the variables and scale them if necessary.
*/

struct Parameters
{
    // computational
    unsigned int n_global_refine;
    unsigned int n_local_refine;
    unsigned int time_stamps;
    double 	 	 	 h_max;
    double			 h_min;
    double 			 t_end;
    double 			 delta_t;
    double 			 penalty;

    double scaled_electron_mobility;
    double scaled_electron_recombo_t;

    double scaled_hole_mobility;
    double scaled_hole_recombo_t;

    double scaled_intrinsic_density;
    double material_permittivity;

    double scaled_absorption_coeff;
    double scaled_photon_flux;

    double scaled_debeye_length;


    double characteristic_length;
    double characteristic_time;
    double characteristic_denisty;
};
}


namespace PhysicalConstants
{
const double thermal_voltage  	 = 0.02585; // [V]
const double electron_charge  	 = 1.62e-19;  // [C]
const double vacuum_permittivity = 8.85e-14; // [A s V^{-1} cm^{-1}
}
#endif
