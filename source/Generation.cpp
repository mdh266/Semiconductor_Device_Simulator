#include <deal.II/base/function.h>
#include <math.h>

#include "Parameters.hpp"


using namespace dealii;

///////////////////////////////////////////////////////////////////////////////
// Carrier Recombination Functions
///////////////////////////////////////////////////////////////////////////////


template <int dim>
class Generation : public Function<dim>
{
public:
    Generation() : Function<dim>(1)
    {}

    void		set_dark_params();
    void		set_illuminated_params(const ParameterSpace::Parameters & Params);

    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0 ) const;

private:
    double scaled_photon_flux;
    double scaled_absorption_coeff;

};

template <int dim>
void
Generation<dim>::
set_dark_params()
{
    scaled_photon_flux = 0.0;
    scaled_absorption_coeff	= 0.0;
}



template <int dim>
void
Generation<dim>::
set_illuminated_params(const ParameterSpace::Parameters & params)
{
    scaled_photon_flux = params.scaled_photon_flux;
    scaled_absorption_coeff	= params.scaled_absorption_coeff;
}

template <int dim>
double Generation<dim>::value(const Point<dim> &p,
                              const unsigned int ) const
{

    // G_0 * aplpha * exp(\alpha ( 1 - y))
    return scaled_absorption_coeff *
           scaled_photon_flux *
           exp(-scaled_absorption_coeff * (1.0 - p[1]));


}


