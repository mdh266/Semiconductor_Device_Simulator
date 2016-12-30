#ifndef _GRID_H__
#define _GRID_H__
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
//#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/grid_out.h>
#include <iostream>
#include <fstream>

#include "Parameters.hpp"


/// \namespace Namespace for Grid class which creates all different types of meshes.
namespace Grid_Maker
{
using namespace dealii;

template<int dim>
class Grid
{
public:
    /// Grid object constructor.
    Grid();

    /** \brief Creates a simple cubic grid. */
    /** Takes <code>triangulation<code/> object and creates a mesh on it.
    * Globally refines the mesh using the refinement level from
    * <code>params.n_global_refine</code>.
    *
    * Calculates <code>params.h_min</code>,  <code>params.h_max</code>, and
    * <code>params.penalty</code> (which is for the LDG method) and stores them in
    * <code>params<code/>.
    */
    void make_grid(Triangulation<dim> 				& triangulation,
                   ParameterSpace::Parameters	& params);

    /** \brief Functions that manually refine the grid. */
    /** Takes <code>triangulation<code/> object and creates a mesh on it.
    * Globally refines the mesh using the refinement level from
    * <code>params.n_global_refine</code>.
    * Then it locally refines cells to the level
    * <code>params.n_local_refine</code>.
    * The locally refined cells are in specific areas of the mesh which have been
    * preset in this functions source code.
    *
    * Calculates <code>params.h_min</code>,  <code>params.h_max</code>, and
    * <code>params.penalty</code> (which is for the LDG method) and stores them in
    * <code>params<code/>.
    *
    */
    void make_local_refined_grid(Triangulation<dim> 				& triangulation,
                                 ParameterSpace::Parameters	& params);

    ///\brief Function that tags boundaries to be Dirichlet portions.
    /** This function loops over all the cells in <code>triangulation<code/h>
    * and finds which functions are on the boundary.  It then
    * flags these faces on the boundary to be <code>Dirichlet<code/>
    * portions of boundary.
    */
    void make_Dirichlet_boundaries(Triangulation<dim> & triangulation);

    /// \brief Function that tags boundaries to be Neumann/Robin portions.
    /** This function loops over all the cells in <code>triangulation<code/>
    * and finds which functions are on the boundary.  It then
    * flags these faces to be <code>Neumann<code/> portions of boundary.
    * The choice of which boundary faces are <code>Neumann<code/> is
    * preset in this functions source code.
    *
    * NOTE: BECAREFUL TO MATCH WITH CUBE END POINTS AND DIMENSION.
    * NOTE: <code>Neumann<code/> corresponds to a Neumann bounary condition
    * for Poisson and a Robin boundary condition for the drift-diffusion equation.
    */
    void make_Neumann_boundaries(Triangulation<dim> & triangulation);

    /// \brief Function that prints the grid into a .eps file
    void print_grid(Triangulation<dim> & triangulation);

private:
    unsigned int Dirichlet;
    unsigned int Neumann;
};

}

#endif
