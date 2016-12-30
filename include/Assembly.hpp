#ifndef _ASSEMBLY_H__
#define _ASSEMBLY_H__

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h> // Lagrange dg fe elements
//#include <deal.II/fe/fe_dgp.h> // Legendre dg fe elements
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nothing.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>


#include <fstream>
#include <iostream>

/** \namespace Assembly namespace holds the temporary scratch assembly objects
*		 as well as the local data to be copied to global data for each equation.
*/
namespace Assembly
{

using namespace dealii;

////////////////////////////////////////////////////////////////////////////
/// \struct Temporary scratch objects used to assemble local matrices and vectors
////////////////////////////////////////////////////////////////////////////
template<int dim>
struct AssemblyScratch
{
    /** This is the constructor you would call if you wanted to create
     * 	 an AssemblyScratch object explicitly and loop over all the cells and
     * 	 assemble the local matrices and vectors by hand.  This would be done
     *   sequentially.
    */
    AssemblyScratch(const FiniteElement<dim> & Poisson_fe,
                    const FiniteElement<dim> & carrier_fe,
                    const Quadrature<dim>		& quadrature,
                    const Quadrature<dim-1>	& face_quadrature);

    /** This is the copy constructor that will be called in the WorkStream
    * 	 call that will build and distribute the local matrices and vectors
    *   in parallel using Thread Building Blocks.
    */
    AssemblyScratch(const AssemblyScratch & scratch);

    FEValues<dim>					Poisson_fe_values;
    FEFaceValues<dim>			Poisson_fe_face_values;
    FEValues<dim>					carrier_fe_values;
    FEFaceValues<dim>			carrier_fe_face_values;
    FESubfaceValues<dim>	carrier_fe_subface_values;
    FEFaceValues<dim>			carrier_fe_neighbor_face_values;

    /// vectorwhich holds right hand values of Poissons equation
    /// at the quadrature points of the local cell
    std::vector<double>			Poisson_rhs_values;


    /// Vector which holds boundary conditions
    /// values of Poissons equation at the quadrature points of the local cell
    std::vector<double>			Poisson_bc_values;

    /// Vector which holds built_in potential
    /// values of Poissons equation at the quadrature points of the local cell
    std::vector<double>			Poisson_bi_values;

    /// Vector which holds the electric field values
    ///	at the quadrature points of the local cell
    std::vector<Tensor<1,dim>>  electric_field_values;


    /// Vector which holds the electron density values
    ///	at the quadrature points of the local cell
    std::vector<double> 		old_electron_density_values;

    /// Vector which holds the hole density values
    ///	at the quadrature points of the local cell
    std::vector<double>			old_hole_density_values;

    /// Vector which holds the donor doping profile density values
    ///	at the quadrature points of the local cell
    std::vector<double>			donor_doping_values;

    /// Vector which holds the acceptor doping profile density values
    ///	at the quadrature points of the local cell
    std::vector<double>			acceptor_doping_values;

    /// Vector which holds the generation functions values
    ///	at the quadrature points of the local cell
    std::vector<double>			generation_values;

    /// Vector which holds the electron dirichlet boundary conditions
    ///	at the quadrature points of the local face
    std::vector<double>			electron_bc_values;

    /// Vector which holds the hole dirichlet boundary conditions
    ///	at the quadrature points of the local face
    std::vector<double>			hole_bc_values;

};


/// \namespace Poisson's local copy data
namespace Poisson
{
/** \struct  Objects to store local Poisson data  before being distributed
* to global data */
template<int dim>
struct CopyData
{
    /** This is the constructor you will call explicitly when looping over
     * the cells and building the local matrices and vectors by hand sequentially.
    */
    CopyData(const FiniteElement<dim> & Poisson_fe);

    /** This is the copy constructor that will be called in the WorkStream
     * 	 call that will store the local matrices and vectors
     *   in parallel using Thread Building Blocks.
    */
    CopyData(const CopyData & data);

    /// local vector for this cells contribution to Poissons equation's
    /// right hand side
    Vector<double>												local_rhs;
    /// local matrix for this cells contribution to Poissons equation's
    /// left hand side matrix
    FullMatrix<double>										local_matrix;
    /// Vector which holds the local to global degrees of freedom info
    ///  of this cell
    std::vector<types::global_dof_index>	local_dof_indices;

}; // end struct

} // end namespace Poisson

/// \namespace drift diffusion equations 's local copy data
namespace DriftDiffusion
{
/** \struct  Objects to store local drift-diffusion equation data
*  before being distributed to global data */
template<int dim>
struct CopyData
{
    /** This is the constructor you will call explicitly when looping over
     * the cells and building the local matrices and vectors by hand sequentially.
    */
    CopyData(const FiniteElement<dim> & carrier_fe);

    /** This is the copy constructor that will be called in the WorkStream
     * 	 call that will store the local matrices and vectors
     *   in parallel using Thread Building Blocks.
    */
    CopyData(const CopyData & data);

    /// local vector for this cells contribution to electron transport
    /// equation's right hand side: drift, generation recombination and
    /// boundary terms
    Vector<double>												local_electron_rhs;
    /// local vector for this cells contribution to hole transport
    /// equation's right hand side: drift, generation recombination and
    /// boundary terms
    Vector<double>												local_hole_rhs;
    /// local matrix for this cells contribution to drift diffusion equation's
    /// left hand side matrix which involves body intergrals
    FullMatrix<double>										local_matrix;

    /// local matrix for this cells contribution to drift diffusion equation's
    /// mass matrix
    FullMatrix<double>										local_mass_matrix;

    // flux matrices
    /// Matrix for flux from both test function and trial function interior
    /// to this cell
    FullMatrix<double>										vi_ui_matrix;
    /// Matrix for flux from test function interior
    /// to this cell and trial function exterior to this cell
    FullMatrix<double>										vi_ue_matrix;
    /// Matrix for flux from test function exterior
    /// to this cell and trial function interior to this cell
    FullMatrix<double>										ve_ui_matrix;
    /// Matrix for flux from both test function and trial function exterior
    /// to this cell
    FullMatrix<double>										ve_ue_matrix;

    /// Vector which holds the local to global degrees of freedom info
    ///  of this cell
    std::vector<types::global_dof_index>	local_dof_indices;
    /// Vector which holds the local to global degrees of freedom info
    /// of this cell's neighbor cell
    std::vector<types::global_dof_index>	local_neighbor_dof_indices;

}; // end struct

} // end DriftDiffusion

} // end namespace Assembly
#endif
