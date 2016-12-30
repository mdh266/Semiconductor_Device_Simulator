///
// Poisson solver using Local Discontinuous Galerkin Method
//
//

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h> // Mobility and debeye 
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h> // for block structuring
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h> // for block structuring
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

//#include <deal.II/fe/fe_dgq.h> // Lagrange dg fe elements
#include <deal.II/fe/fe_dgp.h> // Legendre dg fe elements
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

// Matrix object which acts as the inverse of a matrix by
// calling an iterative solver.
#include <deal.II/lac/iterative_inverse.h>

// Tensor valued functions

// multithreading
#include<deal.II/base/work_stream.h>
#include<deal.II/base/multithread_info.h>


#include "../source/Generation.cpp"
#include "../source/BiasValues.cpp"
#include "../source/DopingProfiles.cpp"
#include "../source/Assembly.cpp"
#include "../source/MixedFEM.cpp"
#include "../source/LDG.cpp"
#include "../source/Grid.cpp"
#include "ParameterReader.hpp"


namespace LDG_MX
{
using namespace dealii;
using namespace ParameterSpace;


///////////////////////////////////////////////////////////////////////////////
// LDG POISSON CLASS
///////////////////////////////////////////////////////////////////////////////

/** Schockley-Reed-Hall Recombination functional
* @param Holds the intrinsic density and recombination times.
* @param electron_density \f$\rho_{n}^{k-1}\f$.
* @param hole_density \f$\rho_{p}^{k-1}\f$.
*/
inline	double SRH_Recombination(const double & electron_density,
                                 const double & hole_density,
                                 const Parameters & params)
{
    return ((params.scaled_intrinsic_density *
             params.scaled_intrinsic_density -
             electron_density * hole_density) /
            (params.scaled_electron_recombo_t * (electron_density -
                    params.scaled_intrinsic_density) +
             (params.scaled_hole_recombo_t * (hole_density -
                     params.scaled_intrinsic_density)) ) );
}

/** DriftDiffusionPoisson class used to solve the coupled flow transport problem.
* Solves the coupled flow-transport problem discussed on the main page.
* Solves Poissons equation using a mixed finite element (LDG_MX::MixedFEM) and the
* drift-diffusion equation using a local discontinuous Galerkin (LDG_MX::LDG) method.
* This class will call those classes to construct the appropriate matrices and
* the members of this class will be used to construct the appropriate right
* hand sides for each eqauation. Poisson's equation is used solving implicit
* electron and hole values while the drift-diffusion equations use an IMEX
* time stepping method on the the non-linearities and coupling to Poisson's
* equation.  Please see the classe defitions as well as
* LDG_MX::DriftDiffusionPoisson.assemble_Poisson_rhs()
* and LDG_MX::DriftDiffusionPoisson.assemble_local_LDG_rhs() for more details.
*/
template <int dim>
class DriftDiffusionPoisson
{
public:
    /** Class constructor which initializes FEValue objects for all
    * the equations.  It also initializes functions as well as reading
    * <code>input_file.prm<code/> to read the input variables and then
    * scales them using singular perturbation scaling.
    */
    DriftDiffusionPoisson(const unsigned int degree,
                          ParameterHandler &);

    ~DriftDiffusionPoisson();

    void run();

private:
    /** Reads <code>input_file.prm<code/>, reads in the input parameters and
    * scales appropriete values for singular pertubation scaling.
    */
    void parse_and_scale_parameters();

    /** Allocates memory for matrices and vectors, distribute degrees
    * of freedom for each of the equation.
    * NOTE: electrons and holes use the same FEValues, DoFHandler,
    * sparsity pattern, but different system matrices.  This implicitly
    * assumes that we will have all the same simulation features, just
    * different parameter values.
    */
    void make_dofs_and_allocate_memory();

    // Poisson
    /** The mixed finite element method on Poissons equation (MixedFEM)
    * imposes and Neumann boundary condtions strongly.  This function
    * therefore constrains the DOFS of freedom to be zero on the Neumann
    * segments of the boundary.
    * NOTE: This must be called AFTER GridMaker::make_Neumann_boundaries() has been
    * called in the Grid object.
    */
    void enforce_Neumann_boundaries_on_Poisson();

    /** This function assembles the matrix for the MixedFEM on a global
    * level. It uses the workstream function from deal.ii to
    * assemble the matrix in parallel.  It needs a MixedFEM object to call
    * in order to assemble.
    * It will also call
    * LDG_MX::DriftDiffusionPoisson.copy_local_to_global_Poisson_matrix()
    * to assemble the global matrix from the local ones.
    */
    void assemble_Poisson_matrix();

    /** Copys the local matrices assembled my and MixedFEM object into
    * the global matrix.  Used for assembly in parallel by workstream.
    */
    void copy_local_to_global_Poisson_matrix(
        const Assembly::Poisson::CopyData<dim> 	& data);

    /** This function assembles the right hand side for the MixedFEM on
    * a global level. It uses the workstream function from deal.ii to
    * in order to assemble using the assemble_local_Poisson_rhs()function.
    * It will also call copy_local_to_global_rhs() to assemble
    * the global matrix from the local ones.
    */
    void assemble_Poisson_rhs();

    /** This assembles the coupled right hand side,
    *	\f[
    *	= ; \left[ N_{D} - N_{A}  - (\rho_{n}^{k-1} -\rho_{p}^{k-1}) \right]
    *	\f]
    *
    *  NOTE: It needs to be in this class and not in MixedFEM, because we dont want
    *	pass carrier's DoFHandler so we can access a carrier cell.
    */
    void assemble_local_Poisson_rhs(
        const typename DoFHandler<dim>::active_cell_iterator 	& cell,
        Assembly::AssemblyScratch<dim>											  & scratch,
        Assembly::Poisson::CopyData<dim>											& data);

    /** Copys the local right hand side vectors assembled my and this
    *  object into the global right hand side vector.
    *  Used for assembly in parallel by workstream.
    */
    void copy_local_to_global_Poisson_rhs(
        const Assembly::Poisson::CopyData<dim> 								& data);

    /** Calls the direct solver */
    void solve_Poisson_system();

    // LDG
    /** This function assembles the mass matrix as well
    * as the electron and hole system matrices for LDG on a global
    * level. It uses the workstream function from deal.ii
    * assemble the right hand side in parallel.
    * It needs a LDG object to call in order to assemble things in parallel.
    * It will also call
    * LDG_MX::DriftDiffusionPoisson.copy_local_to_global_LDG_mass_matrix(),
    * LDG_MX::DriftDiffusionPoisson.copy_local_to_global_electron_LDG_matrix(),
    * LDG_MX::DriftDiffusionPoisson.copy_local_to_global_hole_LDG_matrix()
    * to copy the local contributions
    * to the global matrices.
    */
    void assemble_carrier_matrix();

    /** Copies the local mass matrix to the global mass matrix.  To be used
    * by workstream do assemble things in parallel.*/
    void copy_local_to_global_LDG_mass_matrix(
        const Assembly::DriftDiffusion::CopyData<dim>				 & data);

    /** Copies the local electron matrix to the global mass matrix.  To be used
    * by workstream do assemble things in parallel.*/
    void copy_local_to_global_electron_LDG_matrix(
        const Assembly::DriftDiffusion::CopyData<dim>				 & data);

    /** Copies the local hole matrix to the global mass matrix.  To be used
    * by workstream do assemble things in parallel.*/
    void copy_local_to_global_hole_LDG_matrix(
        const Assembly::DriftDiffusion::CopyData<dim>				 & data);

    /** This function assembles the right hand side for the LDG on
    * a global level. It uses the workstream function from deal.ii
    * in order to assemble the terms in parallel
    * using the LDG_MX::DriftDiffusionPoisson.assemble_local_LDG_rhs() function.
    * It will also call LDG_MX::DriftDiffusionPoisson.copy_local_to_global_LDG_rhs()
    * to assemble
    * the global right hand side from the local ones.
    */
    void assemble_carrier_rhs();

    /** This function assembles the right hand side for the LDG on
    * a local cell level.  The right hand side corresponds to,
    * 	\f[
    *	= \ \left( v ,  R(u^{k-1})  + G \right) -
    *	\langle   v, K( u^{k})    \rangle_{\Sigma}
    *	+
    *	\langle  \textbf{p}   ,  u_{D}  \rangle_{ \partial \Omega_{N} } +
    *	\left( s \textbf{P}  \cdot \boldsymbol \nabla \Phi , u^{k-1} \right)
    *	\f]
    *
    *  NOTE: It needs to be in this class and not in LDG, because we dont want
    *	pass Poisson's DoFHandler so we can access a Poisson cell.
    */
    void assemble_local_LDG_rhs(
        const typename DoFHandler<dim>::active_cell_iterator & cell,
        Assembly::AssemblyScratch<dim>											 & scratch,
        Assembly::DriftDiffusion::CopyData<dim>							 & data);

    /** Copies the local right hand side vector into the global one. */
    void copy_local_to_global_LDG_rhs(
        const Assembly::DriftDiffusion::CopyData<dim> 			 & data);

    /** solves the system of equation for the electron equations. */
    void solve_electron_system();
    /** solves the system of equation for the hole equations. */
    void solve_hole_system();

    /** Runs throught values of electrons and holes and if they are
    * less than zero, sets them to be zero. */
    void project_back_densities();

    void output_carrier_system() const;

    /* Prints all the data for this time stamps */
    void output_results() const;

    unsigned int Dirichlet;
    unsigned int Neumann;


    unsigned int time_step_number;
    unsigned int degree;




private:
    Parameters 						parameters;
    ParameterHandler				&prm;

    Triangulation<dim>		triangulation;

    // MIXED POSSION DATA STRUCTURES
    FESystem<dim>								Poisson_fe;
    DoFHandler<dim>							Poisson_dof_handler;

    ConstraintMatrix						Poisson_constraints;

    BlockSparsityPattern				Poisson_sparsity_pattern;
    BlockSparseMatrix<double>		Poisson_system_matrix;

    BlockVector<double>					Poisson_system_rhs;
    BlockVector<double>					Poisson_solution;

    // LDG DRIFT-DIFFUSION DATA STRUCTURES
    FESystem<dim>									carrier_fe;

    DoFHandler<dim>								carrier_dof_handler;

    BlockSparsityPattern					carrier_sparsity_pattern;
    BlockSparsityPattern					carrier_mass_sparsity_pattern;

    BlockSparseMatrix<double>			carrier_mass_matrix;

    BlockSparseMatrix<double>			electron_system_matrix;
    BlockSparseMatrix<double>			hole_system_matrix;

    BlockVector<double>						electron_solution;
    BlockVector<double>						hole_solution;

    BlockVector<double>						old_electron_solution;
    BlockVector<double>						old_hole_solution;


    BlockVector<double>						electron_system_rhs;
    BlockVector<double>						hole_system_rhs;

    MixedFEM<dim>									Assembler_MX;
    LDG<dim>											Assembler_LDG;

    // PHYSICAL FUNCTIONS
    const DonorDoping<dim>				donor_function;
    const AcceptorDoping<dim>			acceptor_function;
    // TODO: Maybe allow them to be declared constant in the actual
    // functions that way we can instaniate them after reading
    // paramtervalues/ assign them in input_file.prm
    Generation<dim>								generation_function;
    const Applied_Bias<dim>				applied_bias;
    const Built_In_Bias<dim>			built_in_bias;


    // Direct solvers
    SparseDirectUMFPACK Poisson_solver;
    SparseDirectUMFPACK electron_solver;
    SparseDirectUMFPACK hole_solver;


};

}
