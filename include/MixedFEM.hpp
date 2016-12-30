#ifndef _MIXED_FEM_H__
#define _MIXED_FEM_H__

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

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

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include "../include/Assembly.hpp"
#include "../source/BiasValues.cpp"
#include "../source/test_functions.cpp"

namespace LDG_MX
{
using namespace dealii;

/** Assembles each cells local left hand side matrix for a constant
*		Debeye length. Also has can compute the error of the approximation
* 	for testing.
*
*   The general problem is,
* 	\f[ \begin{align}
*		\textbf{D} \ + \ \nabla \Phi \ &= \ 0  && \text{in} \; \Omega \\
*	  \lambda^{2} \ \nabla \ \cdot \ \textbf{D} \  &=
*		\ N_{D} \ - \ N_{A} \ -\  \left( \rho_{n} \ - \ \rho_{p} \right)
*						&& \text{in} \; \Omega \\
*		\textbf{D} \ \cdot \ \boldsymbol \eta \ &= \ 0
*						&& \text{on} \; \partial \Omega_{N} \\
*		\Phi \ &=  \ \Phi_{\text{app}} \ + \ \Phi_{\text{bi}}
*							 && \text{on} \; \partial \Omega_{D}
*   \end{align} \f]
*
*
*	This becomes the problem in the weak formulation: Find
*	\f$( \ \Phi \ , \ \textbf{D} \ ) \ \in
*	\left( \ \text{W} \ \times \ [ \ 0 , \ T \ ] ,
*	\ \textbf{V}^{d} \ \times \
*  	[ \ 0 , \ T \ ] \ \right) \f$ such that:
*
* 	\f[ \begin{align}
* \left(  \textbf{p} \ , \ \textbf{D} \right)_{\Omega}
*	\  - \
* \left( \ \boldsymbol \nabla \cdot \ \textbf{p}  \ , \  \Phi  \right)_{\Omega}
*  \; &= \;
*  - \langle \textbf{p} \ , \ \Phi_{\text{app}}  +  \Phi_{\text{bi}} \rangle_{\Gamma_{D}}   \\
* - \lambda^{-1} \ \left(  v \ , \ \boldsymbol \nabla  \cdot \textbf{D} \right)_{\Omega} \;
*  &= \; -
* \left(  v,\ N_{D} \ - \ N_{A} \ -\ \left( \rho_{n} \ - \ \rho_{p} \right) \right)
*	\end{align} \f]
*
*
*  For all \f$( v \  , \ \textbf{p}  ) \, \in \, W \, \times\, \textbf{V}^{d}\f$.
*
*
*
*  This method only assembles the left hand side of the weak formulation.
*/

template<int dim>
class MixedFEM
{
public:
    /** \brief Simple constructor which instantiates test function. */
    MixedFEM();

    /** \brief Assembles a local cells matrix for Poissons equation. */
    /** This function can either be called when looping throug the cells by
    * 	hand and assemblying the global matrix sequentially or by using
    * 	the WorkStream to assemble it in parallel.  If you use it in squential
    * 	mode, than you must have the AssemblyScratch and Poisson::CopyData
    *		instantiated before calling this function.
    *
    *		This function loops through the quadrature points of this cell and
    * 	assembles a local matrix corresponding to the Mixed Method applied
    * 	Poisson equation and stores it in Poisson::CopyData.
    *
    *	 @param debeye length \f$\lambda{^2}\f$ is a constant for now.
    *	 @param scratch is the temporary scratch objects and data structures
    * 			that do the work.
    *	 @param data is the local data structures to this cell that the computed
    *			results get stored in before being passed off onto the global
    *			data structures.
    *
    * 	The matrix will be of the form,
    *   \f[ \left[ \begin{matrix}
    *			 A & B \\
    *			 \lambda^{2} B^{T} & 0
    *			 \end{matrix} \right]  \f]
    *
    * where,
    *
    *	 \f[ A(\textbf{p},\textbf{D}) \; = \;
    *				\int_{\Omega} \  \textbf{p} \  \cdot \  \textbf{D} \ dx \f]
    *
    *	 \f[ B(\textbf{p},\Phi) \; = \;
    *				\int_{\Omega} \ \nabla \ \cdot \ \textbf{p} \  \Phi  \ dx \f]
    *
    *	 \f[ B^{T}(v,\textbf{D} ) \; = \;
    *				\int_{\Omega} \ \nabla v \ \cdot \ \textbf{D} \ dx \f]
    *
    *
    */

    void
    assemble_local_Poisson_matrix(
        const typename DoFHandler<dim>::active_cell_iterator & cell,
        Assembly::AssemblyScratch<dim>											 & scratch,
        Assembly::Poisson::CopyData<dim>										 & data,
        const double 													 							 & debeye_length);

    /** \brief Computes the local error of your approximation for the
    	*  mixed method on the cell and stores the errors in
    	* <code>potential_error<code/> and <code>field_error<code/>.
    	*
    	* @param solution is the <code>Poisson_solution<code/>
    	*  vector to the Mixed Method.
    	* @param potential_error is \f$L^{2}\f$  error of the approximation
    	*					the potential.
    	* @param field_error is \f$L^{2}\f$ error of the approximation
    	*					electric field.
    	*/
    void
    compute_errors(const Triangulation<dim>	& triangulation,
                   DoFHandler<dim>							& Poisson_dof_handler,
                   BlockVector<double>					& solution,
                   double 											& potential_error,
                   double 											& field_error) const;

private:
    unsigned int Dirichlet;
    unsigned int Neumann;

public:
    const test_Poisson::TrueSolution<dim>								test_Poisson_solution;
    const test_Poisson::DirichletBoundaryValues<dim>		test_Poisson_bc;
    const test_Poisson::RightHandSide<dim>							test_Poisson_rhs;


};

}

#endif
