#ifndef _LDG_H__
#define _LDG_H__

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h> // Lagrange dg fe elements
//#include <deal.II/fe/fe_dgp.h> // Legendre dg fe elements
//#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>


#include "../include/Assembly.hpp"
#include "../source/test_functions.cpp"


namespace LDG_MX
{
using namespace dealii;

/** This class will build portions of the LDG local cell matrix for the problem,
	*	The drift diffusion equation is :
	*
	*	\f[ \begin{align}
	* 	u_{t} \  - \  \boldsymbol \nabla \ \cdot
	* 	\ \mu \left( s \boldsymbol \nabla \Phi u \ + \ \boldsymbol \nabla u \ \right)
	* 	\; &= \;
	* 	R(u) + G	&& \text{in} \;  \Omega   \\
	* 	u \; &= \; u_{D} &&  \text{on} \;  \Omega_{D}     \\
	* 	- \mu \left(s \boldsymbol \nabla \Phi \ u \ + \ \boldsymbol \nabla u \ \right)
	* 	\ \cdot \ \boldsymbol \eta
	*	\;  &= \; K (u) && \text{on} \; \partial \Omega_{N}
	* 	\end{align} \f]
	*
	* 	We rewrite this in mixed form:
	*
	* \f[ \begin{align}
	*			u_{t} \ + \ \nabla \ \textbf{q} \ &= \ R(u) \ + G && \text{in} \ \Omega \\
	*			\mu^{-1} \ \textbf{q} \ & =
	*								 \ -s \nabla \Phi \ u \ - \nabla u && \text{in} \ \Omega \\
	*			\mu^{-1} \ \textbf{q} \ \cdot \boldsymbol \eta &=
	*									\ K(u) && \text{on} \ \partial \ \Omega_{N} \\
	*			u \ &= \ u_{D} && \text{on} \ \partial \Omega_{D}
	*	\end{align} \f]
	*
	* 	The weak formulation for IMEX will be:
	*
	*	Find \f$(u, \textbf{q}) \in W \times [t^{k-1}, t^{k}]
	*			\times \textbf{W}^{d} \times[ t^{k-1}, t^{k}] \f$ such that,
	*
	*  \f[ \begin{align}
	*	\frac{1}{\Delta t} \left(  v , u^{k}  \right)
	* 	-
	*	\tau \langle  [[ \ v \ ]] ,
	*				 [[ u^{k} ]] \rangle_{\mathcal{E}_{h}^{0}}
	*	-
	*	\left( \boldsymbol \nabla v  ,  \textbf{q}^{k}  \right)
	*	+
	*	 \langle [[ \ v \ ]] ,
	*	\{ \textbf{q}^{k} \} \rangle_{\mathcal{E}_{h}^{0} \cap \partial \Omega_{D}}
	*	\ &= \
	*	\left( v ,  R(u^{k-1})  + G \right) -
	*	\langle   v, K( u^{k})    \rangle_{\Sigma}   \nonumber \\
	*	 - \left(  \boldsymbol \nabla \cdot \textbf{p} ,   u^{k-1} \right)
	*  \ - \
	*	\langle  [[ \,  \textbf{p} \, ]] ,
	*	\{  u^{k} \}  \rangle_{\mathcal{E}^{0}_{h} \cap \partial \Omega_{N}}
	*	\ + \
	*	 \left( \textbf{p} , \textbf{q}^{k} \right)
	*	\ &= \
	*	\langle  \textbf{p}   ,  u_{D}  \rangle_{ \partial \Omega_{N} } +
	*	\left( s \textbf{P}  \cdot \boldsymbol \nabla \Phi , u^{k-1} \right)
	*	\end{align} \f]
	*
	*  For all \f$(v,\textbf{p}) \in W \times \textbf{W}^{d})\f$.
	*
	* 	The corresponding matrix will be of the form,
	*   \f[ \left[ \begin{matrix}
	*		\mu^{-1} A & B_{1} + F_{1} \\
	*		 B_{2} + F_{2} & \frac{1}{\Delta t} M + C
	*	 \end{matrix} \right]  \f]
	*
	*
	* 	NOTE: We use IMEX time stepping so all non-linear terms and drift terms are
	*				time lagged and therefore on the right hand side. While they are
	*				are built in parallel, this place takes place outside of this class.
	*
	*	NOTE: The LDG flux matrices are built sequentially and this occurs in a loop
	*  			outside this class, but calls the function assemble_local_flux_terms
	*				or in the case of a locally/adaptively refined mesh calls
	*				assemble_local_child_flux_terms.
	*
	*/
template<int dim>
class LDG
{
public:
    /** \brief A simple constructor which instantiates test functions. */
    LDG();

    /* \brief Assembles the local mass matrix for this cell. */
    /** This function can either be called when looping throug the cells by
    * 	hand and assemblying the global matrix sequentially or by using
    * 	the WorkStream to assemble it in parallel.  If you use it in squential
    * 	mode, than you must have the AssemblyScratch and DriftDiffusion::CopyData
    *		instantiated before calling this function.
    *
    *		This function loops through the quadrature points of this cell and
    * 	assembles a local mass matrix for the LDG method applied to the
    * 	drift-diffusion equation and stores it in DriftDiffusionCopyData.
    *
    *	 @param delta_t is the fixed time step size.
    *  @param scaled_mobility is the scaled mobility constant \f$\mu\f$.
    *
    * 	The matrix will be of the form,
    *   \f[ \left[ \begin{matrix}
    *			 0 & 0 \\
    *			 0 & \frac{1}{\Delta t} M
    *			 \end{matrix} \right]  \f]
    *
    * where,
    *
    *
    *	 \f[ M(v,u ) \; = \;
    *				\int_{\Omega} \ v \ u \ dx \f]
    */

    void
    assemble_local_LDG_mass_matrix(
        const typename DoFHandler<dim>::active_cell_iterator & cell,
        Assembly::AssemblyScratch<dim>												 & scratch,
        Assembly::DriftDiffusion::CopyData<dim>								 & data,
        const double	 																			 	 & delta_t);



    /** \brief Assembles the local sytem matrix for this cell. */
    /** This function can either be called when looping throug the cells by
    * 	hand and assemblying the global matrix sequentially or by using
    * 	the WorkStream to assemble it in parallel.  If you use it in squential
    * 	mode, than you must have the AssemblyScratch and DriftDiffusion::CopyData
    *		instantiated before calling this function.
    *
    *		This function loops through the quadrature points of this cell and
    * 	assembles a local mass matrix for the LDG method applied to the
    * 	drift-diffusion equation and stores it in DriftDiffusionCopyData.
    *
    *	 @param delta_t is the fixed time step size.
    *  @param scaled_mobility is the scaled mobility constant \f$\mu\f$.
    *
    * 	The matrix will be of the form,
    *   \f[ \left[ \begin{matrix}
    *			 \mu^{-1} A & B_{1} \\
    *			 B_{2} & \frac{1}{\Delta t} M + C
    *			 \end{matrix} \right]  \f]
    *
    * where,
    *
    *
    *	 \f[ A(\textbf{p},\textbf{q} ) \; = \;
    *				\int_{\Omega} \ \textbf{p} \ \cdot \textbf{q} \ dx \f]
    *
    *	 \f[ M(v,u ) \; = \;
    *				\int_{\Omega} \ v \ u \ dx \f]
    *
    *	 \f[ B_{1}(v,\textbf{q} ) \; = \;
    *				\int_{\Omega} \ \nabla \ v  \ \textbf{q} \ dx
    *				\ + \
    *				\int_{\partial \Omega_{D}} v  \  \textbf{q}
    *							\ \cdot \boldsymbol \eta \ ds	\f]
    *
    *	 \f[ B_{2}(\textbf{p},u ) \; = \;
    *				\int_{\Omega} \ \nabla \ \cdot \ \textbf{p} \ u \ dx
    *				\ + \
    *				\int_{\partial \Omega_{N}} \textbf{p}
    *							\cdot \boldsymbol \eta \ u \ ds	\f]
    *
    */
    void
    assemble_local_LDG_cell_and_bc_terms(
        const typename DoFHandler<dim>::active_cell_iterator & cell,
        Assembly::AssemblyScratch<dim>											 & scratch,
        Assembly::DriftDiffusion::CopyData<dim>							 & data,
        const double																				 & scaled_mobility,
        const double 																				 & delta_t);

    /** 	\brief Assemble the local LDG flux matrices with no local refinement. */
    /** 	These are the matrices which correspond to
    *
    *	 \f[ F_{1}(v,\textbf{q} ) \; = \;
    *				\langle [[ \ v \ ]] ,
    *			\{ \textbf{q} \} \rangle_{\mathcal{E}_{h}^{0} }  \f]
    *
    *	 \f[ F_{2}(\textbf{p},u ) \; = \;
    *				\langle  [[ \,  \textbf{p} \, ]] ,
    *							\{  u \}  \rangle_{\mathcal{E}^{0}_{h} }  \f]
    *
    */
    void
    assemble_local_flux_terms(
        Assembly::AssemblyScratch<dim>						& scratch,
        Assembly::DriftDiffusion::CopyData<dim>	  & data,
        const double 														  & penalty);


    /** 	\brief Assemble the local LDG flux matrices with local refinement. */
    /** 	These are the matrices which correspond to
    *
    *	 \f[ F_{1}(v,\textbf{q} ) \; = \;
    *				\langle [[ \ v \ ]] ,
    *			\{ \textbf{q} \} \rangle_{\mathcal{E}_{h}^{0} }  \f]
    *
    *	 \f[ F_{2}(\textbf{p},u ) \; = \;
    *				\langle  [[ \,  \textbf{p} \, ]] ,
    *							\{  u \}  \rangle_{\mathcal{E}^{0}_{h} }  \f]
    *
    * 	but with local refinement across interior face edges.
    */
    void
    assemble_local_child_flux_terms(
        Assembly::AssemblyScratch<dim>						& scratch,
        Assembly::DriftDiffusion::CopyData<dim>	  & data,
        const double 														  & penalty);

    /** \brief Computes the local error of your approximation for the
    *  LDG method on the cell and stores the errors in
    * <code>potential_error<code/> and <code>field_error<code/>.
    *
    * @param solution is the <code>carrier_solution<code/>
    *  vector to the LDG on Poisson's equation!
    * @param potential_error is \f$L^{2}\f$  error of the approximation
    *					the potential.
    * @param field_error is \f$L^{2}\f$ error of the approximation
    *					electric field.
    */
    void
    compute_errors(const Triangulation<dim>	& triangulation,
                   DoFHandler<dim>							& carrier_dof_handler,
                   BlockVector<double>					& solution,
                   double 											& potential_error,
                   double 											& field_error) const;

private:
    unsigned int Dirichlet;
    unsigned int Neumann;

public:
    const test_Poisson::TrueSolution<dim>								test_carrier_solution;
    const test_Poisson::DirichletBoundaryValues<dim>		test_carrier_bc;
    const test_Poisson::RightHandSide<dim>							test_carrier_rhs;

};

}

#endif
