#include "../include/MixedFEM.hpp"



namespace LDG_MX
{
using namespace std;

//  mixed FEM constructor
template<int dim>
MixedFEM<dim>::
MixedFEM()	:
    test_Poisson_solution(),
    test_Poisson_bc(),
    test_Poisson_rhs() //: instantiate function objects
{
    Dirichlet = 0;
    Neumann   = 1;
}

template<int dim>
void
MixedFEM<dim>::
assemble_local_Poisson_matrix(
    const typename DoFHandler<dim>::active_cell_iterator & cell,
    Assembly::AssemblyScratch<dim>		& scratch,
    Assembly::Poisson::CopyData<dim>	& data,
    const double 						& debeye_length)
{
    const unsigned int	dofs_per_cell 	=
        scratch.Poisson_fe_values.dofs_per_cell;
    const unsigned int 	n_q_points 			=
        scratch.Poisson_fe_values.n_quadrature_points;

    cell->get_dof_indices(data.local_dof_indices);

    // Get the actual values for vector field and potential from FEValues
    // Use Extractors instead of having to deal with shapefunctions directly
    const FEValuesExtractors::Vector VectorField(0); // Vector as in Vector field
    const FEValuesExtractors::Scalar Potential(dim);

    scratch.Poisson_fe_values.reinit(cell);
    data.local_matrix=0;

    // loop over all the quadrature points in this cell
    for(unsigned int q=0; q<n_q_points; q++)
    {
        //  loop over test functions dofs for this cell
        for(unsigned int i=0; i<dofs_per_cell; i++)
        {
            // i-th VectorField basis functions at point q
            const Tensor<1, dim>	psi_i_field =
                scratch.Poisson_fe_values[VectorField].value(i,q);

            // div. of the i-th VectorField basis functions at the point q
            const double	div_psi_i_field =
                scratch.Poisson_fe_values[VectorField].divergence(i,q);

            // i-th potential basis functions at the point q
            const double psi_i_potential =
                scratch.Poisson_fe_values[Potential].value(i,q);

            // loop over all the trial functions dofs for this cell
            for(unsigned int j=0; j<dofs_per_cell; j++)
            {
                // j-th VectorField basis functions at point q
                const Tensor<1, dim>	psi_j_field =
                    scratch.Poisson_fe_values[VectorField].value(j,q);

                // div. of the j-th VectorField basis functions at the point q
                const double	div_psi_j_field =
                    scratch.Poisson_fe_values[VectorField].divergence(j,q);


                // i-th potential basis functions at the point q
                const double psi_j_potential =
                    scratch.Poisson_fe_values[Potential].value(j,q);

                // build whole matrix at once, not blocks individually.
                // \int (P * k^{-1} D - div P * phi - v * D ) dx
                data.local_matrix(i,j) += (psi_i_field *  psi_j_field
                                           - div_psi_i_field * psi_j_potential
                                           - psi_i_potential * debeye_length * div_psi_j_field
                                          ) * scratch.Poisson_fe_values.JxW(q);

            } // for j

        } // for i
    } // for q
} // assemble_local_Poisson_matrix




template<int dim>
void
MixedFEM<dim>::
compute_errors(const Triangulation<dim>			& triangulation,
               DoFHandler<dim>					& Poisson_dof_handler,
               BlockVector<double>				& solution,
               double 							& potential_error,
               double 							& field_error) const
{
    const ComponentSelectFunction<dim> potential_mask(dim, dim+1);
    const ComponentSelectFunction<dim>
    vectorField_mask(std::make_pair(0,dim), dim+1);

    unsigned int degree = Poisson_dof_handler.get_fe().degree;
    unsigned int n_cells = triangulation.n_active_cells();

    QTrapez<1>				q_trapez;
    QIterated<dim> 		quadrature(q_trapez, degree+2);
    Vector<double> 		cellwise_errors(n_cells);


    VectorTools::integrate_difference(Poisson_dof_handler,
                                      solution,
                                      test_Poisson_solution,
                                      cellwise_errors, quadrature,
                                      VectorTools::L2_norm,
                                      &potential_mask);

    potential_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference(Poisson_dof_handler,
                                      solution,
                                      test_Poisson_solution,
                                      cellwise_errors, quadrature,
                                      VectorTools::L2_norm,
                                      &vectorField_mask);

    field_error = cellwise_errors.l2_norm();

    std::cout << "\nErrors: ||e_pot||_L2 = " << potential_error
              << ",		||e_vec_field||_L2 = " << field_error
              << std::endl << std::endl;


}

} // end namespace
