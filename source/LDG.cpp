#include "../include/LDG.hpp"



// TO MAKE A STEADY STATE SOLVER: Replace 1.0/delta_t -> 0.0/delta_t

namespace LDG_MX
{
using namespace std;

// LDG FEM constructor
template<int dim>
LDG<dim>::
LDG()
    :
    test_carrier_solution(),
    test_carrier_bc(),
    test_carrier_rhs()	//: instantiate function objects
{
    Dirichlet = 0;
    Neumann   = 1;
}

template<int dim>
void
LDG<dim>::
assemble_local_LDG_mass_matrix(
    const typename DoFHandler<dim>::active_cell_iterator & cell,
    Assembly::AssemblyScratch<dim>						 & scratch,
    Assembly::DriftDiffusion::CopyData<dim>				 & data,
    const double	 									 & delta_t)
{
    const unsigned int dofs_per_cell = scratch.carrier_fe_values.dofs_per_cell;
    const unsigned int n_q_points		 = scratch.carrier_fe_values.n_quadrature_points;

    cell->get_dof_indices(data.local_dof_indices);

    const FEValuesExtractors::Scalar Density(dim);

    // reinitialize everything for this cell
    scratch.carrier_fe_values.reinit(cell);
    data.local_mass_matrix=0;

    // loop over all the quadrature points of this cell
    for(unsigned int q=0; q<n_q_points; q++)
    {
        // loop over all the test function dofs on this cell
        for(unsigned int i=0; i<dofs_per_cell; i++)
        {

            const double					psi_i_density			 =
                scratch.carrier_fe_values[Density].value(i,q);

            // loop over all the trial function dofs for this cell
            for(unsigned int j=0; j<dofs_per_cell; j++)
            {
                const double					psi_j_density			 =
                    scratch.carrier_fe_values[Density].value(j,q);
                // construct the local mass matrix
                // int_{Omega} (1/dt) * v * u dx
                data.local_mass_matrix(i,j) +=
                    (1.0/delta_t) * psi_i_density * psi_j_density
                    * scratch.carrier_fe_values.JxW(q);
            }
        }
    }
}


template<int dim>
void
LDG<dim>::
assemble_local_LDG_cell_and_bc_terms(
    const typename DoFHandler<dim>::active_cell_iterator & cell,
    Assembly::AssemblyScratch<dim>							 & scratch,
    Assembly::DriftDiffusion::CopyData<dim>			 & data,
    const double											 					 & scaled_mobility,
    const double 											 				   & delta_t)
{
    // NOTE: this has been called by assemble_carrier_system so it is local to a cell
    //  Assembles the body intergral matrix terms as well as the flux matrices for
    // the boundary conditions

    const unsigned int dofs_per_cell = scratch.carrier_fe_values.dofs_per_cell;
    const unsigned int n_q_points		 = scratch.carrier_fe_values.n_quadrature_points;
    const unsigned int n_face_q_points =
        scratch.carrier_fe_face_values.n_quadrature_points;

    cell->get_dof_indices(data.local_dof_indices);

    const FEValuesExtractors::Vector Current(0);
    const FEValuesExtractors::Scalar Density(dim);

    // reinitialize everything for this cell
    scratch.carrier_fe_values.reinit(cell);
    data.local_matrix=0;

    // loop over all the quadrature points of this cell
    for(unsigned int q=0; q<n_q_points; q++)
    {
        // loop over all the test function dofs on this cell
        for(unsigned int i=0; i<dofs_per_cell; i++)
        {
            // get the test functions for this cell at quadrature point q
            const Tensor<1, dim>  psi_i_field			 	 =
                scratch.carrier_fe_values[Current].value(i,q);
            const double 					div_psi_i_field		 =
                scratch.carrier_fe_values[Current].divergence(i,q);
            const Tensor<1, dim>  grad_psi_i_density =
                scratch.carrier_fe_values[Density].gradient(i,q);
            const double 					psi_i_density		 	=
                scratch.carrier_fe_values[Density].value(i,q);


            // loop over all the trial function dofs for this cell
            for(unsigned int j=0; j<dofs_per_cell; j++)
            {
                // get the trial functions for this cell at the quadrature point 1
                const Tensor<1, dim>	psi_j_field			 =
                    scratch.carrier_fe_values[Current].value(j,q);
                const double 					psi_j_density		 =
                    scratch.carrier_fe_values[Density].value(j,q);

                // construct the local LDG stiffness matrix i.e. all the solid integrals
                // int_{Omega} ((1/dt) * v * u + p * \mu^{-1} * q - div(p) * u - grad(v) * q) dx
                data.local_matrix(i,j)  +=
                    (
                        ((1.0/delta_t) * psi_i_density * psi_j_density )
                        +
                        ( psi_i_field * (1.0/scaled_mobility) * psi_j_field)
                        -
                        (div_psi_i_field * psi_j_density)
                        -
                        (grad_psi_i_density * psi_j_field)
                    )
                    * scratch.carrier_fe_values.JxW(q);
            } // for j
        } // for i
    }	// for q

    // loop over all the faces of this cell to see which one are
    // on the boundary and calculate the local flux matrices corresponding to
    // 1.) Dirichlet boundary conditions
    // 2.) Neumann boundary conditions
    for(unsigned int face_no=0; face_no< GeometryInfo<dim>::faces_per_cell; face_no++)
    {
        // get the face_no-th face of this cell
        typename DoFHandler<dim>::face_iterator 	face = cell->face(face_no);

        // test to see if it is as the boundary
        if(face->at_boundary() )
        {
            // reinitialize fe_face_values for this face and then
            // if compute corresponding boundary condition matrix
            scratch.carrier_fe_face_values.reinit(cell, face_no);

            // construct the dirichlet matrix
            if(face->boundary_id() == Dirichlet)
            {
                // loop over alll the quadrature points of this face
                for(unsigned int q=0; q<n_face_q_points; q++)
                {
                    // loop over test function dofs of this face
                    for(unsigned int i=0; i<dofs_per_cell; i++)
                    {
                        // get the test function
                        const double 		psi_i_density	=
                            scratch.carrier_fe_face_values[Density].value(i,q);

                        // loop over all trial function dofs of this face
                        for(unsigned int j=0; j<dofs_per_cell; j++)
                        {
                            // get the trial function
                            const Tensor<1, dim>  psi_j_field	=
                                scratch.carrier_fe_face_values[Current].value(j,q);

                            // int_{\Gamma_{D}} v^{-} n^{-} q^{-} ds
                            data.local_matrix(i,j) += psi_i_density *
                                                      scratch.carrier_fe_face_values.normal_vector(q) *
                                                      psi_j_field *
                                                      scratch.carrier_fe_face_values.JxW(q);

                        } // end for j
                    } // end for i
                } // for q
            } // if Dirichlet
            else if(face->boundary_id() == Neumann) // Neumman matrix
            {
                // loop over alll the quadrature points of this face
                for(unsigned int q=0; q<n_face_q_points; q++)
                {
                    // loop over test function dofs of this face
                    for(unsigned int i=0; i<dofs_per_cell; i++)
                    {
                        // get the test function
                        const Tensor<1, dim>  psi_i_field	=
                            scratch.carrier_fe_face_values[Current].value(i,q);

                        // loop over all the trial function dofs of this face
                        for(unsigned int j=0; j<dofs_per_cell; j++)
                        {
                            // get the trial function
                            const double 	psi_j_density =
                                scratch.carrier_fe_face_values[Density].value(j,q);

                            // int_{\Gamma_{N}} p^{-} n^{-} u^{-} ds
                            data.local_matrix(i,j) +=
                                psi_i_field *
                                scratch.carrier_fe_face_values.normal_vector(q) *
                                psi_j_density *
                                scratch.carrier_fe_face_values.JxW(q);
                        } // end j
                    } // end i
                } // end q
            } // end Neumann
            else
                Assert(false, ExcNotImplemented() ); // no other boundary terms
        } //if on boundary

    } // for face_no
} // assemble cell


template<int dim>
void
LDG<dim>::
assemble_local_flux_terms(
    Assembly::AssemblyScratch<dim>									 & scratch,
    Assembly::DriftDiffusion::CopyData<dim>					 & data,
    const double 																		 & penalty)
{
    // this has been called from a cells face and constructs the local ldg flux
    // matrices across that face
    const unsigned int n_face_points 			=
        scratch.carrier_fe_face_values.n_quadrature_points;
    const unsigned int dofs_this_cell 		=
        scratch.carrier_fe_face_values.dofs_per_cell;
    const unsigned int dofs_neighbor_cell =
        scratch.carrier_fe_neighbor_face_values.dofs_per_cell;

    const FEValuesExtractors::Vector Current(0);
    const FEValuesExtractors::Scalar Density(dim);

    // reset the local LDG flux matrices to zero
    data.vi_ui_matrix = 0;
    data.vi_ue_matrix = 0;
    data.ve_ui_matrix = 0;
    data.ve_ue_matrix = 0;

    // loop over all the quadrature points on this face
    for(unsigned int q=0; q<n_face_points; q++)
    {
        // loop over all the test functiion dofs of this face
        // and get the test function values at this quadrature point
        for(unsigned int i=0; i<dofs_this_cell; i++)
        {
            const Tensor<1,dim>  psi_i_field_minus	  =
                scratch.carrier_fe_face_values[Current].value(i,q);
            const double			 	 psi_i_density_minus	 		=
                scratch.carrier_fe_face_values[Density].value(i,q);

            // loop over all the trial function dofs of this face
            for(unsigned int j=0; j<dofs_this_cell; j++)
            {
                // loop over all the trial functiion dofs of this face
                // and get the trial function values at this quadrature point

                const Tensor<1,dim>	psi_j_field_minus		=
                    scratch.carrier_fe_face_values[Current].value(j,q);
                const double 			psi_j_density_minus		=
                    scratch.carrier_fe_face_values[Density].value(j,q);

                // int_{face} n^{-} * ( p_{i}^{-} u_{j}^{-} + v^{-} q^{-} ) dx
                // 					  + penalty v^{-}u^{-} dx
                data.vi_ui_matrix(i,j)	+= (
                                           0.5 * (
                                        	 psi_i_field_minus *
                                           scratch.carrier_fe_face_values.normal_vector(q) *
                                           psi_j_density_minus
                                           +
                                           psi_i_density_minus *
                                           scratch.carrier_fe_face_values.normal_vector(q) *
                                           psi_j_field_minus )
                                           +
                                           penalty *
                                           psi_i_density_minus *
                                           psi_j_density_minus
                                           ) *
                                           scratch.carrier_fe_face_values.JxW(q);
            } // for j

            for(unsigned int j=0; j<dofs_neighbor_cell; j++)
            {
                const Tensor<1, dim>	psi_j_field_plus	=
                    scratch.carrier_fe_neighbor_face_values[Current].value(j,q);
                const double 			psi_j_density_plus		=
                    scratch.carrier_fe_neighbor_face_values[Density].value(j,q);

                // int_{face} n^{-} * ( p_{i}^{-} u_{j}^{+} + v^{-} q^{+} ) dx
                // 					  - penalty v^{-}u^{+} dx
                data.vi_ue_matrix(i,j) += (
                                          0.5 * (
                                          psi_i_field_minus *
                                          scratch.carrier_fe_face_values.normal_vector(q) *
                                          psi_j_density_plus
                                          +
                                          psi_i_density_minus *
                                          scratch.carrier_fe_face_values.normal_vector(q) *
                                          psi_j_field_plus )
                                          -
                                          penalty *
                                          psi_i_density_minus *
                                          psi_j_density_plus
                                          ) *
                                          scratch.carrier_fe_face_values.JxW(q);
            } // for j
        } // for i

        for(unsigned int i=0; i<dofs_neighbor_cell; i++)
        {
            const Tensor<1,dim>  psi_i_field_plus		 	 =
                scratch.carrier_fe_neighbor_face_values[Current].value(i,q);
            const double				 psi_i_density_plus	 	 =
                scratch.carrier_fe_neighbor_face_values[Density].value(i,q);

            for(unsigned int j=0; j<dofs_this_cell; j++)
            {
                const Tensor<1, dim>	psi_j_field_minus	=
                    scratch.carrier_fe_face_values[Current].value(j,q);
                const double 			psi_j_density_minus		=
                    scratch.carrier_fe_face_values[Density].value(j,q);

                // int_{face} -n^{-} * ( p_{i}^{+} u_{j}^{-} + v^{+} q^{-} )
                // 					  - penalty v^{+}u^{-} dx

                data.ve_ui_matrix(i,j) +=	(
                                          -0.5 * (
                                            psi_i_field_plus *
                                            scratch.carrier_fe_face_values.normal_vector(q) *
                                            psi_j_density_minus
                                            +
                                            psi_i_density_plus *
                                            scratch.carrier_fe_face_values.normal_vector(q) *
                                            psi_j_field_minus)
                                            -
                                            penalty *
                                            psi_i_density_plus *
                                            psi_j_density_minus
                                            ) *
                                            scratch.carrier_fe_face_values.JxW(q);
            } // for j

            for(unsigned int j=0; j<dofs_neighbor_cell; j++)
            {
                const Tensor<1, dim>	psi_j_field_plus	=
                    scratch.carrier_fe_neighbor_face_values[Current].value(j,q);
                const double 			psi_j_density_plus		=
                    scratch.carrier_fe_neighbor_face_values[Density].value(j,q);

                // int_{face} -n^{-} * ( p_{i}^{+} u_{j}^{+} + v^{+} q^{+} )
                // 					  + penalty v^{+}u^{+} dx
                data.ve_ue_matrix(i,j) +=	(
                                          -0.5 * (
                                          psi_i_field_plus *
                                          scratch.carrier_fe_face_values.normal_vector(q) *
                                          psi_j_density_plus
                                          +
                                          psi_i_density_plus *
                                          scratch.carrier_fe_face_values.normal_vector(q) *
                                          psi_j_field_plus )
                                          +
                                          penalty *
                                          psi_i_density_plus *
                                          psi_j_density_plus
                                          ) *
                                          scratch.carrier_fe_face_values.JxW(q);
            } // for j
        } // for i
    } // for q
} // end assemble_flux_terms()

template<int dim>
void
LDG<dim>::
assemble_local_child_flux_terms(
    Assembly::AssemblyScratch<dim>								& scratch,
    Assembly::DriftDiffusion::CopyData<dim>			  & data,
    const double 													 				& penalty)
{
    // this has been called from a cells face and constructs the local ldg flux
    // matrices across that face
    const unsigned int n_face_points 			=
        scratch.carrier_fe_subface_values.n_quadrature_points;
    const unsigned int dofs_this_cell 		=
        scratch.carrier_fe_subface_values.dofs_per_cell;
    const unsigned int dofs_neighbor_cell =
        scratch.carrier_fe_neighbor_face_values.dofs_per_cell;

    const FEValuesExtractors::Vector Current(0);
    const FEValuesExtractors::Scalar Density(dim);

    //reset the local LDG flux matrices to zero
    data.vi_ui_matrix = 0;
    data.vi_ue_matrix = 0;
    data.ve_ui_matrix = 0;
    data.ve_ue_matrix = 0;

    // loop over all the quadrature points on this face
    for(unsigned int q=0; q<n_face_points; q++)
    {
        // loop over all the test functiion dofs of this face
        // and get the test function values at this quadrature point
        for(unsigned int i=0; i<dofs_this_cell; i++)
        {
            const Tensor<1,dim>  psi_i_field_minus	  =
                scratch.carrier_fe_subface_values[Current].value(i,q);
            const double			 	 psi_i_density_minus	 		=
                scratch.carrier_fe_subface_values[Density].value(i,q);

            // loop over all the trial function dofs of this face
            for(unsigned int j=0; j<dofs_this_cell; j++)
            {
                // loop over all the trial functiion dofs of this face
                // and get the trial function values at this quadrature point

                const Tensor<1,dim>	psi_j_field_minus		=
                    scratch.carrier_fe_subface_values[Current].value(j,q);
                const double 			psi_j_density_minus		=
                    scratch.carrier_fe_subface_values[Density].value(j,q);

                // int_{face} n^{-} * ( p_{i}^{-} u_{j}^{-} + v^{-} q^{-} ) dx
                // 					  + penalty v^{-}u^{-} dx
                data.vi_ui_matrix(i,j)	+= (
                                            0.5 * (
                                            psi_i_field_minus *
                                            scratch.carrier_fe_subface_values.normal_vector(q) *
                                            psi_j_density_minus
                                            +
                                            psi_i_density_minus *
                                            scratch.carrier_fe_subface_values.normal_vector(q) *
                                            psi_j_field_minus )
                                            +
                                            penalty *
                                            psi_i_density_minus *
                                            psi_j_density_minus
                                           ) *
                                           scratch.carrier_fe_subface_values.JxW(q);
            } // for j

            for(unsigned int j=0; j<dofs_neighbor_cell; j++)
            {
                const Tensor<1, dim>	psi_j_field_plus	=
                    scratch.carrier_fe_neighbor_face_values[Current].value(j,q);
                const double 			psi_j_density_plus		=
                    scratch.carrier_fe_neighbor_face_values[Density].value(j,q);

                // int_{face} n^{-} * ( p_{i}^{-} u_{j}^{+} + v^{-} q^{+} ) dx
                // 					  - penalty v^{-}u^{+} dx
                data.vi_ue_matrix(i,j) += (
                                          0.5 * (
                                          psi_i_field_minus *
                                          scratch.carrier_fe_subface_values.normal_vector(q) *
                                          psi_j_density_plus
                                          +
                                          psi_i_density_minus *
                                          scratch.carrier_fe_subface_values.normal_vector(q) *
                                          psi_j_field_plus )
                                          -
                                          penalty *
                                          psi_i_density_minus *
                                          psi_j_density_plus
                                          ) *
                                          scratch.carrier_fe_subface_values.JxW(q);
            } // for j
        } // for i

        for(unsigned int i=0; i<dofs_neighbor_cell; i++)
        {
            const Tensor<1,dim>  psi_i_field_plus		 	 =
                scratch.carrier_fe_neighbor_face_values[Current].value(i,q);
            const double				 psi_i_density_plus	 	 =
                scratch.carrier_fe_neighbor_face_values[Density].value(i,q);

            for(unsigned int j=0; j<dofs_this_cell; j++)
            {
                const Tensor<1, dim>	psi_j_field_minus	=
                    scratch.carrier_fe_subface_values[Current].value(j,q);
                const double 			psi_j_density_minus		=
                    scratch.carrier_fe_subface_values[Density].value(j,q);

                // int_{face} -n^{-} * ( p_{i}^{+} u_{j}^{-} + v^{+} q^{-} )
                // 					  - penalty v^{+}u^{-} dx

                data.ve_ui_matrix(i,j) +=	(
                                         -0.5 * (
                                         psi_i_field_plus *
                                         scratch.carrier_fe_subface_values.normal_vector(q) *
                                         psi_j_density_minus
                                         +
                                         psi_i_density_plus *
                                         scratch.carrier_fe_subface_values.normal_vector(q) *
                                         psi_j_field_minus)
                                         -
                                         penalty *
                                         psi_i_density_plus *
                                         psi_j_density_minus
                                        ) *
                                        scratch.carrier_fe_subface_values.JxW(q);
            } // for j

            for(unsigned int j=0; j<dofs_neighbor_cell; j++)
            {
                const Tensor<1, dim>	psi_j_field_plus	=
                    scratch.carrier_fe_neighbor_face_values[Current].value(j,q);
                const double 			psi_j_density_plus		=
                    scratch.carrier_fe_neighbor_face_values[Density].value(j,q);

                // int_{face} -n^{-} * ( p_{i}^{+} u_{j}^{+} + v^{+} q^{+} )
                // 					  + penalty v^{+}u^{+} dx
                data.ve_ue_matrix(i,j) +=	(
                                        -0.5 * (
                                         psi_i_field_plus *
                                         scratch.carrier_fe_subface_values.normal_vector(q) *
                                         psi_j_density_plus
                                         +
                                         psi_i_density_plus *
                                         scratch.carrier_fe_subface_values.normal_vector(q) *
                                         psi_j_field_plus )
                                         +
                                         penalty *
                                         psi_i_density_plus *
                                         psi_j_density_plus
                                         ) *
                                         scratch.carrier_fe_subface_values.JxW(q);
            } // for j
        } // for i
    } // for q
} // end assemble_flux_terms()


template<int dim>
void
LDG<dim>::
compute_errors(const Triangulation<dim>			& triangulation,
               DoFHandler<dim>							& carrier_dof_handler,
               BlockVector<double>					& solution,
               double 											& potential_error,
               double 											& field_error) const
{
    const ComponentSelectFunction<dim> potential_mask(dim, dim+1);
    const ComponentSelectFunction<dim>
    vectorField_mask(std::make_pair(0,dim), dim+1);

    unsigned int degree = carrier_dof_handler.get_fe().degree;
    unsigned int n_cells = triangulation.n_active_cells();

    QTrapez<1>				q_trapez;
    QIterated<dim> 		quadrature(q_trapez, degree+2);
    Vector<double> 		cellwise_errors(n_cells);


    VectorTools::integrate_difference(carrier_dof_handler,
                                      solution,
                                      test_carrier_solution,
                                      cellwise_errors, quadrature,
                                      VectorTools::L2_norm,
                                      &potential_mask);

    potential_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference(carrier_dof_handler,
                                      solution,
                                      test_carrier_solution,
                                      cellwise_errors, quadrature,
                                      VectorTools::L2_norm,
                                      &vectorField_mask);

    field_error = cellwise_errors.l2_norm();

    std::cout << "\nErrors: ||e_pot||_L2 = " << potential_error
              << ",		||e_vec_field||_L2 = " << field_error
              << std::endl << std::endl;


}
} // end namespace
