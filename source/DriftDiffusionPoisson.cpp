#include "../include/DriftDiffusionPoisson.hpp"

namespace LDG_MX
{
using namespace dealii;
using namespace ParameterSpace;

template <int dim>
DriftDiffusionPoisson<dim>::
DriftDiffusionPoisson(const unsigned int degree,
                      ParameterHandler &param)
    :
    degree(degree),
    prm(param),
    Poisson_fe( FE_RaviartThomas<dim>(degree), 1,
               FE_DGP<dim>(degree), 		   1),
    Poisson_dof_handler(triangulation),
    carrier_fe( FESystem<dim>(FE_DGP<dim>(degree), dim),	1,
               FE_DGP<dim>(degree), 						1),
    carrier_dof_handler(triangulation),
    Assembler_MX(),
    Assembler_LDG(),
    donor_function(),
    acceptor_function(),
    generation_function(),
    applied_bias(),
    built_in_bias()
{
    // set the enum type to be used for the bondary id
    Dirichlet = 0;
    Neumann		= 1; // in this case for drift-diffusin Neumann is the
    // same as robin

    parse_and_scale_parameters();
//		generation_function.set_dark_params();
    generation_function.set_illuminated_params(parameters);
}

template <int dim>
DriftDiffusionPoisson<dim>::
~DriftDiffusionPoisson()
{
    Poisson_dof_handler.clear();
    carrier_dof_handler.clear();
}


template <int dim>
void
DriftDiffusionPoisson<dim>::
parse_and_scale_parameters()
{

    // read in the parameters
    prm.enter_subsection("computational");
    parameters.n_global_refine = prm.get_integer("global refinements");
    parameters.n_local_refine  = prm.get_integer("local refinements");
    parameters.delta_t  = prm.get_double("time step size");
    parameters.time_stamps = prm.get_integer("time stamps");
    prm.leave_subsection();

    prm.enter_subsection("physical");
    parameters.t_end = prm.get_double("end time");
    parameters.characteristic_length = prm.get_double("characteristic length");
    parameters.characteristic_denisty = prm.get_double("characteristic density");
    parameters.characteristic_time = prm.get_double("characteristic time");
    parameters.scaled_intrinsic_density = prm.get_double("intrinsic density");
    parameters.scaled_photon_flux = prm.get_double("photon flux");
    parameters.scaled_absorption_coeff = prm.get_double("absorption coefficient");
    parameters.material_permittivity = prm.get_double("material permittivity");
    prm.leave_subsection();

    prm.enter_subsection("electrons");
    parameters.scaled_electron_mobility = prm.get_double("mobility");
    parameters.scaled_electron_recombo_t = prm.get_double("recombination time");
    prm.leave_subsection();

    prm.enter_subsection("holes");
    parameters.scaled_hole_mobility = prm.get_double("mobility");
    parameters.scaled_hole_recombo_t = prm.get_double("recombination time");
    prm.leave_subsection();

    // scale the parameters
    parameters.scaled_intrinsic_density /= parameters.characteristic_denisty;
    parameters.scaled_electron_recombo_t /= parameters.characteristic_time;
    parameters.scaled_hole_recombo_t /= parameters.characteristic_time;


    parameters.scaled_photon_flux *= (parameters.characteristic_time /
                                      parameters.characteristic_denisty);

    parameters.scaled_absorption_coeff /= parameters.characteristic_length;

    parameters.scaled_debeye_length =
        (PhysicalConstants::thermal_voltage *
         PhysicalConstants::vacuum_permittivity *
         parameters.material_permittivity) /
        (PhysicalConstants::electron_charge *
         parameters.characteristic_denisty *
         parameters.characteristic_length *
         parameters.characteristic_length);

    parameters.scaled_electron_mobility *= (parameters.characteristic_time *
                                            PhysicalConstants::thermal_voltage) /
                                           (parameters.characteristic_length *
                                            parameters.characteristic_length);

    parameters.scaled_hole_mobility *= (parameters.characteristic_time *
                                        PhysicalConstants::thermal_voltage) /
                                       (parameters.characteristic_length *
                                        parameters.characteristic_length);


    cout << "debeye length = " << parameters.scaled_debeye_length << endl;
    cout << "scaled electron mobility = " << parameters.scaled_electron_mobility << endl;
    cout << "scaled hole mobilit = " << parameters.scaled_hole_mobility << endl;
}


template <int dim>
void
DriftDiffusionPoisson<dim>::
make_dofs_and_allocate_memory()
{

    ////////////////////////////////////////////////////////////////////////
    // MIXED SYSTEM
    ///////////////////////////////////////////////////////////////////////
    // distribute the dofs
    Poisson_dof_handler.distribute_dofs(Poisson_fe);
    DoFRenumbering::component_wise(Poisson_dof_handler);

    std::vector<types::global_dof_index> dofs_per_component(dim+1);
    DoFTools::count_dofs_per_component(Poisson_dof_handler, dofs_per_component);

    const unsigned int n_electric_field = dofs_per_component[0];
    const unsigned int n_potential			= dofs_per_component[dim];

    std::cout << "Number of active cells : "
              << triangulation.n_active_cells()
              << std::endl
//							<< "Total number of cells: "
//							<< triangulation.n_cells()
//							<< std::endl
              << "h_{max} : "
              << parameters.h_max
              << std::endl
              << "Number of DOFS Poisson: "
              << Poisson_dof_handler.n_dofs()
              << " (" << n_electric_field
              << " + " << n_potential << ")"
              << std::endl;

    // make hanging node constraints
    Poisson_constraints.clear();
    DoFTools::make_hanging_node_constraints(Poisson_dof_handler,
                                            Poisson_constraints);

    enforce_Neumann_boundaries_on_Poisson();
    Poisson_constraints.close();

    BlockDynamicSparsityPattern Poisson_dsp(2,2);

    // allocate size for A
    Poisson_dsp.block(0,0).reinit (n_electric_field,
                                   n_electric_field);

    //Allocate size for B^{T}
    Poisson_dsp.block(1,0).reinit (n_potential,
                                   n_electric_field);

    // allocate size for B
    Poisson_dsp.block(0,1).reinit (n_electric_field,
                                   n_potential);

    // allocate size for 0
    Poisson_dsp.block(1,1).reinit (n_potential,
                                   n_potential);

    Poisson_dsp.collect_sizes();

    DoFTools::make_sparsity_pattern(Poisson_dof_handler,
                                    Poisson_dsp);

    Poisson_constraints.condense(Poisson_dsp);

    Poisson_sparsity_pattern.copy_from(Poisson_dsp);


    // allocate memory for the Poisson matrix
    Poisson_system_matrix.reinit(Poisson_sparsity_pattern);

    // allocate memory for the Poisson solution vector
    Poisson_solution.reinit(2);
    Poisson_solution.block(0).reinit(n_electric_field);
    Poisson_solution.block(1).reinit(n_potential);
    Poisson_solution.collect_sizes();

    // allocate memory for the Poisson RHS vector
    Poisson_system_rhs.reinit(2);
    Poisson_system_rhs.block(0).reinit(n_electric_field);
    Poisson_system_rhs.block(1).reinit(n_potential);
    Poisson_system_rhs.collect_sizes();


    ////////////////////////////////////////////////////////////////////////
    // LDG SYSTEM
    ///////////////////////////////////////////////////////////////////////
    // distribute the dofs for the drift-diffusion equation
    carrier_dof_handler.distribute_dofs(carrier_fe);

    // Renumber dofs for [ Current, Density ]^{T} set up
    DoFRenumbering::component_wise(carrier_dof_handler);

    // dofs_per_component was declared before
    DoFTools::count_dofs_per_component(carrier_dof_handler, dofs_per_component);

    // get number of dofs in vector field components and of density
    // in each component/dimension of vector field has same number of dofs
    const unsigned int n_current = dim * dofs_per_component[0];
//		std::cout << "n_current = " << n_current << std::endl;

    const unsigned int n_density = dofs_per_component[1];
//		std::cout << "n_density = " << n_density << std::endl;

    std::cout << "Number of DOFS carrier: "
              << 2 * carrier_dof_handler.n_dofs()
              << " = ( " << carrier_dof_handler.n_dofs() << " ) "
              << " = 2 x  (" << n_current << " + " << n_density << ")"
              << std::endl
              << "Total DOFS = " << n_potential + n_electric_field
              + 2*n_density + 2*n_current << std::endl;

//		std::cout << "base elements : " << fe.n_base_elements() << std::endl;
//		std::cout << "n_blocks : " << fe.n_blocks() << std::endl;
//		std::cout << "dofs_per_cell : " << fe.dofs_per_cell;


    BlockDynamicSparsityPattern carrier_system_dsp(2,2);

    // allocate size for A
    carrier_system_dsp.block(0,0).reinit (n_current,
                                          n_current);

    //Allocate size for B^{T}
    carrier_system_dsp.block(1,0).reinit (n_density,
                                          n_current);

    // allocate size for B
    carrier_system_dsp.block(0,1).reinit (n_current,
                                          n_density);

    // allocate size for C
    carrier_system_dsp.block(1,1).reinit (n_density,
                                          n_density);

    carrier_system_dsp.collect_sizes();



    // allocate memory for [0 , 0 ; 0 , M ]
    BlockDynamicSparsityPattern carrier_mass_dsp(2,2);

    // allocate size for 0
    carrier_mass_dsp.block(0,0).reinit (n_current,
                                        n_current);

    //Allocate size for 0
    carrier_mass_dsp.block(1,0).reinit (n_density,
                                        n_current);

    // allocate size for 0
    carrier_mass_dsp.block(0,1).reinit (n_current,
                                        n_density);

    // allocate size for m
    carrier_mass_dsp.block(1,1).reinit (n_density,
                                        n_density);

    carrier_mass_dsp.collect_sizes();


    // create the actual sparsity pattern and allocate memory for the matrices

    // electrons:
    DoFTools::make_flux_sparsity_pattern (carrier_dof_handler, carrier_system_dsp);
    carrier_sparsity_pattern.copy_from(carrier_system_dsp);
    electron_system_matrix.reinit (carrier_sparsity_pattern);

    hole_system_matrix.reinit (carrier_sparsity_pattern);

    DoFTools::make_sparsity_pattern (carrier_dof_handler, carrier_mass_dsp);
    carrier_mass_sparsity_pattern.copy_from(carrier_mass_dsp);
    carrier_mass_matrix.reinit (carrier_mass_sparsity_pattern);


//		std::ofstream output("spartsity_pattern");
//		carrier_mass_sparsity_pattern.print_gnuplot(output);
//		output.close();


    // allocate memory for carrier_solutions
    electron_solution.reinit (2); // [vector field, density]
    electron_solution.block(0).reinit (n_current); // VECTOR FIELD
    electron_solution.block(1).reinit (n_density); // POTENTIAL
    electron_solution.collect_sizes ();

    hole_solution.reinit (2); // [vector field, density]
    hole_solution.block(0).reinit (n_current); // VECTOR FIELD
    hole_solution.block(1).reinit (n_density); // POTENTIAL
    hole_solution.collect_sizes ();

    // allocate memory for old_carrier_solutions
    old_electron_solution.reinit (2); // [vector field, density]
    old_electron_solution.block(0).reinit (n_current); // VECTOR FIELD
    old_electron_solution.block(1).reinit (n_density); // POTENTIAL
    old_electron_solution.collect_sizes ();

    old_hole_solution.reinit (2); // [vector field, density]
    old_hole_solution.block(0).reinit (n_current); // VECTOR FIELD
    old_hole_solution.block(1).reinit (n_density); // POTENTIAL
    old_hole_solution.collect_sizes ();

    // memeory for RHS
    electron_system_rhs.reinit (2);
    electron_system_rhs.block(0).reinit (n_current); // DIRICHLET BC
    electron_system_rhs.block(1).reinit (n_density); // RIGHT HAND SIDE
    electron_system_rhs.collect_sizes ();

    // memeory for RHS
    hole_system_rhs.reinit (2);
    hole_system_rhs.block(0).reinit (n_current); // DIRICHLET BC
    hole_system_rhs.block(1).reinit (n_density); // RIGHT HAND SIDE
    hole_system_rhs.collect_sizes ();


    // close the poisson constrains
}

////////////////////////////////////////////////////////////////////////////////////////
// POISSON MEMBER FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////
template<int dim>
void
DriftDiffusionPoisson<dim>::
enforce_Neumann_boundaries_on_Poisson()
{
    // add constaints to Poisson_Neumann_constraints to constrain dofs
    // of electric field to be 0 along the Neumann boundary.
    //
    // NOTE: The constraints must be added to the constraint matrix before
    // assembling Poisson's system matrix or right hand side.
    const FEValuesExtractors::Vector	ElectricField(0);
    ComponentMask	electric_field_mask	= Poisson_fe.component_mask(ElectricField);

    DoFTools::make_zero_boundary_constraints(Poisson_dof_handler,
            Neumann, // NEUMANN BOUNDARY INDICATOR
            Poisson_constraints,
            electric_field_mask);

} // enforce_Neumann

template <int dim>
void
DriftDiffusionPoisson<dim>::
assemble_Poisson_matrix()
{
    // BUILD THE MATRIX
    // [ A  B^{T}	]
    // [ B	0			]
    //
    // where A 		 = \int_{omega} p * D dx
    // 			 B^{T} = \int_{omega} div(p) \Phi dx
    // 			 B		 = \int_{omega} lambda * v div(D) dx
    //
    // Using the mixed method assembler object

    // use unit values
//		parameters.scaled_debeye_length = 1.0;

    ////////////////////////////////////////////////////////////////////////////
    // Parallel Assembly
    //////////////////////////////////////////////////////////////////////////

    // this is shown in deal.II on shared memory parallelism
    // it builds the above matrix by distributing over multi threads locally
    // and then building up the global matrix sequentially (kinda)

    WorkStream::run(Poisson_dof_handler.begin_active(),
                    Poisson_dof_handler.end(),
                    std_cxx11::bind(&MixedFEM<dim>::
                                    assemble_local_Poisson_matrix, // the assembly objects func
                                    Assembler_MX, // the assembly object
                                    std_cxx11::_1,
                                    std_cxx11::_2,
                                    std_cxx11::_3,
                                    parameters.scaled_debeye_length),
                    std_cxx11::bind(&DriftDiffusionPoisson<dim>::
                                    copy_local_to_global_Poisson_matrix,
                                    this, // this object
                                    std_cxx11::_1),
                    Assembly::AssemblyScratch<dim>(Poisson_fe,
                            carrier_fe,
                            QGauss<dim>(degree+2),
                            QGauss<dim-1>(degree+2)),
                    Assembly::Poisson::CopyData<dim>(Poisson_fe)
                   );

} // end assemble_Poisson_matrix

template <int dim>
void
DriftDiffusionPoisson<dim>::
copy_local_to_global_Poisson_matrix(
    const Assembly::Poisson::CopyData<dim> & data)
{
    // distribute local matrix to global Poisson matrix
    Poisson_constraints.distribute_local_to_global(data.local_matrix,
            data.local_dof_indices,
            Poisson_system_matrix);
}	// copy_local_to_global_poisson

template <int dim>
void
DriftDiffusionPoisson<dim>::
assemble_Poisson_rhs()
{
    // this assembles the right hand side of Poissons equation.  It can be used
    // at every time step so that we are only building the rhs and NOT the
    // entire linear system.

    // set equal to 0
    Poisson_system_rhs=0;

    // right hand side is not assembled by Mixed Method assembler since it
    // requires carrier_dof_handler

    //////////////////////////////////////////////////////////////////////////
    // Parallel Assembly
    //////////////////////////////////////////////////////////////////////////

    WorkStream::run(Poisson_dof_handler.begin_active(),
                    Poisson_dof_handler.end(),
                    std_cxx11::bind(&DriftDiffusionPoisson<dim>::
                                    assemble_local_Poisson_rhs,
                                    this, // this object
                                    std_cxx11::_1,
                                    std_cxx11::_2,
                                    std_cxx11::_3),
                    std_cxx11::bind(&DriftDiffusionPoisson<dim>::
                                    copy_local_to_global_Poisson_rhs,
                                    this, // this object
                                    std_cxx11::_1),
                    Assembly::AssemblyScratch<dim>(Poisson_fe,
                            carrier_fe,
                            QGauss<dim>(degree+2),
                            QGauss<dim-1>(degree+2)),
                    Assembly::Poisson::CopyData<dim>(Poisson_fe)
                   );

} // end Poisson_assemble_rhs()

template <int dim>
void
DriftDiffusionPoisson<dim>::
assemble_local_Poisson_rhs(
    const typename DoFHandler<dim>::active_cell_iterator & cell,
    Assembly::AssemblyScratch<dim>						 & scratch,
    Assembly::Poisson::CopyData<dim>					 & data)
{
    const unsigned int	dofs_per_cell 	=
        scratch.Poisson_fe_values.dofs_per_cell;
    const unsigned int 	n_q_points 			=
        scratch.Poisson_fe_values.n_quadrature_points;

    const unsigned int	n_face_q_points =
        scratch.Poisson_fe_face_values.n_quadrature_points;

    cell->get_dof_indices(data.local_dof_indices);

    // Get the actual values for vector field and potential from FEValues
    // Use Extractors instead of having to deal with shapefunctions directly
    const FEValuesExtractors::Vector VectorField(0); // Vector as in Vector field
    const FEValuesExtractors::Scalar Potential(dim);

    const FEValuesExtractors::Scalar Density(dim);

    //////////////////////////////////////////////////////////////////////////////
    // NOTE: Since we are working with both Poisson's fe_value and the
    // carrier fe_value we must have iterators over the cells held by
    // each of the dof_handlers.  While we do this, the actual cell that we are
    // working with will always be the same.
    ////////////////////////////////////////////////////////////////////////////
    typename DoFHandler<dim>::active_cell_iterator
    carrier_cell(&triangulation,
                 cell->level(),
                 cell->index(),
                 &carrier_dof_handler);

    // reinitialize the fe_values on this cell
    scratch.Poisson_fe_values.reinit(cell);
    scratch.carrier_fe_values.reinit(carrier_cell);

    // reset the local_rhs vector to be zero
    data.local_rhs=0;


    // TODO: see if should be current time step!
    // get the electron density values at the previous time step
    scratch.carrier_fe_values[Density].get_function_values(
        old_electron_solution,
        scratch.old_electron_density_values);

    // get the hole density values at the previous time step
    scratch.carrier_fe_values[Density].get_function_values(
        old_hole_solution,
        scratch.old_hole_density_values);


    // get doping profiles values on this cell
    donor_function.value_list(scratch.carrier_fe_values.get_quadrature_points(),
                              scratch.donor_doping_values,
                              dim); // calls the density values of the donor profile

    acceptor_function.value_list(scratch.carrier_fe_values.get_quadrature_points(),
                                 scratch.acceptor_doping_values,
                                 dim); // calls the density values of the donor profile

    // get the test rhs for poisson
//		Assembler_MX.test_Poisson_rhs.value_list(
//												scratch.Poisson_fe_values.get_quadrature_points(),
//												scratch.Poisson_rhs_values);


    // Loop over all the quadrature points in this cell
    for(unsigned int q=0; q<n_q_points; q++)
    {
        // loop over the test function dofs for this cell
        for(unsigned int i=0; i<dofs_per_cell; i++)
        {
            // i-th potential basis functions at the point q
            const double psi_i_potential =
                scratch.Poisson_fe_values[Potential].value(i,q);

            // get the local RHS values for this cell
            // = int_{Omega_{e}} 	(N_{D} - N_{A}) - (n - p)
            data.local_rhs(i) += -psi_i_potential
                                 *(
//																 scratch.Poisson_rhs_values[q]
                                     + (scratch.donor_doping_values[q]
                                        -
                                        scratch.acceptor_doping_values[q])
                                     - (scratch.old_electron_density_values[q]
                                        -
                                        scratch.old_hole_density_values[q])
                                 ) * scratch.Poisson_fe_values.JxW(q);
        } // for i
    } // for q

    // loop over all the faces of this cell to calculate the vector
    // from the dirichlet boundary conditions if the face is on the
    // Dirichlet portion of the boundary
    for(unsigned int face_no=0;
            face_no<GeometryInfo<dim>::faces_per_cell;
            face_no++)
    {
        // obtain the face_no-th face of this cell
        typename DoFHandler<dim>::face_iterator 	face = cell->face(face_no);

        // apply Dirichlet boundary conditions
        if((face->at_boundary()) && (face->boundary_id() == Dirichlet))
        {
            // get the values of the shape functions at this boundary face
            scratch.Poisson_fe_face_values.reinit(cell,face_no);

            // get the values of the dirichlet boundary conditions evaluated
            // on the quadrature points of this face

            // TEST ONE
//				Assembler_MX.test_Poisson_bc.value_list(
//													scratch.Poisson_fe_face_values.get_quadrature_points(),
//													scratch.Poisson_bc_values);
            // REAL ONE
            built_in_bias.value_list(
                scratch.Poisson_fe_face_values.get_quadrature_points(),
                scratch.Poisson_bi_values);


            // REAL ONE
            applied_bias.value_list(
                scratch.Poisson_fe_face_values.get_quadrature_points(),
                scratch.Poisson_bc_values);

            // loop over all the quadrature points of this face
            for(unsigned int q=0; q<n_face_q_points; q++)
            {
                // loop over all the test function dofs of this face
                for(unsigned int i=0; i<dofs_per_cell; i++)
                {
                    // - \int_{face} p * n * (phi_{Dichlet}) dx
                    data.local_rhs(i)	+=
                        -(scratch.Poisson_fe_face_values[VectorField].value(i,q) *
                          scratch.Poisson_fe_face_values.normal_vector(q) *
                          (scratch.Poisson_bc_values[q]
                           +
                           scratch.Poisson_bi_values[q]) *
                          scratch.Poisson_fe_face_values.JxW(q));
                } // for i
            } // for q
        } // end if
    } // end for face_no

}	// end assemble_local_Poisson_rhs

template <int dim>
void
DriftDiffusionPoisson<dim>::
copy_local_to_global_Poisson_rhs(
    const Assembly::Poisson::CopyData<dim> & data)
{
    // copy the local RHS into the global RHS for Poisson
    Poisson_constraints.distribute_local_to_global(data.local_rhs,
            data.local_dof_indices,
            Poisson_system_rhs);
}


template <int dim>
void
DriftDiffusionPoisson<dim>::
solve_Poisson_system()
{
    // proform solve
    Poisson_solver.vmult(Poisson_solution, Poisson_system_rhs);

    // distribute hanging node constraints
    Poisson_constraints.distribute(Poisson_solution);
}




////////////////////////////////////////////////////////////////////////////////////////
// LDG MEMBER FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////

template <int dim>
void
DriftDiffusionPoisson<dim>::
assemble_carrier_matrix()
{

    // use unit scaling values
//		parameters.scaled_electron_mobility = 1.0;
//		parameters.scaled_hole_mobility = 1.0;

    ///////////////////////////////////////////////////////////////
    // Parallel assemble them mass matrix					
    ///////////////////////////////////////////////////////////////
    WorkStream::run(carrier_dof_handler.begin_active(),
                    carrier_dof_handler.end(),
                    std_cxx11::bind(&LDG<dim>::
                                    assemble_local_LDG_mass_matrix,
                                    Assembler_LDG, // the Assembler object
                                    std_cxx11::_1,
                                    std_cxx11::_2,
                                    std_cxx11::_3,
                                    parameters.delta_t),
                    std_cxx11::bind(&DriftDiffusionPoisson<dim>::
                                    copy_local_to_global_LDG_mass_matrix,
                                    this, 			// tthis object
                                    std_cxx11::_1),
                    Assembly::AssemblyScratch<dim>(Poisson_fe,
                            carrier_fe,
                            QGauss<dim>(degree+2),
                            QGauss<dim-1>(degree+2)),
                    Assembly::DriftDiffusion::CopyData<dim>(carrier_fe)
                   );


    //////////////////////////////////////////////////////////////////////////
    // Sequential Assembly of LDG flux matrices over all the interior
    // faces
    //////////////////////////////////////////////////////////////////////////
    Assembly::AssemblyScratch<dim>				scratch(Poisson_fe,
            carrier_fe,
            QGauss<dim>(degree+2),
            QGauss<dim-1>(degree+2));


    Assembly::DriftDiffusion::CopyData<dim>			data(carrier_fe);

    typename DoFHandler<dim>::active_cell_iterator
    cell = carrier_dof_handler.begin_active(),
    endc = carrier_dof_handler.end();

    // loop over all cells
    for(; cell != endc; cell++)
    {
        // get the map for the local dofs to global dofs for this cell
        cell->get_dof_indices(data.local_dof_indices);

        // loop over all the faces of this cell to calculate
        // the local flux matrices corresponding to the LDG central fluxes
        for(unsigned int face_no=0;
                face_no< GeometryInfo<dim>::faces_per_cell;
                face_no++)
        {
            // get the face_no-th face of this cell
            typename DoFHandler<dim>::face_iterator 	face = cell->face(face_no);

            // make sure that this face is an interior face
            if( !(face->at_boundary()) )
            {
                // now we are on the interior face elements and we want to make
                // sure that the neighbor cell to this cell is a valid cell
                Assert(cell->neighbor(face_no).state() == IteratorState::valid,
                       ExcInternalError());

                // get the neighbor cell that is adjacent to this cell's face
                typename DoFHandler<dim>::cell_iterator neighbor =
                    cell->neighbor(face_no);

                // if this face has children (more refined faces) then
                // the neighbor cell to this cell is more refined than
                // this cell is so we have to deal with that case
                if(face->has_children())
                {
                    // get the face such that
                    // neighbor->face(neighbor_face_no) = cell->face(face_no)
                    const unsigned int neighbor_face_no =
                        cell->neighbor_of_neighbor(face_no);

                    // loop over all the subfaces of this face
                    for(unsigned int subface_no=0;
                            subface_no < face->number_of_children();
                            subface_no++)
                    {
                        // get the refined neighbor cell that matches this
                        // face and subface number
                        typename DoFHandler<dim>::cell_iterator	neighbor_child =
                            cell->neighbor_child_on_subface(face_no, subface_no);

                        // parent cant be more than one refinement level above
                        // the child
                        Assert(!neighbor_child->has_children(), ExcInternalError());

                        // reinitialize the fe_subface_values to this cell's subface and
                        // neighbor_childs fe_face_values to its face
                        scratch.carrier_fe_subface_values.reinit(cell, face_no, subface_no);
                        scratch.carrier_fe_neighbor_face_values.reinit(neighbor_child,
                                neighbor_face_no);

                        // get the map for the local dofs to global dofs for the neighbor
                        neighbor_child->get_dof_indices(data.local_neighbor_dof_indices);

                        Assembler_LDG.assemble_local_child_flux_terms(scratch,
                                data,
                                parameters.penalty);

                        // now add the local ldg flux matrices to the global one
                        // for the electrons and holes.

                        // NOTE: There are the same only the A matrix will be changed by
                        // scaling

                        // loop over all the test function dofs
                        for(unsigned int i=0;
                                i < scratch.carrier_fe_values.dofs_per_cell; i++)
                        {
                            // loop over all the trial function dofs
                            for(unsigned int j=0;
                                    j < scratch.carrier_fe_values.dofs_per_cell; j++)
                            {
                                // note: not sure how to do this with constraints matrix....
                                electron_system_matrix.add(data.local_dof_indices[i],
                                                           data.local_dof_indices[j],
                                                           data.vi_ui_matrix(i,j));

                                electron_system_matrix.add(data.local_dof_indices[i],
                                                           data.local_neighbor_dof_indices[j],
                                                           data.vi_ue_matrix(i,j));

                                electron_system_matrix.add(data.local_neighbor_dof_indices[i],
                                                           data.local_dof_indices[j],
                                                           data.ve_ui_matrix(i,j));

                                electron_system_matrix.add(data.local_neighbor_dof_indices[i],
                                                           data.local_neighbor_dof_indices[j],
                                                           data.ve_ue_matrix(i,j));
                            } // for i

                            // loop over all the trial function dofs
                            for(unsigned int j=0;
                                    j < scratch.carrier_fe_values.dofs_per_cell; j++)
                            {
                                // note: not sure how to do this with constraints matrix....
                                hole_system_matrix.add(data.local_dof_indices[i],
                                                       data.local_dof_indices[j],
                                                       data.vi_ui_matrix(i,j));

                                hole_system_matrix.add(data.local_dof_indices[i],
                                                       data.local_neighbor_dof_indices[j],
                                                       data.vi_ue_matrix(i,j));

                                hole_system_matrix.add(data.local_neighbor_dof_indices[i],
                                                       data.local_dof_indices[j],
                                                       data.ve_ui_matrix(i,j));

                                hole_system_matrix.add(data.local_neighbor_dof_indices[i],
                                                       data.local_neighbor_dof_indices[j],
                                                       data.ve_ue_matrix(i,j));
                            } // for i
                        } // for j


                    } // for subface_no
                } // if face has children
                else
                {
                    // we now know that the neighbor cell of this cell's face
                    // is on the the same refinement level and therefore
                    // cell with the lower index does the work
                    if((neighbor->level() == cell->level()) &&
                            (neighbor->index() > cell->index()) )
                    {
                        // get the face of the nighbor such that
                        // neighbor->face(neighbor_face_no) = cell->face(face_no)
                        const unsigned int neighbor_face_no =
                            cell->neighbor_of_neighbor(face_no);

                        // reinitialize the fe_face_values on their respective face
                        scratch.carrier_fe_face_values.reinit(cell, face_no);
                        scratch.carrier_fe_neighbor_face_values.reinit(neighbor,
                                neighbor_face_no);

                        // get the map for the local dofs to global dofs for the neighbor
                        neighbor->get_dof_indices(data.local_neighbor_dof_indices);

                        // assmble the local LDG flux matrices for this face using
                        // the assemblr object
                        Assembler_LDG.assemble_local_flux_terms(scratch,
                                                                data,
                                                                parameters.penalty);

                        // now add the local ldg flux matrices to the global one
                        // for the electrons and holes.

                        // NOTE: There are the same only the A matrix will be changed by
                        // scaling

                        // loop over all the test function dofs
                        for(unsigned int i=0;
                                i < scratch.carrier_fe_values.dofs_per_cell; i++)
                        {
                            // loop over all the trial function dofs
                            for(unsigned int j=0;
                                    j < scratch.carrier_fe_values.dofs_per_cell; j++)
                            {
                                // note: not sure how to do this with constraints matrix....
                                electron_system_matrix.add(data.local_dof_indices[i],
                                                           data.local_dof_indices[j],
                                                           data.vi_ui_matrix(i,j));

                                electron_system_matrix.add(data.local_dof_indices[i],
                                                           data.local_neighbor_dof_indices[j],
                                                           data.vi_ue_matrix(i,j));

                                electron_system_matrix.add(data.local_neighbor_dof_indices[i],
                                                           data.local_dof_indices[j],
                                                           data.ve_ui_matrix(i,j));

                                electron_system_matrix.add(data.local_neighbor_dof_indices[i],
                                                           data.local_neighbor_dof_indices[j],
                                                           data.ve_ue_matrix(i,j));
                            } // for i

                            // loop over all the trial function dofs
                            for(unsigned int j=0;
                                    j < scratch.carrier_fe_values.dofs_per_cell; j++)
                            {
                                // note: not sure how to do this with constraints matrix....
                                hole_system_matrix.add(data.local_dof_indices[i],
                                                       data.local_dof_indices[j],
                                                       data.vi_ui_matrix(i,j));

                                hole_system_matrix.add(data.local_dof_indices[i],
                                                       data.local_neighbor_dof_indices[j],
                                                       data.vi_ue_matrix(i,j));

                                hole_system_matrix.add(data.local_neighbor_dof_indices[i],
                                                       data.local_dof_indices[j],
                                                       data.ve_ui_matrix(i,j));

                                hole_system_matrix.add(data.local_neighbor_dof_indices[i],
                                                       data.local_neighbor_dof_indices[j],
                                                       data.ve_ue_matrix(i,j));
                            } // for i
                        } // for j
                    }	// end if index() >
                } // else cell not have children
            } // end if interior
        }	// end face_no

    } // for cell

    ///////////////////////////////////////////////////////////////
    // parallel assemble the cell integrals and bc flux matrices //
    // for electrons 											 //
    ///////////////////////////////////////////////////////////////
    WorkStream::run(carrier_dof_handler.begin_active(),
                    carrier_dof_handler.end(),
                    std_cxx11::bind(&LDG<dim>::
                                    assemble_local_LDG_cell_and_bc_terms,
                                    Assembler_LDG, // Assembler object
                                    std_cxx11::_1,
                                    std_cxx11::_2,
                                    std_cxx11::_3,
                                    parameters.scaled_electron_mobility,
                                    parameters.delta_t),
                    std_cxx11::bind(&DriftDiffusionPoisson<dim>::
                                    copy_local_to_global_electron_LDG_matrix,
                                    this, 		 // this object
                                    std_cxx11::_1),
                    Assembly::AssemblyScratch<dim>(Poisson_fe,
                            carrier_fe,
                            QGauss<dim>(degree+2),
                            QGauss<dim-1>(degree+2)),
                    Assembly::DriftDiffusion::CopyData<dim>(carrier_fe)
                   );


    ///////////////////////////////////////////////////////////////
    // parallel assemble the cell integrals and bc flux matrices //
    // for hole													//
    ///////////////////////////////////////////////////////////////
    WorkStream::run(carrier_dof_handler.begin_active(),
                    carrier_dof_handler.end(),
                    std_cxx11::bind(&LDG<dim>::
                                    assemble_local_LDG_cell_and_bc_terms,
                                    Assembler_LDG,	// Assembler object
                                    std_cxx11::_1,
                                    std_cxx11::_2,
                                    std_cxx11::_3,
                                    parameters.scaled_hole_mobility,
                                    parameters.delta_t),
                    std_cxx11::bind(&DriftDiffusionPoisson<dim>::
                                    copy_local_to_global_hole_LDG_matrix,
                                    this,				// this object
                                    std_cxx11::_1),
                    Assembly::AssemblyScratch<dim>(Poisson_fe,
                            carrier_fe,
                            QGauss<dim>(degree+2),
                            QGauss<dim-1>(degree+2)),
                    Assembly::DriftDiffusion::CopyData<dim>(carrier_fe)
                   );

} // end assemble system



// NOTE: These are all seperate since accessing a sparse matrix is expensive

template<int dim>
void
DriftDiffusionPoisson<dim>::
copy_local_to_global_LDG_mass_matrix(
    const Assembly::DriftDiffusion::CopyData<dim>				 & data)
{
    // copy local mass matrix into the global mass matrix..
    for(unsigned int i=0; i<data.local_dof_indices.size(); i++)
        for(unsigned int j=0; j<data.local_dof_indices.size(); j++)
            carrier_mass_matrix.add(data.local_dof_indices[i],
                                    data.local_dof_indices[j],
                                    data.local_mass_matrix(i,j) );
}


template<int dim>
void
DriftDiffusionPoisson<dim>::
copy_local_to_global_electron_LDG_matrix(
    const Assembly::DriftDiffusion::CopyData<dim>				& data)
{
    // places the local matrix in global electron_system_matrix
    for(unsigned int i=0; i<data.local_dof_indices.size(); i++)
        for(unsigned int j=0; j<data.local_dof_indices.size(); j++)
            electron_system_matrix.add(data.local_dof_indices[i],
                                       data.local_dof_indices[j],
                                       data.local_matrix(i,j) );
}

template<int dim>
void
DriftDiffusionPoisson<dim>::
copy_local_to_global_hole_LDG_matrix(
    const Assembly::DriftDiffusion::CopyData<dim>			 & data)
{
    // places the local matrix in global hole_system_matrix
    for(unsigned int i=0; i<data.local_dof_indices.size(); i++)
        for(unsigned int j=0; j<data.local_dof_indices.size(); j++)
            hole_system_matrix.add(data.local_dof_indices[i],
                                   data.local_dof_indices[j],
                                   data.local_matrix(i,j) );
}


template <int dim>
void
DriftDiffusionPoisson<dim>::
assemble_carrier_rhs()
{
    // set carrier_system_rhs = M * u^{n-1}
    carrier_mass_matrix.vmult(electron_system_rhs, old_electron_solution);
    carrier_mass_matrix.vmult(hole_system_rhs, old_hole_solution);

    // NOTE:  In future maybe break these up into computing
    // 1.) Generation/Recombination
    // 2.) Drift term
    // 3.) Boundary condition terms
    // so that each can be done individually and save some time

    // 11/4/2015: NO! Dont listen to old mike, the way you have it
    // here is great. Otherwise you will be runnning over the mesh
    // more times than needed and use up more memory.

    // This needs the poisson_dof_handler so this function is
    // not done in the assembler

    ///////////////////////////////////////////////////////////////
    // Parallel assemble the electron and hole rhs term					 //
    ///////////////////////////////////////////////////////////////
    WorkStream::run(carrier_dof_handler.begin_active(),
                    carrier_dof_handler.end(),
                    *this, // this object
                    &DriftDiffusionPoisson::assemble_local_LDG_rhs,
                    &DriftDiffusionPoisson::copy_local_to_global_LDG_rhs,
                    Assembly::AssemblyScratch<dim>(Poisson_fe,
                            carrier_fe,
                            QGauss<dim>(degree+2),
                            QGauss<dim-1>(degree+2)),
                    Assembly::DriftDiffusion::CopyData<dim>(carrier_fe)
                   );


} // end assemble_hole_rhs

template<int dim>
void
DriftDiffusionPoisson<dim>::
assemble_local_LDG_rhs(
    const typename DoFHandler<dim>::active_cell_iterator & cell,
    Assembly::AssemblyScratch<dim>									 & scratch,
    Assembly::DriftDiffusion::CopyData<dim>							 & data)
{

    // this assembles the drift term in the ldg formulation.  it uses the electric field
    // at the current iteration and the density of the carrier at the previous time step
    const unsigned int dofs_per_cell			 =
        scratch.carrier_fe_values.dofs_per_cell;
    const unsigned int n_q_points					 =
        scratch.carrier_fe_values.n_quadrature_points;
    const unsigned int n_face_q_points		 =
        scratch.carrier_fe_face_values.n_quadrature_points;;


    cell->get_dof_indices(data.local_dof_indices);

    //////////////////////////////////////////////////////////////////////////////
    // NOTE: Since we are working with both Poisson's fe_value and the
    // carriers fe_value we must have iterators over the cells held by
    // each of the dof_handlers.  While we do this, the actual cell that we are
    // working with will always be the same.
    ////////////////////////////////////////////////////////////////////////////

    typename DoFHandler<dim>::active_cell_iterator
    Poisson_cell(&triangulation,
                 cell->level(),
                 cell->index(),
                 &Poisson_dof_handler);

    // reinitialize the fe_values
    scratch.carrier_fe_values.reinit(cell);
    scratch.Poisson_fe_values.reinit(Poisson_cell);

    // reset the local_rhs to be zero
    data.local_electron_rhs=0;
    data.local_hole_rhs=0;

    const FEValuesExtractors::Vector Current(0);
    const FEValuesExtractors::Scalar Density(dim);

    const FEValuesExtractors::Vector ElectricField(0);

    // get the values of the electron and hole densities at the previous time step
    // previous time step
    scratch.carrier_fe_values[Density].get_function_values(
        old_electron_solution,
        scratch.old_electron_density_values);

    scratch.carrier_fe_values[Density].get_function_values(
        old_hole_solution,
        scratch.old_hole_density_values);

    // get the values of the solution to poissons equation at the previous
    // time step
    scratch.Poisson_fe_values[ElectricField].get_function_values(Poisson_solution,
            scratch.electric_field_values);

    // get the recombination/generation values at the quadrature points in
    // this cell
    generation_function.value_list(scratch.carrier_fe_values.get_quadrature_points(),
                                   scratch.generation_values);

    // the test version
//		Assembler_LDG.test_carrier_rhs.value_list(
//												scratch.carrier_fe_values.get_quadrature_points(),
//												scratch.generation_values);

    // loop over all the quadrature points in this cell and compute body integrals
    for(unsigned int q=0; q<n_q_points; q++)
    {
        // loop over all the test function dofs and get the test functions
        for(unsigned int i=0; i<dofs_per_cell; i++)
        {
            const double					psi_i_density	=
                scratch.carrier_fe_values[Density].value(i,q);

            const Tensor<1,dim>		psi_i_field	=
                scratch.carrier_fe_values[Current].value(i,q);

            // contribution from RHS function + Drift
            // int_{Omega} v * R dx
            data.local_electron_rhs(i) += (
                                        (psi_i_density * scratch.generation_values[q])
                                        +
                                        (psi_i_density *
                                        SRH_Recombination(scratch.old_electron_density_values[q],
                                                       scratch.old_hole_density_values[q],
                                                       parameters))
                                        -
                                        (psi_i_field *
                                        scratch.electric_field_values[q] *
                                        scratch.old_electron_density_values[q])
                                        ) *
                                        scratch.carrier_fe_values.JxW(q);

            data.local_hole_rhs(i) += (
                                      (psi_i_density * scratch.generation_values[q])
                                      +
                                      (psi_i_density *
                                      SRH_Recombination(scratch.old_electron_density_values[q],
                                                   scratch.old_hole_density_values[q],
                                                   parameters))
                                       +
                                      (psi_i_field *
                                      scratch.electric_field_values[q] *
                                      scratch.old_hole_density_values[q])
                                      ) *
                                      scratch.carrier_fe_values.JxW(q);


        } // for i
    }	// for q

    // loop over all the faces of this cell and compute the contribution from the
    // boundary conditions
    for(unsigned int face_no=0;
            face_no< GeometryInfo<dim>::faces_per_cell;
            face_no++)
    {
        // get the face_no-th face of this cell
        typename DoFHandler<dim>::face_iterator 	face = cell->face(face_no);

        // if on boundary apply boundayr conditions
        if(face->at_boundary() )
        {
            if(face->boundary_id() == Dirichlet)
            {
                // reinitialize the fe_face_values for this cell ONLY if it is as the
                //boundary otherwise its a waste.  then assemble the appropriate
                //boundary conditions
                scratch.carrier_fe_face_values.reinit(cell, face_no);


                donor_function.value_list(
                    scratch.carrier_fe_face_values.get_quadrature_points(),
                    scratch.electron_bc_values,
                    dim); // calls the density values of the donor profile
                // not the current ones

                acceptor_function.value_list(
                    scratch.carrier_fe_face_values.get_quadrature_points(),
                    scratch.hole_bc_values,
                    dim); // calls the density values of the donor profile
                // not the current ones


                // loop over all the quadrature points on this face
                for(unsigned int q=0; q<n_face_q_points; q++)
                {
                    // loop over all the test function dofs on this face
                    for(unsigned int i=0; i<dofs_per_cell; i++)
                    {
                        // get the test function
                        const Tensor<1, dim>  psi_i_field	=
                            scratch.carrier_fe_face_values[Current].value(i,q);

                        // int_{\Gamma_{D}} -p^{-} n^{-} u_{D} ds
                        data.local_electron_rhs(i) +=
                            -1.0 * psi_i_field *
                            scratch.carrier_fe_face_values.normal_vector(q) *
                            scratch.electron_bc_values[q] *
                            scratch.carrier_fe_face_values.JxW(q);

                        data.local_hole_rhs(i) +=
                            -1.0 * psi_i_field *
                            scratch.carrier_fe_face_values.normal_vector(q) *
                            scratch.hole_bc_values[q] *
                            scratch.carrier_fe_face_values.JxW(q);


                    } // for i
                }	// for q
            } // end Dirichlet
            else if(face->boundary_id() == Neumann)
            {
                // NOTHIN TO DO IF INSULATING
            }
            else
                Assert(false, ExcNotImplemented() );
        } // end at boundary
    } // end for face_no

}	// end assemble_local_rhs

template<int dim>
void
DriftDiffusionPoisson<dim>::
copy_local_to_global_LDG_rhs(
    const Assembly::DriftDiffusion::CopyData<dim> & data)
{
    for(unsigned int i=0; i<data.local_dof_indices.size(); i++)
        electron_system_rhs(data.local_dof_indices[i]) += data.local_electron_rhs(i);

    for(unsigned int i=0; i<data.local_dof_indices.size(); i++)
        hole_system_rhs(data.local_dof_indices[i]) += data.local_hole_rhs(i);
}	// copy_local_to_global_DD_drift_term

template<int dim>
void
DriftDiffusionPoisson<dim>::
solve_electron_system()
{
    electron_solver.vmult(electron_solution, electron_system_rhs);
    old_electron_solution = electron_solution;
}

template<int dim>
void
DriftDiffusionPoisson<dim>::
solve_hole_system()
{
    hole_solver.vmult(hole_solution, hole_system_rhs);
    old_hole_solution = hole_solution;
}


template<int dim>
void
DriftDiffusionPoisson<dim>::
project_back_densities()
{
    //TODO:  THIS FAILS BECAUSE NEED TO LOOP OVER ELEMENTS IN SOLUTION.BLOCK(1)
    //			 NOT OVER SOLUTION

    // loops through the solution vectors and if the densities are negative
    //  projects them to be zero
    for(unsigned int i=0; i<electron_solution.size(); i++)
    {
        if(electron_solution(i) < 0)
            electron_solution(i) = 0.0;

        if(hole_solution(i) < 0 )
            hole_solution(i) = 0.0;
    }
}

////////////////////////////////////////////////////////////////////////////////
// OUTPUTTING, RUNNING, ETC
///////////////////////////////////////////////////////////////////////////////

template<int dim>
void
DriftDiffusionPoisson<dim>::
output_results()	const
{
    std::vector<std::string> Poisson_solution_names;
    switch(dim)
    {
    case 1:
        Poisson_solution_names.push_back("E_x");
        Poisson_solution_names.push_back("Potential");
        break;

    case 2:
        Poisson_solution_names.push_back("E_x");
        Poisson_solution_names.push_back("E_y");
        Poisson_solution_names.push_back("Potential");
        break;

    case 3:
        Poisson_solution_names.push_back("E_x");
        Poisson_solution_names.push_back("E_y");
        Poisson_solution_names.push_back("E_z");
        Poisson_solution_names.push_back("Potential");
        break;

    default:
        Assert(false, ExcNotImplemented() );
    }

    DataOut<dim> 					data_out;
    data_out.attach_dof_handler(Poisson_dof_handler);

    std::string Poisson_file =	"Poisson_solution-"
                                +	Utilities::int_to_string(time_step_number,3)
                                +  ".vtk";

    std::ofstream output(Poisson_file.c_str() );

    data_out.add_data_vector(Poisson_solution, Poisson_solution_names);
    data_out.build_patches(); //degree+1);
    data_out.write_vtk(output);
    output.close();


    std::vector<std::string> carrier_solution_names;
    switch(dim)
    {
    case 1:
        carrier_solution_names.push_back("J");
        carrier_solution_names.push_back("Density");
        break;

    case 2:
        carrier_solution_names.push_back("J_x");
        carrier_solution_names.push_back("J_y");
        carrier_solution_names.push_back("Density");
        break;

    case 3:
        carrier_solution_names.push_back("J_x");
        carrier_solution_names.push_back("J_y");
        carrier_solution_names.push_back("J_z");
        carrier_solution_names.push_back("Density");
        break;

    default:
        Assert(false, ExcNotImplemented() );
    }

    DataOut<dim>	data_out2;
    data_out2.attach_dof_handler(carrier_dof_handler);
    data_out2.add_data_vector(electron_solution, carrier_solution_names);

    data_out2.build_patches();
    std::string electron_file = "electron_solution-"
                                +
                                Utilities::int_to_string(time_step_number,3)
                                +
                                ".vtk";

    output.open(electron_file.c_str());
    data_out2.write_vtk(output);
    output.close();


    DataOut<dim>	data_out3;
    data_out3.attach_dof_handler(carrier_dof_handler);
    data_out3.add_data_vector(hole_solution, carrier_solution_names);

    data_out3.build_patches();
    std::string hole_file = "hole_solution-"
                            +
                            Utilities::int_to_string(time_step_number,3)
                            +
                            ".vtk";

    output.open(hole_file.c_str());
    data_out3.write_vtk(output);
    output.close();

}

template<int dim>
void DriftDiffusionPoisson<dim>::output_carrier_system() const
{

    std::ofstream output_system("A.mtx");
    electron_system_matrix.print_formatted(output_system);
    output_system.close();

    output_system.open("M.mtx");
    carrier_mass_matrix.print_formatted(output_system);
    output_system.close();

    output_system.open("b.vec");
    electron_system_rhs.print(output_system);
    output_system.close();
}

template <int dim>
void
DriftDiffusionPoisson<dim>::
run()
{
    TimerOutput timer(std::cout,
                      TimerOutput::summary,
                      TimerOutput::wall_times);

    timer.enter_subsection("Make Grid");

    Grid_Maker::Grid<dim> grid_maker;
    grid_maker.make_local_refined_grid(triangulation,
                                       parameters);

    grid_maker.make_Dirichlet_boundaries(triangulation);
//		grid_maker.make_Neumann_boundaries(triangulation);
    timer.leave_subsection("Make Grid");

    timer.enter_subsection("Allocate Memory");
    make_dofs_and_allocate_memory();
    timer.leave_subsection("Allocate Memory");

    timer.enter_subsection("Assemble Poisson Matrix");
    assemble_Poisson_matrix();
    timer.leave_subsection("Assemble Poisson Matrix");

    timer.enter_subsection("Factorize Poisson Matrix");
    Poisson_solver.initialize(Poisson_system_matrix);
    timer.leave_subsection("Factorize Poisson Matrix");


    timer.enter_subsection("Assemble DD Matrix");
    assemble_carrier_matrix();
    timer.leave_subsection("Assemble DD Matrix");

    timer.enter_subsection("Factorize DD Matrix");
    electron_solver.initialize(electron_system_matrix);
    hole_solver.initialize(hole_system_matrix);
    timer.leave_subsection("Factorize DD Matrix");




    // need to have a constraint matrix to do L2 projections
    ConstraintMatrix constraints;
    constraints.close();

    // L2 project the electron initial conditions
    VectorTools::project(carrier_dof_handler,
                         constraints,
                         QGauss<dim>(degree+1),
                         DonorDoping<dim>(),
                         old_electron_solution);


    // L2 project the hole initial conditions
    VectorTools::project(carrier_dof_handler,
                         constraints,
                         QGauss<dim>(degree+1),
                         AcceptorDoping<dim>(),
                         old_hole_solution);


    // need to assign time_step_number before output initial conditions
    double time					=  0.0;
    time_step_number		=  0;

    // output intial conditions
    assemble_Poisson_rhs();
    solve_Poisson_system();

    electron_solution = old_electron_solution;
    hole_solution = old_hole_solution;
//		project_back_densities();
    output_results();

    unsigned int number_outputs = parameters.time_stamps;
    vector<double>			timeStamps(number_outputs);

    for(unsigned int i=0; i<number_outputs; i++)
        timeStamps[i] = (i+1) * parameters.t_end / number_outputs;

    for(time_step_number = 1;
            time_step_number<=number_outputs;
            time_step_number++)
    {
        std::cout << time << std::endl;

        while(time < timeStamps[time_step_number-1])
        {

            timer.enter_subsection("Assemble Poisson RHS");
            assemble_Poisson_rhs();
            timer.leave_subsection("Assemble Poisson RHS");


            timer.enter_subsection("Poisson Solver");
            solve_Poisson_system();
            timer.leave_subsection("Poisson Solver");

            timer.enter_subsection("Assemble electron & hole RHS");
            assemble_carrier_rhs();
            timer.leave_subsection("Assemble electron & hole RHS");


            timer.enter_subsection("electron & hole Solver");
            solve_electron_system();
            solve_hole_system();
            timer.leave_subsection("electron & hole Solver");

//					project_back_densities();

            time += parameters.delta_t;
        } // while

        timer.enter_subsection("print state");
        output_results();
        timer.leave_subsection("print state");

    } // for
}	// end run

} // NAMESPACE LDG_MX




