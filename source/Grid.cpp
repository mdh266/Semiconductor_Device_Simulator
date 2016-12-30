#include "Grid.hpp"


/////////////////////////////////////////////////////////////////////////////////
// GRID AND BOUNDARIES
/////////////////////////////////////////////////////////////////////////////////
namespace Grid_Maker
{
using namespace dealii;

template<int dim>
Grid<dim>::
Grid()
{

    Dirichlet = 0;
    Neumann   = 1;
}

template <int dim>
void
Grid<dim>::
make_grid(Triangulation<dim> 						& triangulation,
          ParameterSpace::Parameters 		 & params)
{

    GridGenerator::hyper_cube(triangulation,-1,1);
    triangulation.refine_global(params.n_global_refine);

    params.h_min = GridTools::minimal_cell_diameter(triangulation);
    params.h_max = GridTools::maximal_cell_diameter(triangulation);

    params.penalty = 1.0/params.h_max;

}

template <int dim>
void
Grid<dim>::
make_local_refined_grid(Triangulation<dim> & triangulation,
                        ParameterSpace::Parameters 				& params)
{
    // make the triangulation and refine globally n_refine times
    GridGenerator::hyper_cube(triangulation,0,1);
    triangulation.refine_global(params.n_global_refine);

    // loop over the number of refinements and at each iteration
    // mark the cells which need to be refined and then refine them

    // loop over the number of local refinements
    for(unsigned int i =0; i < params.n_local_refine; i++)
    {
        typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();

        // loop over all the cells and mark which ones need to be refined
        for(; cell != endc; cell++)
        {
//			if(cell->at_boundary())
            if( (cell->center()[1] < 0.6 ) &&
                    (cell->center()[1] > 0.4 ) )
            {
//					if( (cell->center()[1] < 0.3 ) )
//									if((cell->center()[0] > 0.9)  ||
//						 	 (cell->center()[0] < 0.1) )
                {
                    cell->set_refine_flag();
                } // if on x coordinate
            } // if on y coordinate
        } // for cell

        // refine them in an appropriate way such that neighbor cells
        // do not have have more than one refinement level difference
        triangulation.execute_coarsening_and_refinement();
    }

    params.h_min = GridTools::minimal_cell_diameter(triangulation);
    params.h_max = GridTools::maximal_cell_diameter(triangulation);

    params.penalty = 1.0/params.h_max;

}

template<int dim>
void
Grid<dim>::
make_Dirichlet_boundaries(Triangulation<dim> & triangulation)
{
    typename Triangulation<dim>::cell_iterator
    cell = triangulation.begin(),
    endc = triangulation.end();
    // loop over all the cells
    for(; cell != endc; cell++)
    {
        // loop over all the faces of the cell and find which are on the boundary
        for(unsigned int face_no=0;
                face_no < GeometryInfo<dim>::faces_per_cell;
                face_no++)
        {
            if(cell->face(face_no)->at_boundary() )
            {
                //NOTE: Default is 0, which implies Dirichlet
                cell->face(face_no)->set_boundary_id(Dirichlet);
            } // end if on boundary
        } // for face_no
    } // for cell

}


template<int dim>
void
Grid<dim>::
make_Neumann_boundaries(Triangulation<dim> & triangulation)
{
    typename Triangulation<dim>::cell_iterator
    cell = triangulation.begin(),
    endc = triangulation.end();
    // loop over all the cells
    for(; cell != endc; cell++)
    {
        // loop over all the faces of the cell and find which are on the bondary
        for(unsigned int face_no=0;
                face_no < GeometryInfo<dim>::faces_per_cell;
                face_no++)
        {
            if(cell->face(face_no)->at_boundary() )
            {
                // set the portions of the boundary y = 0 and y = +1
                // to be Neumann boundary conditions
                if((cell->face(face_no)->center()[0] == 0) ||
                        (cell->face(face_no)->center()[0] == +1) )
                {
                    // set it to be Neumann boundary condition by setting the boundary
                    // indicator to be 1.  NOTE: Default is 0, which implies Dirichlet
                    cell->face(face_no)->set_boundary_id(Neumann);
                } // end if Neumann segment
            } // end if on boundary
        } // for face_no
    } // for cell

}

template<int dim>
void
Grid<dim>::
print_grid(Triangulation<dim> & triangulation)
{
    std::ofstream out("grid.eps");
    GridOut grid_out;
    grid_out.write_eps(triangulation,out);
    out.close();
}



}
