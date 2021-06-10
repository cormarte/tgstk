/*==========================================================================

  This file is part of the Tumor Growth Simulation ToolKit (TGSTK)
  (<https://github.com/cormarte/TGSTK>, <https://cormarte.github.io/TGSTK>).

  Copyright (C) 2021  Corentin Martens

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <https://www.gnu.org/licenses/>.

  Contact: corentin.martens@ulb.be

==========================================================================*/

#include <tgstk/algorithms/tgstkImageToMeshFilter.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Image_3.h>
#include <CGAL/Labeled_image_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/read_vtk_image_data.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkImageCast.h>
#include <vtkSmartPointer.h>
#include <vtkSparseArray.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>

using namespace CGAL;
using namespace CGAL::parameters;
using namespace CGAL::parameters::internal;
using namespace std;

tgstkImageToMeshFilter::tgstkImageToMeshFilter() {

    this->objectName = "tgstkImageToMeshFilter";

    this->maximumCellSize = 8.0;
    this->maximumFacetDistance = 4.0;
    this->maximumFacetSize = 6.0;
    this->maximumCellRadiusEdgeRatio = 3.0;
    this->minimumFacetAngle = 30.0;
    this->useExude = true;
    this->useLloyd = false;
    this->useODT = false;
    this->usePerturb = true;

    this->inputImage = nullptr;
    this->outputMesh = nullptr;
}

tgstkImageToMeshFilter::~tgstkImageToMeshFilter() {

}

bool tgstkImageToMeshFilter::check() {

    if (!assertNotNullPtr(this->inputImage)) return false;
    if (!assertImageScalarType(0, std::vector<int>({VTK_UNSIGNED_SHORT}))) return false;

    return true;
}

void tgstkImageToMeshFilter::execute() {

     // CGAL types definition

#ifdef CGAL_CONCURRENT_MESH_3
    typedef Parallel_tag ConcurrencyTagType;
#else
    typedef Sequential_tag ConcurrencyTagType;
#endif

    typedef Exact_predicates_inexact_constructions_kernel KernelType;
    typedef Labeled_image_mesh_domain_3<Image_3, KernelType> DomainType;
    typedef Mesh_triangulation_3<DomainType, Default, ConcurrencyTagType>::type TriangulationType;
    typedef TriangulationType::Finite_vertices_iterator VertexIteratorType;
    typedef TriangulationType::Vertex_handle VertexHandleType;
    typedef TriangulationType::Weighted_point PointType;
    typedef Mesh_complex_3_in_triangulation_3<TriangulationType> Complex3InTriangulation3Type;
    typedef Complex3InTriangulation3Type::Cells_in_complex_iterator CellIteratorType;
    typedef Complex3InTriangulation3Type::Facets_in_complex_iterator FacetsIteratorType;
    typedef Mesh_criteria_3<TriangulationType> CriteriaType;


    // Casting

    vtkSmartPointer<vtkImageCast> caster = vtkSmartPointer<vtkImageCast>::New();
    caster->SetOutputScalarType(VTK_UNSIGNED_CHAR);
    caster->SetInputData(this->inputImage);
    caster->Update();


    // VTK image to CGAL image

    Image_3 image = read_vtk_image_data(caster->GetOutput());


    // Domain construction

    DomainType domain(image);


    // Meshing criteria

    CriteriaType criteria(facet_angle = this->minimumFacetAngle,
                          facet_size = this->maximumFacetSize,
                          facet_distance = this->maximumFacetDistance,
                          cell_radius_edge_ratio = this->maximumCellRadiusEdgeRatio,
                          cell_size = this->maximumCellSize);


    // Triangulation

    Lloyd_options lloydOptions = this->useLloyd ? lloyd() : no_lloyd();
    Odt_options odtOptions = this->useODT ? odt() : no_odt();
    Perturb_options perturbOptions = this->usePerturb ? perturb() : no_perturb();
    Exude_options exudeOptions = this->useExude ? exude() : no_exude();

    Complex3InTriangulation3Type complex3InTriangulation3 = make_mesh_3<Complex3InTriangulation3Type>(domain, criteria, lloydOptions, odtOptions, perturbOptions, exudeOptions);
    const TriangulationType& mesh = complex3InTriangulation3.triangulation();


    // Points

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->SetNumberOfPoints(mesh.number_of_vertices());

    map<VertexHandleType, vtkIdType> vertices;
    vtkIdType id = 0;

    for (VertexIteratorType it=mesh.finite_vertices_begin(); it!=mesh.finite_vertices_end(); ++it) {

        const PointType& point = it->point();
        points->SetPoint(id, CGAL::to_double(point.x()), CGAL::to_double(point.y()), CGAL::to_double(point.z()));
        vertices[it] = id;

        id++;
    }


    // Volume

    vtkSmartPointer<vtkCellArray> tetrahedra = vtkSmartPointer<vtkCellArray>::New();
    tetrahedra->Allocate(complex3InTriangulation3.number_of_cells_in_complex());

    vtkSmartPointer<vtkUnsignedCharArray> regions = vtkSmartPointer<vtkUnsignedCharArray>::New();
    regions->SetName("Regions");
    regions->SetNumberOfComponents(1);
    regions->SetNumberOfTuples(complex3InTriangulation3.number_of_cells_in_complex());
    regions->SetNumberOfValues(complex3InTriangulation3.number_of_cells_in_complex() * 1);

    id = 0;

    for (CellIteratorType it=complex3InTriangulation3.cells_in_complex_begin(); it!=complex3InTriangulation3.cells_in_complex_end(); ++it) {

        vtkIdType cell[4];

        for (int i=0; i!=4; ++i) {

            cell[i] = vertices[it->vertex(i)];
        }

        tetrahedra->InsertNextCell(4, cell);
        regions->SetValue(id, it->subdomain_index());

        id++;
    }


    // Boundary

    vtkSmartPointer<vtkUnsignedCharArray> boundaries = vtkSmartPointer<vtkUnsignedCharArray>::New();
    boundaries->SetName("Boundaries");
    boundaries->SetNumberOfComponents(3); // For compatibility reason with FEAMesh
    boundaries->SetNumberOfTuples(points->GetNumberOfPoints());
    boundaries->SetNumberOfValues(points->GetNumberOfPoints() * 3);
    boundaries->FillValue(0);

    for (FacetsIteratorType it=complex3InTriangulation3.facets_in_complex_begin(); it!=complex3InTriangulation3.facets_in_complex_end(); ++it) {

        int j=0;

        for (int i = 0; i < 4; ++i) {

            if (i != it->second) {

                auto boundary = complex3InTriangulation3.surface_patch_index(*it);

                boundaries->SetTuple3(vertices[(*it).first->vertex(i)], boundary.first, boundary.second, 0);
                j++;
            }
        }

        CGAL_assertion(j==3);
    }

    vtkSmartPointer<vtkUnstructuredGrid> volume = vtkSmartPointer<vtkUnstructuredGrid>::New();
    volume->SetPoints(points);
    volume->GetPointData()->AddArray(boundaries);
    volume->SetCells(VTK_TETRA, tetrahedra);
    volume->GetCellData()->AddArray(regions);


    // Origin translation

    double* origin = this->inputImage->GetOrigin();

    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->Identity();
    transform->Translate(origin[0], origin[1], origin[2]);

    vtkSmartPointer<vtkTransformFilter> transformFilter = vtkSmartPointer<vtkTransformFilter>::New();
    transformFilter->SetInputData(volume);
    transformFilter->SetTransform(transform);
    transformFilter->Update();


    // Output

    this->outputMesh = vtkUnstructuredGrid::SafeDownCast(transformFilter->GetOutput());
}

void tgstkImageToMeshFilter::exudeOff() {

    this->useExude = false;
}

void tgstkImageToMeshFilter::exudeOn() {

    this->useExude = true;
}

vtkSmartPointer<vtkUnstructuredGrid> tgstkImageToMeshFilter::getOutputMesh() {

    return this->outputMesh;
}

void tgstkImageToMeshFilter::lloydOff() {

    this->useLloyd = false;
}

void tgstkImageToMeshFilter::lloydOn() {

    this->useLloyd = true;
}

void tgstkImageToMeshFilter::odtOff() {

    this->useODT = false;
}

void tgstkImageToMeshFilter::odtOn() {

    this->useODT = true;
}

void tgstkImageToMeshFilter::perturbOff() {

    this->usePerturb = false;
}

void tgstkImageToMeshFilter::perturbOn() {

    this->usePerturb = true;
}

void tgstkImageToMeshFilter::setInputImage(vtkSmartPointer<vtkImageData> image) {

    this->inputImage = image;
}

void tgstkImageToMeshFilter::setMaximumCellRadiusEdgeRatio(double maximumCellRadiusEdgeRatio) {

    this->maximumCellRadiusEdgeRatio = maximumCellRadiusEdgeRatio;
}

void tgstkImageToMeshFilter::setMaximumCellSize(double maximumCellSize) {

    this->maximumCellSize = maximumCellSize;
}

void tgstkImageToMeshFilter::setMaximumFacetDistance(double maximumFacetDistance) {

    this->maximumFacetDistance = maximumFacetDistance;
}

void tgstkImageToMeshFilter::setMaximumFacetSize(double maximumFacetSize) {

    this->maximumFacetSize = maximumFacetSize;
}

void tgstkImageToMeshFilter::setMinimumFacetAngle(double minimumFacetAngle) {

    this->minimumFacetAngle = minimumFacetAngle;
}

void tgstkImageToMeshFilter::setUseExude(bool useExude) {

    this->useExude = useExude;
}

void tgstkImageToMeshFilter::setUseLloyd(bool useLloyd) {

    this->useLloyd = useLloyd;
}

void tgstkImageToMeshFilter::setUseODT(bool useODT) {

    this->useODT = useODT;
}

void tgstkImageToMeshFilter::setUsePerturb(bool usePerturb) {

    this->usePerturb = usePerturb;
}
