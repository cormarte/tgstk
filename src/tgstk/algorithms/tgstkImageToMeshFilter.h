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

/**
 *
 * @class tgstkImageToMeshFilter
 *
 * @brief Tetrahedral mesh generation from a multiclass label image.
 *
 * tgstkImageToMeshFilter generates a tetrahedral mesh from a label image
 * with multiple subdomains using 3D restricted Delaunay triangulation with
 * a protecting phase to guarantee an accurate representation of the
 * subdomain boundaries. The filter is based on the
 * <a href=https://doc.cgal.org/latest/Manual/index.html> CGAL</a> library.
 *
 */


#ifndef TGSTKIMAGETOMESHFILTER_H
#define TGSTKIMAGETOMESHFILTER_H

#include <tgstk/tgstkGlobal.h>
#include <tgstk/algorithms/tgstkImageProcessorBase.h>
#include <tgstk/algorithms/tgstkMeshProcessorBase.h>

class TGSTK_EXPORT tgstkImageToMeshFilter : public virtual tgstkImageProcessorBase, public virtual tgstkMeshProcessorBase {

    public:

        tgstkImageToMeshFilter();
        ~tgstkImageToMeshFilter();

        bool check();
        void execute();

        /**
         *
         * Deactivates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#ga0b79c473082a4ec2a45ed1497f3ac873>
         * CGAL sliver exuder</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void exudeOff();

        /**
         *
         * Activates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#ga16694b09a2acc8ab3f26b7d57633ccb0>
         * CGAL sliver exuder</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void exudeOn();

        /**
         *
         * Gets the generated tetrahedral mesh.
         *
         * The mesh is of type
         * <a href=https://https://vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html>
         * vtkUnstructuredGrid</a>.
         *
         */
        vtkSmartPointer<vtkUnstructuredGrid> getOutputMesh();

        /**
         *
         * Deactivates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#ga6abfd3773eeb47d88ce6cffb91f14d2f>
         * CGAL Lloyd smoother</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void lloydOff();

        /**
         *
         * Activates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#gaa7254c80bba62400f43f1e49506b975a>
         * CGAL Lloyd smoother</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void lloydOn();

        /**
         *
         * Deactivates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#ga1d041e8dbde3860cde3c20107225ecb1>
         * CGAL ODT smoother</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void odtOff();

        /**
         *
         * Activates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#gafbecfd22651a08e6812d8a5ad9b49852>
         * CGAL ODT smoother</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void odtOn();

        /**
         *
         * Deactivates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#ga5714d10decd9eecd30997572b785e03b>
         * CGAL sliver perturber</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void perturbOff();

        /**
         *
         * Activates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#gaedf18f7f3c4647ec5fce5d67e435757a>
         * CGAL sliver perturber</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void perturbOn();

        /**
         *
         * Sets the multiclass label image.
         *
         * The image scalar type must be VTK_UNSIGNED_SHORT.
         *
         */
        void setInputImage(vtkSmartPointer<vtkImageData> image);

        /**
         *
         * Sets the
         * <a href=https://doc.cgal.org/latest/Mesh_3/classCGAL_1_1Mesh__criteria__3.html>
         * CGAL cell radius edge ratio mesh criterion</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void setMaximumCellRadiusEdgeRatio(double maximumCellRadiusEdgeRatio);

        /**
         *
         * Sets the
         * <a href=https://doc.cgal.org/latest/Mesh_3/classCGAL_1_1Mesh__criteria__3.html>
         * CGAL cell size mesh criterion</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void setMaximumCellSize(double maximumCellSize);

        /**
         *
         * Sets the
         * <a href=https://doc.cgal.org/latest/Mesh_3/classCGAL_1_1Mesh__criteria__3.html>
         * CGAL facet distance mesh criterion</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void setMaximumFacetDistance(double maximumFacetDistance);

        /**
         *
         * Sets the
         * <a href=https://doc.cgal.org/latest/Mesh_3/classCGAL_1_1Mesh__criteria__3.html>
         * CGAL facet size mesh criterion</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void setMaximumFacetSize(double maximumFacetSize);

        /**
         *
         * Sets the
         * <a href=https://doc.cgal.org/latest/Mesh_3/classCGAL_1_1Mesh__criteria__3.html>
         * CGAL facet angle mesh criterion</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void setMinimumFacetAngle(double minimumFacetAngle);

        /**
         *
         * Activates/deactivates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#ga16694b09a2acc8ab3f26b7d57633ccb0>
         * CGAL sliver exuder</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void setUseExude(bool useExude);

        /**
         *
         * Activates/deactivates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#gaa7254c80bba62400f43f1e49506b975a>
         * CGAL Lloyd smoother</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void setUseLloyd(bool useLloyd);

        /**
         *
         * Activates/deactivates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#gafbecfd22651a08e6812d8a5ad9b49852>
         * CGAL ODT smoother</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void setUseODT(bool useODT);

        /**
         *
         * Activates/deactivates the
         * <a href=https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html#gaedf18f7f3c4647ec5fce5d67e435757a>
         * CGAL sliver perturber</a>.
         *
         * See \cite cgal_2021.
         *
         */
        void setUsePerturb(bool usePerturb);

    private:

        double maximumCellRadiusEdgeRatio;
        double maximumCellSize;
        double maximumFacetDistance;
        double maximumFacetSize;
        double minimumFacetAngle;
        bool useExude;
        bool useLloyd;
        bool useODT;
        bool usePerturb;

        vtkSmartPointer<vtkImageData> inputImage;
        vtkSmartPointer<vtkUnstructuredGrid> outputMesh;
};

#endif // TGSTKIMAGETOMESHFILTER_H
