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
 * @class tgstkLinearQuadraticTumourCellSurvivalImageFilter
 *
 * @brief Linear-quadratic model for computing cell survival to radiation
 * therapies.
 *
 * tgstkLinearQuadraticTumourCellSurvivalImageFilter computes a survival
 * normalised tumour cell density field \f$c_s(\bar{r})\f$ from
 * user-provided initial normalised tumour cell density field
 * \f$c_0(\bar{r})\f$ and dose map \f$d(\bar{r})\f$ using the
 * linear-quadratic model \cite rockne_2010 :
 *
 * \f[
 * c_s(\bar{r}) = \mathrm{e}^{-\left(\alpha \, d(\bar{r})+\beta \,
 * d^2(\bar{r})\right)} \, c_0(\bar{r})
 * \f]
 *
 * where \f$\alpha\f$ and \f$\beta\f$ are respectively the linear and
 * quadratic radio-sensitivity coefficients.
 *
 * The initial normalised tumour cell density field \f$c_0(\bar{r})\f$ and
 * the dose map \f$d(\bar{r})\f$ can be specified by the user as
 * <a href=https://vtk.org/doc/nightly/html/classvtkImageData.html>
 * vtkImageData</a> objects. The linear and quadratic radio-sensitivity
 * coefficients \f$\alpha\f$ and \f$\beta\f$ can also be specified.
 *
 * @warning As the CUDA implementation performs in single precision floating
 * point, a loss of precision may occur.
 *
 */

#ifndef TGSTKLINEARQUADRATICTUMOURCELLSURVIVALIMAGEFILTER_H
#define TGSTKLINEARQUADRATICTUMOURCELLSURVIVALIMAGEFILTER_H

#include <tgstk/algorithms/tgstkImageProcessorBase.h>

class TGSTK_EXPORT tgstkLinearQuadraticTumourCellSurvivalImageFilter : public virtual tgstkImageProcessorBase {

    public:

        tgstkLinearQuadraticTumourCellSurvivalImageFilter();
        ~tgstkLinearQuadraticTumourCellSurvivalImageFilter();

        bool check();
        void execute();

        /**
         *
         * Gets the survival normalised tumour cell density image
         * \f$c_s(\bar{r})\f$.
         *
         * The image scalar type is VTK_DOUBLE.
         *
         */
        vtkSmartPointer<vtkImageData> getFinalCellDensityImage();

        /**
         *
         * Sets the linear radio-sensitivity coefficient \f$\alpha\f$ in
         * \f$\text{Gy}^{-1}\f$.
         *
         * The value must be \f$\geq 0\f$.
         *
         */
        void setAlpha(double alpha);

        /**
         *
         * Sets the quadratic radio-sensitivity coefficient \f$\beta\f$ in
         * \f$\text{Gy}^{-2}\f$.
         *
         * The value must be \f$\geq 0\f$.
         *
         */
        void setBeta(double beta);

        /**
         *
         * Sets the tumour dose map image \f$d(\bar{r})\f$ in
         * \f$\text{Gy}\f$.
         *
         * The image scalar type must be VTK_DOUBLE. The image values must
         * be \f$\geq 0\f$.
         *
         */
        void setDoseMapImage(vtkSmartPointer<vtkImageData> image);

        /**
         *
         * Sets the initial normalised tumour cell density image
         * \f$c_0(\bar{r})\f$.
         *
         * The image scalar type must be VTK_DOUBLE. The image values must
         * be in range \f$[0; 1]\f$.
         *
         */
        void setInitialCellDensityImage(vtkSmartPointer<vtkImageData> image);

    private:

        double alpha;
        double beta;

        vtkSmartPointer<vtkImageData> doseMapImage;
        vtkSmartPointer<vtkImageData> finalCellDensityImage;
        vtkSmartPointer<vtkImageData> initialCellDensityImage;
};

#endif // TGSTKLINEARQUADRATICTUMOURCELLSURVIVALIMAGEFILTER_H
