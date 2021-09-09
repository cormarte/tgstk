import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def array_to_image(array, reference_image):

    data = numpy_to_vtk(np.swapaxes(array, 0, 2).ravel())
    data.SetNumberOfComponents(array.shape[-1])

    image = vtk.vtkImageData()
    image.SetDimensions(array.shape[:-1])
    image.SetOrigin(reference_image.GetOrigin())
    image.SetSpacing(reference_image.GetSpacing())
    image.GetPointData().SetScalars(data)

    return image


def image_to_array(image):

    w, h, d = image.GetDimensions()
    data = image.GetPointData().GetScalars()
    array = np.swapaxes(vtk_to_numpy(data).reshape(d, h, w, -1), 0, 2)

    return array
